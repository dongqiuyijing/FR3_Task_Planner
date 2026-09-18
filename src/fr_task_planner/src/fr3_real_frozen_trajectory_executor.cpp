#include "fr_task_planner/frozen_executor.hpp"
#include "fr_task_planner/inspection_endpoint_candidates.hpp"
#include "fr_task_planner/winner_trajectory_io.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <fairino_msgs/srv/gripper_bridge.hpp>
#include <moveit_msgs/action/move_group.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/joint_constraint.hpp>
#include <moveit_msgs/msg/motion_plan_request.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/planning_scene_components.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_msgs/srv/apply_planning_scene.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

namespace
{
using fr_task_planner::ExecutorConfig;
using fr_task_planner::ExecutorHooks;
using fr_task_planner::ExecutorTrace;
using fr_task_planner::JointSnapshot;
using fr_task_planner::PlanResult;
using fr_task_planner::SendResult;
using fr_task_planner::TrajectorySegmentRecord;
using fr_task_planner::actionTimeoutSec;
using fr_task_planner::collectDeployableSegments;
using fr_task_planner::computeDuration;
using fr_task_planner::defaultTouchLinks;
using fr_task_planner::deployableLogicalOrder;
using fr_task_planner::kArmJoints;
using fr_task_planner::kDefaultGripperService;
using fr_task_planner::kDefaultTrajectoryActionName;
using fr_task_planner::kLogicalCurrentToHome;
using fr_task_planner::kRealRobotConfirmation;
using fr_task_planner::motionAllowed;
using fr_task_planner::readTrajectoryYaml;
using fr_task_planner::remapSegmentByJointName;
using fr_task_planner::runFrozenExecutor;
using fr_task_planner::scaleDeployableTrajectory;
using fr_task_planner::segmentDuration;
using fr_task_planner::writeTrajectoryYaml;

using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
using MoveGroup = moveit_msgs::action::MoveGroup;

std::string getString(const rclcpp::Node::SharedPtr& node, const std::string& name,
                      const std::string& fallback)
{
  if (!node->has_parameter(name))
  {
    return fallback;
  }
  return node->get_parameter(name).as_string();
}

double getDouble(const rclcpp::Node::SharedPtr& node, const std::string& name, double fallback)
{
  if (!node->has_parameter(name))
  {
    return fallback;
  }
  return node->get_parameter(name).as_double();
}

bool getBool(const rclcpp::Node::SharedPtr& node, const std::string& name, bool fallback)
{
  if (!node->has_parameter(name))
  {
    return fallback;
  }
  return node->get_parameter(name).as_bool();
}

int getInt(const rclcpp::Node::SharedPtr& node, const std::string& name, int fallback)
{
  if (!node->has_parameter(name))
  {
    return fallback;
  }
  return static_cast<int>(node->get_parameter(name).as_int());
}

template <typename T>
void declareDefault(const rclcpp::Node::SharedPtr& node, const std::string& name, const T& value)
{
  if (!node->has_parameter(name))
  {
    node->declare_parameter(name, value);
  }
}

std::vector<std::string> getStringArray(const rclcpp::Node::SharedPtr& node, const std::string& name,
                                        const std::vector<std::string>& fallback)
{
  if (!node->has_parameter(name))
  {
    return fallback;
  }
  return node->get_parameter(name).as_string_array();
}

trajectory_msgs::msg::JointTrajectory toJointTrajectory(const TrajectorySegmentRecord& seg,
                                                        const std::string& frame)
{
  trajectory_msgs::msg::JointTrajectory msg;
  msg.header.frame_id = frame;
  msg.joint_names = seg.joint_names;
  for (const auto& pt : seg.points)
  {
    trajectory_msgs::msg::JointTrajectoryPoint out;
    out.positions = pt.positions;
    out.velocities = pt.velocities;
    out.accelerations = pt.accelerations;
    out.time_from_start.sec = pt.sec;
    out.time_from_start.nanosec = pt.nanosec;
    msg.points.push_back(out);
  }
  return msg;
}

TrajectorySegmentRecord fromRobotTrajectory(const moveit_msgs::msg::RobotTrajectory& traj)
{
  TrajectorySegmentRecord seg;
  seg.name = "Current_to_Home_runtime";
  seg.logical_segment = "Current_to_Home_runtime";
  seg.deployable = false;
  seg.runtime_replan_required = true;
  seg.joint_names = traj.joint_trajectory.joint_names;
  for (const auto& pt : traj.joint_trajectory.points)
  {
    fr_task_planner::TrajectoryPointRecord rec;
    rec.positions = pt.positions;
    rec.velocities = pt.velocities;
    rec.accelerations = pt.accelerations;
    rec.sec = pt.time_from_start.sec;
    rec.nanosec = pt.time_from_start.nanosec;
    seg.points.push_back(rec);
  }
  if (!seg.points.empty())
  {
    seg.start_joints = fr_task_planner::extractArmPositions(seg.joint_names, seg.points.front().positions);
    seg.end_joints = fr_task_planner::extractArmPositions(seg.joint_names, seg.points.back().positions);
  }
  seg.duration = fr_task_planner::segmentDuration(seg);
  return seg;
}

class RealFrozenExecutorNode
{
public:
  explicit RealFrozenExecutorNode(const rclcpp::Node::SharedPtr& node) : node_(node)
  {
  }

  int run()
  {
    loadConfig();
    std::string gate_reason;
    const bool motion = motionAllowed(cfg_, gate_reason);

    RCLCPP_INFO(node_->get_logger(), "========== FR3 REAL FROZEN TRAJECTORY EXECUTOR ==========");
    RCLCPP_INFO(node_->get_logger(), "MODE: %s", motion ? "EXECUTE" : "DRY_RUN");
    RCLCPP_INFO(node_->get_logger(), "REAL ROBOT MOTION ENABLED: %s", motion ? "YES" : "NO");
    RCLCPP_INFO(node_->get_logger(), "trajectory: %s", cfg_.trajectory_file.c_str());
    RCLCPP_INFO(node_->get_logger(), "speed scale: %.6f", cfg_.trajectory_speed_scale);
    RCLCPP_INFO(node_->get_logger(), "Current→Home: RUNTIME REPLAN");
    RCLCPP_INFO(node_->get_logger(), "Frozen path: NO REPLAN");
    RCLCPP_INFO(node_->get_logger(), "A/B/C: FIXED STEP12C ENDPOINTS");
    RCLCPP_INFO(node_->get_logger(), "controller: %s", cfg_.controller_name.c_str());
    RCLCPP_INFO(node_->get_logger(), "action: %s", cfg_.trajectory_action_name.c_str());
    RCLCPP_INFO(node_->get_logger(), "controller config source: %s",
                getString(node_, "controller_config_source", "parameter/default").c_str());
    RCLCPP_INFO(node_->get_logger(), "gripper: %s", cfg_.gripper_service_name.c_str());
    RCLCPP_INFO(node_->get_logger(), "gripper config source: %s",
                getString(node_, "gripper_config_source", "parameter/default").c_str());
    RCLCPP_INFO(node_->get_logger(), "motion gate: %s", gate_reason.c_str());
    if (motion)
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      RCLCPP_ERROR(node_->get_logger(), "REAL ROBOT MOTION IS ENABLED");
      RCLCPP_ERROR(node_->get_logger(),
                   "This will move the FAIRINO FR3 arm and close the real gripper.");
      RCLCPP_ERROR(node_->get_logger(),
                   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    }

    std::string error;
    if (!readTrajectoryYaml(cfg_.trajectory_file, traj_, error))
    {
      RCLCPP_ERROR(node_->get_logger(), "failed to read frozen trajectory: %s", error.c_str());
      return 1;
    }

    auto deployable = collectDeployableSegments(traj_, error);
    if (deployable.empty())
    {
      RCLCPP_ERROR(node_->get_logger(), "frozen trajectory rejected: %s", error.c_str());
      return 1;
    }
    std::vector<TrajectorySegmentRecord> scaled;
    if (!scaleDeployableTrajectory(traj_, cfg_.trajectory_speed_scale, scaled, error))
    {
      RCLCPP_ERROR(node_->get_logger(), "scale failed: %s", error.c_str());
      return 1;
    }
    const double original = computeDuration(deployable, false);
    double scaled_duration = 0.0;
    size_t points = 0;
    RCLCPP_INFO(node_->get_logger(), "deployable segments:");
    for (size_t i = 0; i < deployable.size(); ++i)
    {
      points += deployable[i].points.size();
      scaled_duration += segmentDuration(scaled[i]);
      RCLCPP_INFO(node_->get_logger(),
                  "  %s points=%zu original=%.9f s scaled=%.9f s",
                  deployable[i].logical_segment.c_str(), deployable[i].points.size(),
                  segmentDuration(deployable[i]), segmentDuration(scaled[i]));
    }
    RCLCPP_INFO(node_->get_logger(), "fixed duration original: %.9f s", original);
    RCLCPP_INFO(node_->get_logger(), "fixed duration scaled: %.9f s", scaled_duration);
    RCLCPP_INFO(node_->get_logger(), "fixed points: %zu", points);
    RCLCPP_INFO(node_->get_logger(),
                "DRY-RUN GRIPPER EVENT: REAL_GRIPPER_CLOSE service=%s command=move pos=%d "
                "after PreGrasp_to_Grasp. SENT=%s",
                cfg_.gripper_service_name.c_str(), cfg_.gripper_close_position,
                motion ? "pending-gate" : "NO");
    RCLCPP_INFO(node_->get_logger(),
                "DRY-RUN ATTACH EVENT: %s -> %s after real gripper close. SENT=%s",
                cfg_.object_id.c_str(), cfg_.attach_link.c_str(), motion ? "pending-gate" : "NO");
    RCLCPP_INFO(node_->get_logger(),
                "DRY-RUN RESTORE TABLE COLLISION after Grasp_to_Lift. SENT=%s",
                motion ? "pending-gate" : "NO");

    if (getBool(node_, "write_scaled_debug_yaml", true))
    {
      auto debug = traj_;
      debug.segments = scaled;
      debug.real_execution_speed_scaling = std::to_string(cfg_.trajectory_speed_scale);
      const std::string path =
          getString(node_, "scaled_debug_yaml", "/tmp/fr3_step13_scaled_trajectory.yaml");
      std::string write_error;
      if (!writeTrajectoryYaml(path, debug, write_error))
      {
        RCLCPP_WARN(node_->get_logger(), "could not write scaled debug yaml: %s",
                    write_error.c_str());
      }
      else
      {
        RCLCPP_INFO(node_->get_logger(), "wrote scaled debug yaml (not frozen file): %s",
                    path.c_str());
      }
    }

    joint_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::JointState::SharedPtr msg) { onJointState(msg); });
    display_pub_ = node_->create_publisher<moveit_msgs::msg::DisplayTrajectory>(
        "/display_planned_path", rclcpp::QoS(1).transient_local());
    attach_pub_ = node_->create_publisher<moveit_msgs::msg::AttachedCollisionObject>(
        "/attached_collision_object", 10);
    apply_scene_ = node_->create_client<moveit_msgs::srv::ApplyPlanningScene>("/apply_planning_scene");
    get_scene_ = node_->create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");
    gripper_client_ =
        node_->create_client<fairino_msgs::srv::GripperBridge>(cfg_.gripper_service_name);
    traj_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
        node_, cfg_.trajectory_action_name);
    move_client_ = rclcpp_action::create_client<MoveGroup>(node_, "move_action");

    publishFrozenPreview(scaled);
    inspectLive(motion);

    ExecutorHooks hooks;
    hooks.log = [this](const std::string& text) {
      RCLCPP_INFO(node_->get_logger(), "%s", text.c_str());
    };
    hooks.readJoints = [this]() { return currentSnapshot(); };
    hooks.useSimTimeInvalid = [this]() { return node_->get_parameter("use_sim_time").as_bool(); };
    hooks.gazeboDetected = [this]() { return gazebo_detected_; };
    hooks.controllerReady = [this]() { return controllerReadyLive(); };
    hooks.planCurrentToHome = [this]() { return planCurrentToHome(); };
    hooks.sendSegment = [this](const std::string& logical, const TrajectorySegmentRecord& seg) {
      return sendSegment(logical, seg);
    };
    hooks.closeGripper = [this]() { return closeGripper(); };
    hooks.attachObject = [this]() { return attachObject(); };
    hooks.restoreTableCollision = [this]() { return restoreTableCollision(); };

    const auto trace = runFrozenExecutor(cfg_, traj_, hooks);
    RCLCPP_INFO(node_->get_logger(), "executor finished ok=%s dry_run=%s aborted=%s reason=%s",
                trace.ok ? "true" : "false", trace.dry_run ? "true" : "false",
                trace.aborted ? "true" : "false", trace.abort_reason.c_str());
    RCLCPP_INFO(node_->get_logger(), "commands sent to robot: %d", motion_commands_sent_);
    RCLCPP_INFO(node_->get_logger(), "commands sent to gripper: %d", gripper_commands_sent_);
    if (!motion && (motion_commands_sent_ != 0 || gripper_commands_sent_ != 0))
    {
      RCLCPP_ERROR(node_->get_logger(), "dry-run leaked a motion/gripper command");
      return 2;
    }
    return trace.ok ? 0 : 1;
  }

private:
  void loadConfig()
  {
    cfg_.execute = getBool(node_, "execute", false);
    cfg_.real_robot_confirmation = getString(node_, "real_robot_confirmation", "");
    cfg_.plan_current_to_home = getBool(node_, "plan_current_to_home", false);
    cfg_.gripper_enabled = getBool(node_, "gripper_enabled", true);
    cfg_.trajectory_speed_scale = getDouble(node_, "trajectory_speed_scale", 0.05);
    cfg_.home_tolerance_rad = getDouble(node_, "home_tolerance_rad", 0.02);
    cfg_.start_state_tolerance_rad = getDouble(node_, "start_state_tolerance_rad", 0.02);
    cfg_.segment_start_tolerance_rad = getDouble(node_, "segment_start_tolerance_rad", 0.02);
    cfg_.segment_end_tolerance_rad = getDouble(node_, "segment_end_tolerance_rad", 0.02);
    cfg_.joint_settle_timeout_sec = getDouble(node_, "joint_settle_timeout_sec", 10.0);
    cfg_.joint_settle_poll_period_sec = getDouble(node_, "joint_settle_poll_period_sec", 0.10);
    cfg_.joint_settle_required_samples =
        getInt(node_, "joint_settle_required_samples", 3);
    cfg_.joint_state_max_age_sec = getDouble(node_, "joint_state_max_age_sec", 0.5);
    cfg_.timeout_factor = getDouble(node_, "timeout_factor", 1.5);
    cfg_.timeout_margin_sec = getDouble(node_, "timeout_margin_sec", 10.0);
    cfg_.gripper_timeout_sec = getDouble(node_, "gripper_timeout_sec", 15.0);
    cfg_.gripper_post_close_wait_sec = getDouble(node_, "gripper_post_close_wait_sec", 0.5);
    cfg_.trajectory_file = getString(node_, "trajectory_file", "");
    cfg_.controller_name = getString(node_, "controller_name", "fairino3_controller");
    cfg_.trajectory_action_name =
        getString(node_, "trajectory_action_name", kDefaultTrajectoryActionName);
    cfg_.gripper_service_name = getString(node_, "gripper_service_name", kDefaultGripperService);
    cfg_.planning_group = getString(node_, "planning_group", "fairino3_v6_group");
    cfg_.attach_link = getString(node_, "attach_link", "gripper_tcp");
    cfg_.object_id = getString(node_, "object_name", "small_part");
    cfg_.table_name = getString(node_, "table_name", "table");
    cfg_.controller_joint_names =
        getStringArray(node_, "controller_joint_names",
                       std::vector<std::string>(kArmJoints.begin(), kArmJoints.end()));
    cfg_.touch_links = getStringArray(node_, "touch_links", defaultTouchLinks());
    cfg_.gripper_id = getInt(node_, "gripper_id", 1);
    cfg_.gripper_close_position = getInt(node_, "gripper_close_position", 85);
    cfg_.gripper_velocity = getInt(node_, "gripper_velocity", 20);
    cfg_.gripper_force = getInt(node_, "gripper_force", 20);
    cfg_.gripper_max_time_ms = getInt(node_, "gripper_max_time_ms", 5000);
    cfg_.gripper_block = getInt(node_, "gripper_block", 1);
    cfg_.gripper_type = getInt(node_, "gripper_type", 0);
  }

  bool motionGateOpen() const
  {
    std::string reason;
    return motionAllowed(cfg_, reason);
  }

  void onJointState(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(joint_mu_);
    latest_joints_.joints.clear();
    latest_joints_.velocities.clear();
    const size_t n = std::min(msg->name.size(), msg->position.size());
    for (size_t i = 0; i < n; ++i)
    {
      latest_joints_.joints[msg->name[i]] = msg->position[i];
      if (i < msg->velocity.size())
      {
        latest_joints_.velocities[msg->name[i]] = msg->velocity[i];
      }
    }
    latest_joints_.stamp_valid = true;
    last_joint_stamp_ = rclcpp::Time(msg->header.stamp);
    latest_joints_.age_sec =
        std::max(0.0, (node_->get_clock()->now() - last_joint_stamp_).seconds());
    have_joints_ = true;
  }

  std::optional<JointSnapshot> currentSnapshot()
  {
    for (int i = 0; i < 20 && rclcpp::ok(); ++i)
    {
      rclcpp::spin_some(node_);
      std::lock_guard<std::mutex> lock(joint_mu_);
      if (have_joints_)
      {
        auto snap = latest_joints_;
        snap.age_sec = std::max(0.0, (node_->get_clock()->now() - last_joint_stamp_).seconds());
        return snap;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::lock_guard<std::mutex> lock(joint_mu_);
    if (!have_joints_)
    {
      return std::nullopt;
    }
    auto snap = latest_joints_;
    snap.age_sec = std::max(0.0, (node_->get_clock()->now() - last_joint_stamp_).seconds());
    return snap;
  }

  void inspectLive(bool motion)
  {
    auto snap = currentSnapshot();
    if (snap)
    {
      RCLCPP_INFO(node_->get_logger(), "live /joint_states: available joints=%zu age=%.3f",
                  snap->joints.size(), snap->age_sec);
    }
    else
    {
      RCLCPP_WARN(node_->get_logger(),
                  "live /joint_states: not available (ok for offline dry-run)");
    }
    const bool sim = node_->get_parameter("use_sim_time").as_bool();
    RCLCPP_INFO(node_->get_logger(), "executor use_sim_time=%s", sim ? "true" : "false");
    if (sim)
    {
      RCLCPP_ERROR(node_->get_logger(), "%s", "REAL_EXECUTION_SIM_TIME_INVALID");
    }
    auto names = node_->get_node_graph_interface()->get_node_names();
    for (const auto& name : names)
    {
      if (name.find("gazebo") != std::string::npos || name.find("gz_ros") != std::string::npos)
      {
        gazebo_detected_ = true;
      }
    }
    try
    {
      auto client = std::make_shared<rclcpp::SyncParametersClient>(node_, "move_group");
      if (client->wait_for_service(std::chrono::milliseconds(200)))
      {
        auto values = client->get_parameters({ "use_sim_time", "robot_description" });
        for (const auto& p : values)
        {
          if (p.get_name() == "use_sim_time" && p.as_bool())
          {
            RCLCPP_ERROR(node_->get_logger(), "move_group use_sim_time=true");
          }
          if (p.get_name() == "robot_description")
          {
            const auto urdf = p.as_string();
            if (urdf.find("GazeboSimSystem") != std::string::npos ||
                urdf.find("gz_ros2_control") != std::string::npos)
            {
              gazebo_detected_ = true;
            }
            if (urdf.find("FR3RealSystem") != std::string::npos)
            {
              RCLCPP_INFO(node_->get_logger(), "hardware identifier FR3RealSystem present");
            }
          }
        }
      }
    }
    catch (const std::exception& ex)
    {
      RCLCPP_INFO(node_->get_logger(), "move_group parameter inspect skipped: %s", ex.what());
    }
    const bool action_up =
        traj_client_->wait_for_action_server(std::chrono::milliseconds(200));
    RCLCPP_INFO(node_->get_logger(), "live trajectory action %s: %s",
                cfg_.trajectory_action_name.c_str(), action_up ? "available" : "not online");
    const bool gripper_up = gripper_client_->wait_for_service(std::chrono::milliseconds(200));
    RCLCPP_INFO(node_->get_logger(), "live gripper service %s: %s",
                cfg_.gripper_service_name.c_str(), gripper_up ? "available" : "not online");
    if (gazebo_detected_)
    {
      RCLCPP_WARN(node_->get_logger(),
                  "Gazebo/sim markers detected. Real executor must not run execute:=true here.");
      if (motion)
      {
        RCLCPP_ERROR(node_->get_logger(), "refusing execute because Gazebo/sim was detected");
      }
    }
  }

  bool controllerReadyLive()
  {
    return traj_client_->wait_for_action_server(std::chrono::seconds(2));
  }

  void publishFrozenPreview(const std::vector<TrajectorySegmentRecord>& scaled)
  {
    moveit_msgs::msg::DisplayTrajectory disp;
    disp.model_id = "fairino3_v6_robot";
    if (!scaled.empty())
    {
      disp.trajectory_start.joint_state.name = scaled.front().joint_names;
      disp.trajectory_start.joint_state.position = scaled.front().start_joints;
    }
    for (const auto& seg : scaled)
    {
      moveit_msgs::msg::RobotTrajectory rt;
      rt.joint_trajectory = toJointTrajectory(seg, "base_link");
      disp.trajectory.push_back(rt);
    }
    display_pub_->publish(disp);
    RCLCPP_INFO(node_->get_logger(),
                "published frozen Home→C preview to /display_planned_path (geometry unchanged)");
  }

  PlanResult planCurrentToHome()
  {
    PlanResult result;
    result.attempted = true;
    result.used_frozen_current_to_home = false;
    if (!cfg_.plan_current_to_home && !motionGateOpen())
    {
      result.success = true;
      RCLCPP_INFO(node_->get_logger(),
                  "dry-run: Current→Home remains RUNTIME REPLAN; not calling move_group");
      return result;
    }
    if (!move_client_->wait_for_action_server(std::chrono::seconds(5)))
    {
      result.error = "move_action unavailable";
      return result;
    }
    MoveGroup::Goal goal;
    goal.planning_options.plan_only = true;
    goal.planning_options.replan = false;
    goal.planning_options.planning_scene_diff.is_diff = true;
    goal.planning_options.planning_scene_diff.robot_state.is_diff = true;
    auto& req = goal.request;
    req.group_name = cfg_.planning_group;
    req.num_planning_attempts = 10;
    req.allowed_planning_time = 10.0;
    req.max_velocity_scaling_factor = cfg_.trajectory_speed_scale;
    req.max_acceleration_scaling_factor = cfg_.trajectory_speed_scale;
    req.start_state.is_diff = true;
    req.workspace_parameters.header.frame_id = "base_link";
    req.workspace_parameters.min_corner.x = -2.0;
    req.workspace_parameters.min_corner.y = -2.0;
    req.workspace_parameters.min_corner.z = -2.0;
    req.workspace_parameters.max_corner.x = 2.0;
    req.workspace_parameters.max_corner.y = 2.0;
    req.workspace_parameters.max_corner.z = 2.0;
    moveit_msgs::msg::Constraints constraints;
    const auto home = traj_.winner.home_rad.empty() ? fr_task_planner::kStep12cHomeRad :
                                                      fr_task_planner::jointsToVec(traj_.winner.home_rad);
    for (size_t i = 0; i < kArmJoints.size(); ++i)
    {
      moveit_msgs::msg::JointConstraint jc;
      jc.joint_name = kArmJoints[i];
      jc.position = home[i];
      jc.tolerance_above = 0.001;
      jc.tolerance_below = 0.001;
      jc.weight = 1.0;
      constraints.joint_constraints.push_back(jc);
    }
    req.goal_constraints.push_back(constraints);
    RCLCPP_INFO(node_->get_logger(),
                "planning collision-aware Current→Home via MoveGroup plan_only=true scale=%.3f",
                cfg_.trajectory_speed_scale);
    auto send = move_client_->async_send_goal(goal);
    if (rclcpp::spin_until_future_complete(node_, send, std::chrono::seconds(15)) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
      result.error = "Current→Home plan send timeout";
      return result;
    }
    auto handle = send.get();
    if (!handle)
    {
      result.error = "Current→Home plan rejected";
      return result;
    }
    auto wrapped = move_client_->async_get_result(handle);
    if (rclcpp::spin_until_future_complete(node_, wrapped, std::chrono::seconds(30)) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
      result.error = "Current→Home plan result timeout";
      return result;
    }
    auto wrapped_result = wrapped.get();
    if (!wrapped_result.result || wrapped_result.result->error_code.val != 1)
    {
      result.error = "Current→Home MoveIt plan failed";
      return result;
    }
    result.planned = fromRobotTrajectory(wrapped_result.result->planned_trajectory);
    if (result.planned.points.empty())
    {
      result.error = "Current→Home planned trajectory empty";
      return result;
    }
    moveit_msgs::msg::DisplayTrajectory disp;
    disp.model_id = "fairino3_v6_robot";
    disp.trajectory_start = wrapped_result.result->trajectory_start;
    disp.trajectory.push_back(wrapped_result.result->planned_trajectory);
    display_pub_->publish(disp);
    result.success = true;
    RCLCPP_INFO(node_->get_logger(),
                "Current→Home runtime plan ready points=%zu duration=%.3f s (not executed unless gated)",
                result.planned.points.size(), result.planned.duration);
    return result;
  }

  SendResult sendSegment(const std::string& logical, const TrajectorySegmentRecord& seg)
  {
    SendResult result;
    result.attempted = true;
    if (logical == kLogicalCurrentToHome)
    {
      result.error = "refusing frozen Current_to_Home";
      return result;
    }
    if (!motionGateOpen())
    {
      result.error = "motion gate closed; refusing FollowJointTrajectory";
      RCLCPP_ERROR(node_->get_logger(), "%s", result.error.c_str());
      return result;
    }
    TrajectorySegmentRecord remapped;
    std::string error;
    if (!remapSegmentByJointName(seg, cfg_.controller_joint_names, remapped, error))
    {
      result.error = error;
      return result;
    }
    FollowJointTrajectory::Goal goal;
    goal.trajectory = toJointTrajectory(remapped, "base_link");
    goal.trajectory.header.stamp = node_->get_clock()->now();
    const double timeout =
        actionTimeoutSec(segmentDuration(remapped), cfg_.timeout_factor, cfg_.timeout_margin_sec);
    RCLCPP_INFO(node_->get_logger(),
                "sending FollowJointTrajectory %s points=%zu timeout=%.1f s action=%s",
                logical.c_str(), remapped.points.size(), timeout,
                cfg_.trajectory_action_name.c_str());
    if (!traj_client_->wait_for_action_server(std::chrono::seconds(2)))
    {
      result.error = "trajectory action unavailable";
      return result;
    }
    auto send = traj_client_->async_send_goal(goal);
    if (rclcpp::spin_until_future_complete(
            node_, send, std::chrono::duration<double>(std::min(timeout, 30.0))) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
      result.error = "trajectory goal send timeout";
      return result;
    }
    auto handle = send.get();
    if (!handle)
    {
      result.error = "trajectory goal rejected";
      return result;
    }
    result.sent = true;
    ++motion_commands_sent_;
    auto wrapped = traj_client_->async_get_result(handle);
    if (rclcpp::spin_until_future_complete(node_, wrapped, std::chrono::duration<double>(timeout)) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
      result.error = "trajectory result timeout";
      return result;
    }
    auto wrapped_result = wrapped.get();
    if (wrapped_result.code != rclcpp_action::ResultCode::SUCCEEDED)
    {
      result.error = "trajectory controller failure";
      return result;
    }
    result.success = true;
    return result;
  }

  SendResult closeGripper()
  {
    SendResult result;
    result.attempted = true;
    if (!motionGateOpen())
    {
      result.error = "motion gate closed; refusing gripper command";
      return result;
    }
    auto req = std::make_shared<fairino_msgs::srv::GripperBridge::Request>();
    req->command = "move";
    req->gripper_id = cfg_.gripper_id;
    req->position = cfg_.gripper_close_position;
    req->velocity = cfg_.gripper_velocity;
    req->force = cfg_.gripper_force;
    req->max_time_ms = cfg_.gripper_max_time_ms;
    req->block = cfg_.gripper_block;
    req->gripper_type = cfg_.gripper_type;
    if (!gripper_client_->wait_for_service(std::chrono::seconds(2)))
    {
      result.error = fr_task_planner::kErrorGripperCloseFailed;
      return result;
    }
    auto future = gripper_client_->async_send_request(req);
    result.sent = true;
    ++gripper_commands_sent_;
    if (rclcpp::spin_until_future_complete(
            node_, future, std::chrono::duration<double>(cfg_.gripper_timeout_sec)) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
      result.error = fr_task_planner::kErrorGripperCloseFailed;
      return result;
    }
    auto resp = future.get();
    if (!resp || resp->error_code != 0)
    {
      result.error = fr_task_planner::kErrorGripperCloseFailed;
      return result;
    }
    if (cfg_.gripper_post_close_wait_sec > 0.0)
    {
      std::this_thread::sleep_for(
          std::chrono::duration<double>(cfg_.gripper_post_close_wait_sec));
    }
    result.success = true;
    return result;
  }

  SendResult attachObject()
  {
    SendResult result;
    result.attempted = true;
    if (!motionGateOpen())
    {
      result.error = "motion gate closed; refusing PlanningScene attach";
      return result;
    }
    moveit_msgs::msg::AttachedCollisionObject attached;
    attached.link_name = cfg_.attach_link;
    attached.object.id = cfg_.object_id;
    attached.object.operation = moveit_msgs::msg::CollisionObject::ADD;
    attached.touch_links = cfg_.touch_links;
    attach_pub_->publish(attached);
    result.sent = true;
    result.success = true;
    RCLCPP_INFO(node_->get_logger(), "PlanningScene attach %s -> %s touch_links=%zu",
                cfg_.object_id.c_str(), cfg_.attach_link.c_str(), cfg_.touch_links.size());
    return result;
  }

  SendResult restoreTableCollision()
  {
    SendResult result;
    result.attempted = true;
    if (!motionGateOpen())
    {
      result.error = "motion gate closed; refusing ACM restore";
      return result;
    }
    if (!get_scene_->wait_for_service(std::chrono::seconds(2)) ||
        !apply_scene_->wait_for_service(std::chrono::seconds(2)))
    {
      result.error = "PlanningScene services unavailable";
      return result;
    }
    auto get_req = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
    get_req->components.components =
        moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX;
    auto get_future = get_scene_->async_send_request(get_req);
    if (rclcpp::spin_until_future_complete(node_, get_future, std::chrono::seconds(5)) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
      result.error = "GetPlanningScene timeout";
      return result;
    }
    auto scene_resp = get_future.get();
    auto acm = scene_resp->scene.allowed_collision_matrix;
    auto findIndex = [&](const std::string& name) -> int {
      for (size_t i = 0; i < acm.entry_names.size(); ++i)
      {
        if (acm.entry_names[i] == name)
        {
          return static_cast<int>(i);
        }
      }
      return -1;
    };
    const int a = findIndex(cfg_.object_id);
    const int b = findIndex(cfg_.table_name);
    if (a >= 0 && b >= 0 && static_cast<size_t>(a) < acm.entry_values.size() &&
        static_cast<size_t>(b) < acm.entry_values[a].enabled.size())
    {
      acm.entry_values[a].enabled[b] = false;
      acm.entry_values[b].enabled[a] = false;
    }
    auto apply_req = std::make_shared<moveit_msgs::srv::ApplyPlanningScene::Request>();
    apply_req->scene.is_diff = true;
    apply_req->scene.allowed_collision_matrix = acm;
    auto apply_future = apply_scene_->async_send_request(apply_req);
    if (rclcpp::spin_until_future_complete(node_, apply_future, std::chrono::seconds(5)) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
      result.error = "ApplyPlanningScene timeout";
      return result;
    }
    result.sent = true;
    result.success = true;
    RCLCPP_INFO(node_->get_logger(), "restored part-table collision %s <-> %s",
                cfg_.object_id.c_str(), cfg_.table_name.c_str());
    return result;
  }

  rclcpp::Node::SharedPtr node_;
  ExecutorConfig cfg_;
  fr_task_planner::PersistedTrajectory traj_;
  std::mutex joint_mu_;
  JointSnapshot latest_joints_;
  rclcpp::Time last_joint_stamp_{ 0, 0, RCL_ROS_TIME };
  bool have_joints_ = false;
  bool gazebo_detected_ = false;
  int motion_commands_sent_ = 0;
  int gripper_commands_sent_ = 0;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr display_pub_;
  rclcpp::Publisher<moveit_msgs::msg::AttachedCollisionObject>::SharedPtr attach_pub_;
  rclcpp::Client<moveit_msgs::srv::ApplyPlanningScene>::SharedPtr apply_scene_;
  rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr get_scene_;
  rclcpp::Client<fairino_msgs::srv::GripperBridge>::SharedPtr gripper_client_;
  rclcpp_action::Client<FollowJointTrajectory>::SharedPtr traj_client_;
  rclcpp_action::Client<MoveGroup>::SharedPtr move_client_;
};
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("fr3_real_frozen_trajectory_executor");
  declareDefault(node, "execute", false);
  declareDefault(node, "real_robot_confirmation", std::string(""));
  declareDefault(node, "plan_current_to_home", false);
  declareDefault(node, "gripper_enabled", true);
  declareDefault(node, "trajectory_speed_scale", 0.05);
  declareDefault(node, "home_tolerance_rad", 0.02);
  declareDefault(node, "start_state_tolerance_rad", 0.02);
  declareDefault(node, "segment_start_tolerance_rad", 0.02);
  declareDefault(node, "segment_end_tolerance_rad", 0.02);
  declareDefault(node, "joint_settle_timeout_sec", 10.0);
  declareDefault(node, "joint_settle_poll_period_sec", 0.10);
  declareDefault(node, "joint_settle_required_samples", 3);
  declareDefault(node, "joint_state_max_age_sec", 0.5);
  declareDefault(node, "timeout_factor", 1.5);
  declareDefault(node, "timeout_margin_sec", 10.0);
  declareDefault(node, "gripper_timeout_sec", 15.0);
  declareDefault(node, "gripper_post_close_wait_sec", 0.5);
  declareDefault(node, "trajectory_file", std::string(""));
  declareDefault(node, "controller_name", std::string("fairino3_controller"));
  declareDefault(node, "trajectory_action_name", std::string(kDefaultTrajectoryActionName));
  declareDefault(node, "controller_config_source", std::string(""));
  declareDefault(node, "gripper_service_name", std::string(kDefaultGripperService));
  declareDefault(node, "gripper_config_source", std::string(""));
  declareDefault(node, "planning_group", std::string("fairino3_v6_group"));
  declareDefault(node, "attach_link", std::string("gripper_tcp"));
  declareDefault(node, "object_name", std::string("small_part"));
  declareDefault(node, "table_name", std::string("table"));
  declareDefault(node, "controller_joint_names",
                 std::vector<std::string>(kArmJoints.begin(), kArmJoints.end()));
  declareDefault(node, "touch_links", defaultTouchLinks());
  declareDefault(node, "gripper_id", 1);
  declareDefault(node, "gripper_close_position", 85);
  declareDefault(node, "gripper_velocity", 20);
  declareDefault(node, "gripper_force", 20);
  declareDefault(node, "gripper_max_time_ms", 5000);
  declareDefault(node, "gripper_block", 1);
  declareDefault(node, "gripper_type", 0);
  declareDefault(node, "write_scaled_debug_yaml", true);
  declareDefault(node, "scaled_debug_yaml", std::string("/tmp/fr3_step13_scaled_trajectory.yaml"));
  RealFrozenExecutorNode exec(node);
  const int rc = exec.run();
  rclcpp::shutdown();
  return rc;
}
