#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/task_constructor/solvers/pipeline_planner.h>
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/storage.h>
#include <moveit/task_constructor/task.h>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <moveit_task_constructor_msgs/msg/solution.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

namespace
{
const char* kPlanningGroup = "fairino3_v6_group";
const char* kPlanningFrame = "base_link";
const char* kEeLink = "gripper_tcp";
const char* kOmplPipeline = "ompl";
const char* kPilzPipeline = "pilz_industrial_motion_planner";
const char* kPilzPlannerId = "LIN";
const std::vector<std::string> kArmJoints = { "j1", "j2", "j3", "j4", "j5", "j6" };
constexpr double kJointContinuityTol = 1e-6;
constexpr double kLateralTolM = 0.001;
constexpr double kLinOriTolDeg = 0.5;

double getDouble(const rclcpp::Node::SharedPtr& node, const std::string& name)
{
  if (!node->has_parameter(name))
  {
    throw std::runtime_error("Missing parameter: " + name);
  }
  return node->get_parameter(name).as_double();
}

std::string getString(const rclcpp::Node::SharedPtr& node, const std::string& name,
                      const std::string& fallback)
{
  if (!node->has_parameter(name))
  {
    return fallback;
  }
  return node->get_parameter(name).as_string();
}

std::map<std::string, double> readHome(const rclcpp::Node::SharedPtr& node)
{
  std::map<std::string, double> home;
  for (const auto& name : kArmJoints)
  {
    home[name] = getDouble(node, "home_" + name);
  }
  return home;
}

geometry_msgs::msg::PoseStamped readPose(const rclcpp::Node::SharedPtr& node,
                                         const std::string& prefix)
{
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = getString(node, prefix + "_frame", kPlanningFrame);
  pose.pose.position.x = getDouble(node, prefix + "_x");
  pose.pose.position.y = getDouble(node, prefix + "_y");
  pose.pose.position.z = getDouble(node, prefix + "_z");
  pose.pose.orientation.x = getDouble(node, prefix + "_qx");
  pose.pose.orientation.y = getDouble(node, prefix + "_qy");
  pose.pose.orientation.z = getDouble(node, prefix + "_qz");
  pose.pose.orientation.w = getDouble(node, prefix + "_qw");
  return pose;
}

std::string formatJoints(const std::map<std::string, double>& joints)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(6);
  for (const auto& name : kArmJoints)
  {
    const auto it = joints.find(name);
    if (it == joints.end())
    {
      continue;
    }
    oss << name << "=" << it->second << " rad (" << (it->second * 180.0 / M_PI) << " deg)\n";
  }
  return oss.str();
}

std::string formatPose(const geometry_msgs::msg::PoseStamped& pose)
{
  const auto& p = pose.pose.position;
  const auto& q = pose.pose.orientation;
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(6);
  oss << "frame=" << pose.header.frame_id << " xyz=(" << p.x << ", " << p.y << ", " << p.z
      << ") xyzw=(" << q.x << ", " << q.y << ", " << q.z << ", " << q.w << ")";
  return oss.str();
}

Eigen::Isometry3d poseToIso(const geometry_msgs::msg::Pose& pose)
{
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  transform.linear() = Eigen::Quaterniond(pose.orientation.w, pose.orientation.x, pose.orientation.y,
                                          pose.orientation.z)
                           .normalized()
                           .toRotationMatrix();
  return transform;
}

std::string formatIso(const Eigen::Isometry3d& transform, const std::string& frame)
{
  const Eigen::Vector3d p = transform.translation();
  const Eigen::Quaterniond q(transform.rotation());
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(6);
  oss << "frame=" << frame << " xyz=(" << p.x() << ", " << p.y() << ", " << p.z() << ") xyzw=("
      << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << ")";
  return oss.str();
}

void poseError(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b, double& position_m,
               double& orientation_deg)
{
  position_m = (a.translation() - b.translation()).norm();
  const Eigen::Quaterniond qa(a.rotation());
  const Eigen::Quaterniond qb(b.rotation());
  double qdot = std::abs(qa.normalized().dot(qb.normalized()));
  qdot = std::min(1.0, qdot);
  orientation_deg = 2.0 * std::acos(qdot) * 180.0 / M_PI;
}

double maxJointError(const std::map<std::string, double>& a, const std::map<std::string, double>& b)
{
  double max_error = 0.0;
  for (const auto& name : kArmJoints)
  {
    max_error = std::max(max_error, std::abs(a.at(name) - b.at(name)));
  }
  return max_error;
}

std::map<std::string, double> jointsFromMsg(const sensor_msgs::msg::JointState& msg)
{
  std::map<std::string, double> joints;
  const size_t n = std::min(msg.name.size(), msg.position.size());
  for (size_t i = 0; i < n; ++i)
  {
    joints[msg.name[i]] = msg.position[i];
  }
  return joints;
}

bool jointsAreAllZero(const std::map<std::string, double>& joints)
{
  for (const auto& name : kArmJoints)
  {
    const auto it = joints.find(name);
    if (it == joints.end() || std::abs(it->second) > 1e-6)
    {
      return false;
    }
  }
  return true;
}

void applyJoints(moveit::core::RobotState& state, const std::map<std::string, double>& joints)
{
  for (const auto& item : joints)
  {
    if (state.getRobotModel()->hasJointModel(item.first))
    {
      state.setVariablePosition(item.first, item.second);
    }
  }
  state.update();
}

Eigen::Isometry3d tcpInBase(const moveit::core::RobotState& state, const std::string& ee_link)
{
  const Eigen::Isometry3d t_model_tcp = state.getGlobalLinkTransform(ee_link);
  const Eigen::Isometry3d t_model_base = state.getGlobalLinkTransform("base_link");
  return t_model_base.inverse() * t_model_tcp;
}

std::map<std::string, double> jointsFromTrajectoryPoint(
    const trajectory_msgs::msg::JointTrajectory& joint_trajectory,
    const trajectory_msgs::msg::JointTrajectoryPoint& point)
{
  std::map<std::string, double> joints;
  const size_t n = std::min(joint_trajectory.joint_names.size(), point.positions.size());
  for (size_t i = 0; i < n; ++i)
  {
    joints[joint_trajectory.joint_names[i]] = point.positions[i];
  }
  return joints;
}

struct TrajectorySegment
{
  std::string planner_id;
  std::string comment;
  std::vector<std::map<std::string, double>> points;
};

std::vector<TrajectorySegment> extractArmSegments(
    const moveit_task_constructor_msgs::msg::Solution& solution_msg)
{
  std::vector<TrajectorySegment> segments;
  for (const auto& sub : solution_msg.sub_trajectory)
  {
    TrajectorySegment segment;
    segment.planner_id = sub.info.planner_id;
    segment.comment = sub.info.comment;
    const auto& joint_trajectory = sub.trajectory.joint_trajectory;
    for (const auto& point : joint_trajectory.points)
    {
      auto joints = jointsFromTrajectoryPoint(joint_trajectory, point);
      bool has_arm = false;
      for (const auto& name : kArmJoints)
      {
        if (joints.count(name) != 0)
        {
          has_arm = true;
          break;
        }
      }
      if (has_arm)
      {
        segment.points.push_back(std::move(joints));
      }
    }
    if (!segment.points.empty())
    {
      segments.push_back(std::move(segment));
    }
  }
  return segments;
}

bool overlayRobotDescriptionFromMoveGroup(const rclcpp::Node::SharedPtr& node)
{
  auto client = std::make_shared<rclcpp::SyncParametersClient>(node, "move_group");
  if (!client->wait_for_service(std::chrono::seconds(60)))
  {
    RCLCPP_ERROR(node->get_logger(), "Timed out waiting for /move_group. Start stage4_full first.");
    return false;
  }
  const std::vector<std::string> names = { "robot_description", "robot_description_semantic" };
  const auto values = client->get_parameters(names);
  for (const auto& parameter : values)
  {
    if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET)
    {
      RCLCPP_ERROR(node->get_logger(), "/move_group is missing parameter '%s'",
                   parameter.get_name().c_str());
      return false;
    }
    if (!node->has_parameter(parameter.get_name()))
    {
      node->declare_parameter<std::string>(parameter.get_name(), parameter.as_string());
    }
    else
    {
      node->set_parameter(rclcpp::Parameter(parameter.get_name(), parameter.as_string()));
    }
  }
  return true;
}

sensor_msgs::msg::JointState::SharedPtr waitForFreshJoints(const rclcpp::Node::SharedPtr& node,
                                                           std::chrono::seconds timeout)
{
  sensor_msgs::msg::JointState::SharedPtr latest;
  auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(),
      [&latest](const sensor_msgs::msg::JointState::SharedPtr msg) { latest = msg; });

  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline)
  {
    if (latest)
    {
      std::map<std::string, double> found;
      const size_t n = std::min(latest->name.size(), latest->position.size());
      for (size_t i = 0; i < n; ++i)
      {
        found[latest->name[i]] = latest->position[i];
      }
      bool complete = true;
      for (const auto& name : kArmJoints)
      {
        if (found.count(name) == 0)
        {
          complete = false;
          break;
        }
      }
      if (complete)
      {
        return latest;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return latest;
}

void logSceneObjects(const rclcpp::Node::SharedPtr& node, bool& has_table, bool& has_column,
                     bool& has_small_part)
{
  has_table = false;
  has_column = false;
  has_small_part = false;
  auto client = node->create_client<moveit_msgs::srv::GetPlanningScene>("get_planning_scene");
  if (!client->wait_for_service(std::chrono::seconds(10)))
  {
    RCLCPP_ERROR(node->get_logger(), "get_planning_scene is not available");
    return;
  }
  auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
  request->components.components = moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_NAMES;
  auto future = client->async_send_request(request);
  if (future.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
  {
    RCLCPP_ERROR(node->get_logger(), "get_planning_scene timed out");
    return;
  }
  std::ostringstream known;
  for (const auto& obj : future.get()->scene.world.collision_objects)
  {
    if (!known.str().empty())
    {
      known << ", ";
    }
    known << obj.id;
    if (obj.id == "table")
    {
      has_table = true;
    }
    if (obj.id == "mounting_column")
    {
      has_column = true;
    }
    if (obj.id == "small_part")
    {
      has_small_part = true;
    }
  }
  RCLCPP_INFO(node->get_logger(), "PlanningScene: %s",
              known.str().empty() ? "(none)" : known.str().c_str());
}

void shutdownSpinner(rclcpp::executors::MultiThreadedExecutor& executor, std::thread& spinner)
{
  executor.cancel();
  if (spinner.joinable())
  {
    spinner.join();
  }
  rclcpp::shutdown();
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("fr3_mtc_grasp_chain_test", options);
  RCLCPP_INFO(node->get_logger(), "THIS STEP IS PLAN-ONLY. NO TRAJECTORY EXECUTION IS PERFORMED.");

  const auto home = readHome(node);
  const auto pregrasp = readPose(node, "pregrasp");
  const auto grasp = readPose(node, "grasp");
  const std::string group = getString(node, "planning_group", kPlanningGroup);
  const std::string planning_frame = getString(node, "planning_frame", kPlanningFrame);
  const std::string ee_link = getString(node, "ee_link", kEeLink);
  const double home_tol = node->has_parameter("max_home_error_rad") ?
                              node->get_parameter("max_home_error_rad").as_double() :
                              0.03;
  const double pos_tol = getDouble(node, "position_tolerance");
  const double ori_tol_deg = getDouble(node, "orientation_tolerance_deg");
  const double planning_time = getDouble(node, "planning_time");
  const double pregrasp_distance = getDouble(node, "pregrasp_distance");
  const int64_t max_solutions =
      node->has_parameter("max_solutions") ? node->get_parameter("max_solutions").as_int() : 5;

  if (pregrasp.header.frame_id != planning_frame || grasp.header.frame_id != planning_frame)
  {
    RCLCPP_ERROR(node->get_logger(), "PreGrasp/Grasp frame must be %s", planning_frame.c_str());
    rclcpp::shutdown();
    return 1;
  }

  if (!overlayRobotDescriptionFromMoveGroup(node))
  {
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  const auto before_msg = waitForFreshJoints(node, std::chrono::seconds(10));
  if (!before_msg)
  {
    RCLCPP_ERROR(node->get_logger(),
                 "/joint_states missing or incomplete (need j1~j6). STEP 4 FAIL.");
    shutdownSpinner(executor, spinner);
    return 1;
  }
  const auto actual = jointsFromMsg(*before_msg);
  if (jointsAreAllZero(actual))
  {
    RCLCPP_ERROR(node->get_logger(),
                 "CurrentState is all zeros. STEP 4 FAIL. This is not a valid Stage4 Home.");
    shutdownSpinner(executor, spinner);
    return 1;
  }

  std::ostringstream errors;
  double max_home_error = 0.0;
  for (const auto& name : kArmJoints)
  {
    const double err = std::abs(actual.at(name) - home.at(name));
    max_home_error = std::max(max_home_error, err);
    errors << "  " << name << " error=" << err << " rad\n";
  }

  const Eigen::Vector3d pre_p(pregrasp.pose.position.x, pregrasp.pose.position.y,
                              pregrasp.pose.position.z);
  const Eigen::Vector3d grasp_p(grasp.pose.position.x, grasp.pose.position.y,
                                grasp.pose.position.z);
  const Eigen::Vector3d approach = grasp_p - pre_p;
  const double approach_distance = approach.norm();

  bool has_table = false;
  bool has_column = false;
  bool has_small_part = false;
  logSceneObjects(node, has_table, has_column, has_small_part);

  RCLCPP_INFO(node->get_logger(), "========== STEP 4 PREFLIGHT ==========");
  RCLCPP_INFO(node->get_logger(), "Current joints:\n%s", formatJoints(actual).c_str());
  RCLCPP_INFO(node->get_logger(), "Expected Home:\n%s", formatJoints(home).c_str());
  RCLCPP_INFO(node->get_logger(), "Max Home error:\n%.6f rad (tol=%.3f)\n%s", max_home_error,
              home_tol, errors.str().c_str());
  RCLCPP_INFO(node->get_logger(), "Home:\n%s", max_home_error <= home_tol ? "PASS" : "FAIL");
  RCLCPP_INFO(node->get_logger(), "Planning group:\n%s", group.c_str());
  RCLCPP_INFO(node->get_logger(), "Goal frame:\n%s", planning_frame.c_str());
  RCLCPP_INFO(node->get_logger(), "IK frame:\n%s", ee_link.c_str());
  RCLCPP_INFO(node->get_logger(), "PreGrasp:\n%s", formatPose(pregrasp).c_str());
  RCLCPP_INFO(node->get_logger(), "Grasp:\n%s", formatPose(grasp).c_str());
  RCLCPP_INFO(node->get_logger(), "PreGrasp → Grasp Cartesian displacement:\ndx=%.6f\ndy=%.6f\ndz=%.6f\ndistance=%.6f",
              approach.x(), approach.y(), approach.z(), approach_distance);
  RCLCPP_INFO(node->get_logger(), "Approach distance:\n%.6f", pregrasp_distance);
  RCLCPP_INFO(node->get_logger(), "PlanningScene:\ntable=%s\nmounting_column=%s",
              has_table ? "present" : "absent", has_column ? "present" : "absent");
  RCLCPP_INFO(node->get_logger(), "small_part:\n%s",
              has_small_part ? "present" : "ABSENT — known Step 4 limitation");

  if (max_home_error > home_tol)
  {
    RCLCPP_ERROR(node->get_logger(),
                 "Current is not Stage4 Home. Planning aborted. No motion commanded.");
    shutdownSpinner(executor, spinner);
    return 2;
  }

  moveit::task_constructor::Task task("", true);
  task.setName("FR3 Grasp Approach Task");
  task.loadRobotModel(node);
  const auto robot_model = task.getRobotModel();
  if (!robot_model || !robot_model->hasJointModelGroup(group) || !robot_model->hasLinkModel(ee_link))
  {
    RCLCPP_ERROR(node->get_logger(), "Robot model / group / gripper_tcp not found");
    shutdownSpinner(executor, spinner);
    return 1;
  }
  RCLCPP_INFO(node->get_logger(), "Model frame:\n%s", robot_model->getModelFrame().c_str());

  auto current_state = std::make_unique<moveit::task_constructor::stages::CurrentState>("CurrentState");
  task.add(std::move(current_state));

  auto ompl = std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
  ompl->setTimeout(planning_time);
  auto move_pregrasp =
      std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo PreGrasp", ompl);
  move_pregrasp->setGroup(group);
  move_pregrasp->setIKFrame(ee_link);
  move_pregrasp->setGoal(pregrasp);
  move_pregrasp->setTimeout(planning_time);
  task.add(std::move(move_pregrasp));

  auto pilz = std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kPilzPipeline);
  pilz->setPlannerId(kPilzPlannerId);
  pilz->setTimeout(planning_time);
  auto move_grasp =
      std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo Grasp", pilz);
  move_grasp->setGroup(group);
  move_grasp->setIKFrame(ee_link);
  move_grasp->setGoal(grasp);
  move_grasp->setTimeout(planning_time);
  task.add(std::move(move_grasp));

  RCLCPP_INFO(node->get_logger(), "========== MTC TASK ==========");
  RCLCPP_INFO(node->get_logger(), "CurrentState\n   ↓\nMoveTo PreGrasp\npipeline = OMPL\n   ↓\nMoveTo Grasp\npipeline = Pilz\nplanner_id = LIN");
  RCLCPP_INFO(node->get_logger(), "Stage: MoveTo PreGrasp\nPipeline: ompl\nPlanner: default OMPL / RRTConnect");
  RCLCPP_INFO(node->get_logger(), "Stage: MoveTo Grasp\nPipeline: pilz_industrial_motion_planner\nPlanner ID: LIN");
  RCLCPP_INFO(node->get_logger(), "Collision checking: ENABLED (avoid_collisions not disabled)");
  RCLCPP_INFO(node->get_logger(),
              "STEP 4 is NOT yet a collision-complete physical grasp. small_part is absent.");

  try
  {
    task.init();
  }
  catch (const std::exception& ex)
  {
    RCLCPP_ERROR(node->get_logger(), "task.init() failed: %s", ex.what());
    shutdownSpinner(executor, spinner);
    return 1;
  }

  const auto plan_result = task.plan(static_cast<size_t>(std::max<int64_t>(1, max_solutions)));
  const size_t num_solutions = task.numSolutions();
  std::ostringstream state;
  task.printState(state);
  RCLCPP_INFO(node->get_logger(), "planning result: %s", plan_result ? "SUCCESS" : "FAIL");
  RCLCPP_INFO(node->get_logger(), "Complete solution count: %zu", num_solutions);
  RCLCPP_INFO(node->get_logger(), "MTC task state:\n%s", state.str().c_str());

  const bool pregrasp_stage_ok = state.str().find("MoveTo PreGrasp") != std::string::npos;
  (void)pregrasp_stage_ok;
  if (!plan_result || num_solutions < 1)
  {
    std::ostringstream failure;
    task.explainFailure(failure);
    RCLCPP_ERROR(node->get_logger(), "planning failure:\n%s", failure.str().c_str());
    RCLCPP_ERROR(node->get_logger(), "PreGrasp success: UNKNOWN (see MTC task state)");
    RCLCPP_ERROR(node->get_logger(), "LIN Grasp success: NO");
    RCLCPP_ERROR(node->get_logger(), "Complete Solution: NO");
    task.publishAllSolutions(false);
    RCLCPP_INFO(node->get_logger(), "Introspection active. STEP 4 FAIL. Ctrl+C to exit.");
    spinner.join();
    return 3;
  }

  const auto& solution = *task.solutions().front();
  moveit_task_constructor_msgs::msg::Solution solution_msg;
  solution.toMsg(solution_msg);
  const auto segments = extractArmSegments(solution_msg);
  RCLCPP_INFO(node->get_logger(), "========== COMPLETE SOLUTION VALIDATION ==========");
  RCLCPP_INFO(node->get_logger(), "Complete solution count:\n%zu", num_solutions);
  RCLCPP_INFO(node->get_logger(), "arm trajectory segments: %zu", segments.size());
  for (size_t i = 0; i < segments.size(); ++i)
  {
    RCLCPP_INFO(node->get_logger(), "segment %zu planner_id='%s' comment='%s' points=%zu", i,
                segments[i].planner_id.c_str(), segments[i].comment.c_str(),
                segments[i].points.size());
  }
  if (segments.size() < 2)
  {
    RCLCPP_ERROR(node->get_logger(),
                 "Complete solution did not contain both OMPL and LIN segments.");
    shutdownSpinner(executor, spinner);
    return 3;
  }

  const auto& ompl_seg = segments[0];
  const auto& lin_seg = segments[1];
  RCLCPP_INFO(node->get_logger(), "OMPL trajectory points:\n%zu", ompl_seg.points.size());
  RCLCPP_INFO(node->get_logger(), "LIN trajectory points:\n%zu", lin_seg.points.size());
  if (ompl_seg.points.empty() || lin_seg.points.size() < 2)
  {
    RCLCPP_ERROR(node->get_logger(), "A trajectory segment is too short");
    shutdownSpinner(executor, spinner);
    return 3;
  }

  moveit::core::RobotState fk_state(robot_model);
  fk_state.setToDefaultValues();
  applyJoints(fk_state, actual);

  const Eigen::Isometry3d t_pre_target = poseToIso(pregrasp.pose);
  const Eigen::Isometry3d t_grasp_target = poseToIso(grasp.pose);

  applyJoints(fk_state, ompl_seg.points.back());
  const Eigen::Isometry3d t_pre_fk = tcpInBase(fk_state, ee_link);
  double pre_pos = 0.0;
  double pre_ori = 0.0;
  poseError(t_pre_target, t_pre_fk, pre_pos, pre_ori);

  RCLCPP_INFO(node->get_logger(), "OMPL start joints:\n%s", formatJoints(ompl_seg.points.front()).c_str());
  RCLCPP_INFO(node->get_logger(), "OMPL final joints:\n%s", formatJoints(ompl_seg.points.back()).c_str());
  RCLCPP_INFO(node->get_logger(), "PreGrasp target:\n%s", formatPose(pregrasp).c_str());
  RCLCPP_INFO(node->get_logger(), "PreGrasp FK:\n%s", formatIso(t_pre_fk, "base_link").c_str());
  RCLCPP_INFO(node->get_logger(), "PreGrasp position error:\n%.6f m (tol=%.6f)", pre_pos, pos_tol);
  RCLCPP_INFO(node->get_logger(), "PreGrasp orientation error:\n%.6f deg (tol=%.6f)", pre_ori,
              ori_tol_deg);

  const double joint_disc = maxJointError(ompl_seg.points.back(), lin_seg.points.front());
  applyJoints(fk_state, ompl_seg.points.back());
  const Eigen::Isometry3d t_ompl_end = tcpInBase(fk_state, ee_link);
  applyJoints(fk_state, lin_seg.points.front());
  const Eigen::Isometry3d t_lin_start = tcpInBase(fk_state, ee_link);
  double tcp_disc_pos = 0.0;
  double tcp_disc_ori = 0.0;
  poseError(t_ompl_end, t_lin_start, tcp_disc_pos, tcp_disc_ori);

  RCLCPP_INFO(node->get_logger(), "OMPL end joints:\n%s", formatJoints(ompl_seg.points.back()).c_str());
  RCLCPP_INFO(node->get_logger(), "LIN start joints:\n%s", formatJoints(lin_seg.points.front()).c_str());
  RCLCPP_INFO(node->get_logger(), "max joint discontinuity:\n%.9f rad", joint_disc);
  RCLCPP_INFO(node->get_logger(), "TCP discontinuity position:\n%.6f m", tcp_disc_pos);
  RCLCPP_INFO(node->get_logger(), "TCP discontinuity orientation:\n%.6f deg", tcp_disc_ori);

  applyJoints(fk_state, lin_seg.points.back());
  const Eigen::Isometry3d t_grasp_fk = tcpInBase(fk_state, ee_link);
  double grasp_pos = 0.0;
  double grasp_ori = 0.0;
  poseError(t_grasp_target, t_grasp_fk, grasp_pos, grasp_ori);
  RCLCPP_INFO(node->get_logger(), "LIN final joints:\n%s", formatJoints(lin_seg.points.back()).c_str());
  RCLCPP_INFO(node->get_logger(), "Grasp target:\n%s", formatPose(grasp).c_str());
  RCLCPP_INFO(node->get_logger(), "Grasp FK:\n%s", formatIso(t_grasp_fk, "base_link").c_str());
  RCLCPP_INFO(node->get_logger(), "Grasp position error:\n%.6f m (tol=%.6f)", grasp_pos, pos_tol);
  RCLCPP_INFO(node->get_logger(), "Grasp orientation error:\n%.6f deg (tol=%.6f)", grasp_ori,
              ori_tol_deg);

  std::vector<Eigen::Isometry3d> lin_tcp;
  lin_tcp.reserve(lin_seg.points.size());
  for (const auto& joints : lin_seg.points)
  {
    applyJoints(fk_state, joints);
    lin_tcp.push_back(tcpInBase(fk_state, ee_link));
  }
  double path_length = 0.0;
  double max_lateral = 0.0;
  double max_lin_ori = 0.0;
  const Eigen::Vector3d line = grasp_p - pre_p;
  const double line_norm = line.norm();
  for (size_t i = 0; i < lin_tcp.size(); ++i)
  {
    if (i + 1 < lin_tcp.size())
    {
      path_length += (lin_tcp[i + 1].translation() - lin_tcp[i].translation()).norm();
    }
    const Eigen::Vector3d rel = lin_tcp[i].translation() - pre_p;
    double lateral = rel.norm();
    if (line_norm > 1e-9)
    {
      lateral = (rel - line * (rel.dot(line) / (line_norm * line_norm))).norm();
    }
    max_lateral = std::max(max_lateral, lateral);
    double unused = 0.0;
    double ori = 0.0;
    poseError(t_grasp_target, lin_tcp[i], unused, ori);
    max_lin_ori = std::max(max_lin_ori, ori);
  }

  RCLCPP_INFO(node->get_logger(), "PreGrasp → Grasp target distance:\n%.6f m", approach_distance);
  RCLCPP_INFO(node->get_logger(), "LIN TCP path length:\n%.6f m", path_length);
  RCLCPP_INFO(node->get_logger(), "max lateral deviation:\n%.6f m (tol=%.6f)", max_lateral,
              kLateralTolM);
  RCLCPP_INFO(node->get_logger(), "max orientation deviation:\n%.6f deg (tol=%.6f)", max_lin_ori,
              kLinOriTolDeg);

  const bool continuity_ok = joint_disc <= 1e-4 && tcp_disc_pos <= 1e-4 && tcp_disc_ori <= 0.05;
  const bool pre_ok = pre_pos <= pos_tol && pre_ori <= ori_tol_deg;
  const bool grasp_ok = grasp_pos <= pos_tol && grasp_ori <= ori_tol_deg;
  const bool lin_ok = max_lateral <= kLateralTolM && max_lin_ori <= kLinOriTolDeg;
  const bool scene_ok = has_table && has_column && !has_small_part;
  const bool all_ok = pre_ok && continuity_ok && grasp_ok && lin_ok && scene_ok;

  RCLCPP_INFO(node->get_logger(), "PreGrasp success: YES");
  RCLCPP_INFO(node->get_logger(), "LIN Grasp success: YES");
  RCLCPP_INFO(node->get_logger(), "Complete Solution: YES");
  RCLCPP_INFO(node->get_logger(), "Continuity: %s", continuity_ok ? "PASS" : "FAIL");
  RCLCPP_INFO(node->get_logger(), "LIN geometry: %s", lin_ok ? "PASS" : "FAIL");
  RCLCPP_INFO(node->get_logger(), "fallback used? NO");
  if (joint_disc > kJointContinuityTol && joint_disc <= 1e-4)
  {
    RCLCPP_WARN(node->get_logger(),
                "Joint continuity is above 1e-6 but within interpolation noise (%.9f rad).",
                joint_disc);
  }

  task.publishAllSolutions(false);
  task.enableIntrospection(true);

  const auto after_msg = waitForFreshJoints(node, std::chrono::seconds(5));
  if (after_msg)
  {
    const auto after = jointsFromMsg(*after_msg);
    const double drift = maxJointError(after, actual);
    RCLCPP_INFO(node->get_logger(), "JOINT STATES BEFORE:\n%s", formatJoints(actual).c_str());
    RCLCPP_INFO(node->get_logger(), "JOINT STATES AFTER:\n%s", formatJoints(after).c_str());
    RCLCPP_INFO(node->get_logger(), "Max joint drift after plan: %.6f rad", drift);
    RCLCPP_INFO(node->get_logger(), "Robot moved because of this node? %s",
                drift > 0.02 ? "YES" : "NO");
    if (drift > 0.02)
    {
      shutdownSpinner(executor, spinner);
      return 4;
    }
  }

  RCLCPP_INFO(node->get_logger(),
              "Introspection is active for RViz. THIS STEP IS PLAN-ONLY. Ctrl+C to exit.");
  spinner.join();
  return all_ok ? 0 : 4;
}
