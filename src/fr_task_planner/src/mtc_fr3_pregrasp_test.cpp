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
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/storage.h>
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
const char* kPipelineName = "ompl";
const std::vector<std::string> kArmJoints = { "j1", "j2", "j3", "j4", "j5", "j6" };

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

std::map<std::string, double> jointsFromState(const moveit::core::RobotState& state)
{
  std::map<std::string, double> joints;
  for (const auto& name : kArmJoints)
  {
    joints[name] = state.getVariablePosition(name);
  }
  return joints;
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

std::vector<std::map<std::string, double>> extractArmWaypoints(
    const moveit_task_constructor_msgs::msg::Solution& solution_msg)
{
  std::vector<std::map<std::string, double>> points;
  for (const auto& sub : solution_msg.sub_trajectory)
  {
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
        points.push_back(std::move(joints));
      }
    }
  }
  return points;
}

void applyArmJoints(moveit::core::RobotState& state, const std::map<std::string, double>& joints)
{
  for (const auto& name : kArmJoints)
  {
    const auto it = joints.find(name);
    if (it != joints.end())
    {
      state.setVariablePosition(name, it->second);
    }
  }
  state.update();
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

double maxHomeError(const std::map<std::string, double>& actual,
                    const std::map<std::string, double>& home)
{
  double max_error = 0.0;
  for (const auto& name : kArmJoints)
  {
    max_error = std::max(max_error, std::abs(actual.at(name) - home.at(name)));
  }
  return max_error;
}

void logSceneObjects(const rclcpp::Node::SharedPtr& node)
{
  auto client = node->create_client<moveit_msgs::srv::GetPlanningScene>("get_planning_scene");
  if (!client->wait_for_service(std::chrono::seconds(10)))
  {
    RCLCPP_ERROR(node->get_logger(), "get_planning_scene is not available");
    return;
  }
  auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
  request->components.components = moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_NAMES |
                                   moveit_msgs::msg::PlanningSceneComponents::ROBOT_STATE_ATTACHED_OBJECTS;
  auto future = client->async_send_request(request);
  if (future.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
  {
    RCLCPP_ERROR(node->get_logger(), "get_planning_scene timed out");
    return;
  }
  const auto response = future.get();
  std::ostringstream known;
  for (const auto& name : response->scene.world.collision_objects)
  {
    if (!known.str().empty())
    {
      known << ", ";
    }
    known << name.id;
  }
  std::ostringstream attached;
  for (const auto& obj : response->scene.robot_state.attached_collision_objects)
  {
    if (!attached.str().empty())
    {
      attached << ", ";
    }
    attached << obj.object.id;
  }
  RCLCPP_INFO(node->get_logger(), "Known collision objects: %s",
              known.str().empty() ? "(none)" : known.str().c_str());
  RCLCPP_INFO(node->get_logger(), "Attached collision objects: %s",
              attached.str().empty() ? "(none)" : attached.str().c_str());
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
  auto node = rclcpp::Node::make_shared("fr3_mtc_pregrasp_test", options);
  RCLCPP_INFO(node->get_logger(), "THIS STEP IS PLAN-ONLY. NO TRAJECTORY EXECUTION IS PERFORMED.");

  const auto home = readHome(node);
  const auto pregrasp = readPose(node, "pregrasp");
  const auto object_world = readPose(node, "object_world");
  const auto object_base = readPose(node, "object");
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
      node->has_parameter("max_solutions") ? node->get_parameter("max_solutions").as_int() : 3;

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
                 "/joint_states missing or incomplete (need j1~j6). STEP 3 FAIL.");
    shutdownSpinner(executor, spinner);
    return 1;
  }
  const auto actual = jointsFromMsg(*before_msg);
  if (jointsAreAllZero(actual))
  {
    RCLCPP_ERROR(node->get_logger(),
                 "CurrentState is all zeros. STEP 3 FAIL. This is not a valid Stage4 Home.");
    shutdownSpinner(executor, spinner);
    return 1;
  }

  std::ostringstream errors;
  double max_error = 0.0;
  for (const auto& name : kArmJoints)
  {
    const double err = std::abs(actual.at(name) - home.at(name));
    max_error = std::max(max_error, err);
    errors << "  " << name << " error=" << err << " rad\n";
  }

  RCLCPP_INFO(node->get_logger(), "========== STEP 3 PREFLIGHT ==========");
  RCLCPP_INFO(node->get_logger(), "Planning group:\n%s", group.c_str());
  RCLCPP_INFO(node->get_logger(), "Planning frame:\n%s", planning_frame.c_str());
  RCLCPP_INFO(node->get_logger(), "EE:\n%s", ee_link.c_str());
  RCLCPP_INFO(node->get_logger(), "Current joints:\n%s", formatJoints(actual).c_str());
  RCLCPP_INFO(node->get_logger(), "Expected Home:\n%s", formatJoints(home).c_str());
  RCLCPP_INFO(node->get_logger(), "Home error:\n%sMax joint error: %.6f rad (tol=%.3f)",
              errors.str().c_str(), max_error, home_tol);
  RCLCPP_INFO(node->get_logger(), "Object pose (world):\n%s", formatPose(object_world).c_str());
  RCLCPP_INFO(node->get_logger(), "Object pose (planning frame):\n%s", formatPose(object_base).c_str());
  RCLCPP_INFO(node->get_logger(), "Grasp pose:\n%s", formatPose(grasp).c_str());
  RCLCPP_INFO(node->get_logger(), "PreGrasp pose:\n%s", formatPose(pregrasp).c_str());
  RCLCPP_INFO(node->get_logger(), "PreGrasp distance:\n%.6f", pregrasp_distance);
  RCLCPP_INFO(node->get_logger(), "Planner:\nOMPL");

  if (max_error > home_tol)
  {
    RCLCPP_ERROR(node->get_logger(),
                 "Current is not Stage4 Home. Planning aborted. No motion commanded.");
    shutdownSpinner(executor, spinner);
    return 2;
  }

  logSceneObjects(node);

  moveit::task_constructor::Task task("", true);
  task.setName("FR3 PreGrasp Plan Test");
  task.loadRobotModel(node);
  const auto robot_model = task.getRobotModel();
  if (!robot_model || !robot_model->hasJointModelGroup(group))
  {
    RCLCPP_ERROR(node->get_logger(), "Robot model / group not found");
    shutdownSpinner(executor, spinner);
    return 1;
  }
  if (pregrasp.header.frame_id != planning_frame)
  {
    RCLCPP_ERROR(node->get_logger(), "PreGrasp frame must be %s, got %s", planning_frame.c_str(),
                 pregrasp.header.frame_id.c_str());
    shutdownSpinner(executor, spinner);
    return 1;
  }

  const auto* jmg = robot_model->getJointModelGroup(group);
  std::vector<std::string> tip_links;
  if (jmg)
  {
    jmg->getEndEffectorTips(tip_links);
  }
  std::ostringstream tip_log;
  if (tip_links.empty())
  {
    tip_log << "(none)";
  }
  else
  {
    for (size_t i = 0; i < tip_links.size(); ++i)
    {
      if (i != 0)
      {
        tip_log << ", ";
      }
      tip_log << tip_links[i];
    }
  }
  const auto* default_tip = jmg ? jmg->getOnlyOneEndEffectorTip() : nullptr;
  const std::string default_tip_name = default_tip ? default_tip->getName() : "(none)";

  RCLCPP_INFO(node->get_logger(), "========== STEP 3 FRAME DIAGNOSTICS ==========");
  RCLCPP_INFO(node->get_logger(), "RobotModel model frame:\n%s", robot_model->getModelFrame().c_str());
  RCLCPP_INFO(node->get_logger(), "PlanningScene planning frame / robot model frame:\n%s",
              robot_model->getModelFrame().c_str());
  RCLCPP_INFO(node->get_logger(), "MTC goal PoseStamped.header.frame_id:\n%s",
              pregrasp.header.frame_id.c_str());
  RCLCPP_INFO(node->get_logger(), "MTC goal xyz:\n%.6f %.6f %.6f", pregrasp.pose.position.x,
              pregrasp.pose.position.y, pregrasp.pose.position.z);
  RCLCPP_INFO(node->get_logger(), "MTC goal xyzw:\n%.6f %.6f %.6f %.6f", pregrasp.pose.orientation.x,
              pregrasp.pose.orientation.y, pregrasp.pose.orientation.z, pregrasp.pose.orientation.w);
  RCLCPP_INFO(node->get_logger(), "MoveTo group:\n%s", group.c_str());
  RCLCPP_INFO(node->get_logger(), "MoveTo IK frame:\n%s", ee_link.c_str());
  RCLCPP_INFO(node->get_logger(), "IK FRAME WAS EXPLICIT");
  RCLCPP_INFO(node->get_logger(), "JointModelGroup tip links:\n%s", tip_log.str().c_str());
  RCLCPP_INFO(node->get_logger(), "Group default tip:\n%s", default_tip_name.c_str());
  RCLCPP_INFO(node->get_logger(), "Requested EE:\n%s", ee_link.c_str());
  RCLCPP_INFO(node->get_logger(), "robot_model->hasLinkModel(\"%s\"): %s", ee_link.c_str(),
              robot_model->hasLinkModel(ee_link) ? "true" : "false");

  auto current_state = std::make_unique<moveit::task_constructor::stages::CurrentState>("CurrentState");
  task.add(std::move(current_state));

  auto pipeline = std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kPipelineName);
  pipeline->setTimeout(planning_time);

  auto move_to = std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo PreGrasp", pipeline);
  move_to->setGroup(group);
  move_to->setIKFrame(ee_link);
  move_to->setGoal(pregrasp);
  move_to->setTimeout(planning_time);
  auto* move_to_ptr = move_to.get();
  task.add(std::move(move_to));

  RCLCPP_INFO(node->get_logger(), "MTC Task: CurrentState -> MoveTo PreGrasp");
  RCLCPP_INFO(node->get_logger(), "Collision checking: ENABLED (avoid_collisions not disabled)");

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

  try
  {
    const auto& props = move_to_ptr->properties();
    const auto goal_prop = props.get<geometry_msgs::msg::PoseStamped>("goal");
    const auto ik_prop = props.get<geometry_msgs::msg::PoseStamped>("ik_frame");
    RCLCPP_INFO(node->get_logger(), "MoveTo stage properties:");
    RCLCPP_INFO(node->get_logger(), "  group: %s", props.get<std::string>("group").c_str());
    RCLCPP_INFO(node->get_logger(), "  ik_frame: %s", formatPose(ik_prop).c_str());
    RCLCPP_INFO(node->get_logger(), "  goal: %s", formatPose(goal_prop).c_str());
  }
  catch (const std::exception& ex)
  {
    RCLCPP_WARN(node->get_logger(), "Could not read MoveTo properties: %s", ex.what());
  }

  const auto plan_result = task.plan(static_cast<size_t>(std::max<int64_t>(1, max_solutions)));
  const size_t num_solutions = task.numSolutions();
  std::ostringstream state;
  task.printState(state);
  RCLCPP_INFO(node->get_logger(), "planning result: %s", plan_result ? "SUCCESS" : "FAIL");
  RCLCPP_INFO(node->get_logger(), "solution 数量: %zu", num_solutions);
  RCLCPP_INFO(node->get_logger(), "MTC task state:\n%s", state.str().c_str());
  if (!plan_result || num_solutions < 1)
  {
    std::ostringstream failure;
    task.explainFailure(failure);
    RCLCPP_ERROR(node->get_logger(), "planning failure:\n%s", failure.str().c_str());
    task.publishAllSolutions(false);
    RCLCPP_INFO(node->get_logger(), "Introspection active. STEP 3 FAIL. Ctrl+C to exit.");
    spinner.join();
    return 3;
  }

  const auto& solution = *task.solutions().front();
  const auto* start_interface = solution.start();
  const auto* end_interface = solution.end();
  if (!end_interface || !end_interface->scene())
  {
    RCLCPP_ERROR(node->get_logger(), "Solution has no end scene");
    shutdownSpinner(executor, spinner);
    return 3;
  }

  const auto& end_scene_state = end_interface->scene()->getCurrentState();
  const auto scene_end_joints = jointsFromState(end_scene_state);
  RCLCPP_INFO(node->get_logger(), "Current/Home joints:\n%s", formatJoints(actual).c_str());
  RCLCPP_INFO(node->get_logger(), "Solution final RobotState joints:\n%s",
              formatJoints(scene_end_joints).c_str());
  if (start_interface && start_interface->scene())
  {
    RCLCPP_INFO(node->get_logger(), "Solution start RobotState joints:\n%s",
                formatJoints(jointsFromState(start_interface->scene()->getCurrentState())).c_str());
  }

  moveit_task_constructor_msgs::msg::Solution solution_msg;
  solution.toMsg(solution_msg);
  const auto waypoints = extractArmWaypoints(solution_msg);
  RCLCPP_INFO(node->get_logger(), "trajectory points = %zu", waypoints.size());
  if (waypoints.empty())
  {
    RCLCPP_ERROR(node->get_logger(), "Solution trajectory has no arm waypoints");
    shutdownSpinner(executor, spinner);
    return 3;
  }
  RCLCPP_INFO(node->get_logger(), "first joints:\n%s", formatJoints(waypoints.front()).c_str());
  RCLCPP_INFO(node->get_logger(), "last joints:\n%s", formatJoints(waypoints.back()).c_str());

  const double start_vs_first = maxHomeError(waypoints.front(), actual);
  const double last_vs_scene = maxHomeError(waypoints.back(), scene_end_joints);
  const double last_vs_home = maxHomeError(waypoints.back(), actual);
  RCLCPP_INFO(node->get_logger(), "first vs Current/Home max error: %.6f rad", start_vs_first);
  RCLCPP_INFO(node->get_logger(), "last vs Solution end scene max error: %.6f rad", last_vs_scene);
  RCLCPP_INFO(node->get_logger(), "last vs Current/Home max error: %.6f rad", last_vs_home);
  if (last_vs_home < 1e-4)
  {
    RCLCPP_ERROR(node->get_logger(), "end state extraction 可能错误: last joints == Current/Home");
  }

  moveit::core::RobotState final_state(end_scene_state);
  applyArmJoints(final_state, waypoints.back());

  const std::string model_frame = robot_model->getModelFrame();
  const std::string planning_scene_frame = end_interface->scene()->getPlanningFrame();
  if (!final_state.getRobotModel()->hasLinkModel(ee_link) ||
      !final_state.getRobotModel()->hasLinkModel("base_link"))
  {
    RCLCPP_ERROR(node->get_logger(), "RobotState missing gripper_tcp or base_link");
    shutdownSpinner(executor, spinner);
    return 3;
  }

  const Eigen::Isometry3d t_model_tcp = final_state.getGlobalLinkTransform(ee_link);
  const Eigen::Isometry3d t_model_base = final_state.getGlobalLinkTransform("base_link");
  const Eigen::Isometry3d t_base_tcp = t_model_base.inverse() * t_model_tcp;
  const Eigen::Isometry3d t_base_target = poseToIso(pregrasp.pose);
  const Eigen::Isometry3d t_model_target = t_model_base * t_base_target;

  RCLCPP_INFO(node->get_logger(), "PlanningScene.getPlanningFrame():\n%s",
              planning_scene_frame.c_str());
  RCLCPP_INFO(node->get_logger(), "getGlobalLinkTransform(\"%s\")", ee_link.c_str());
  RCLCPP_INFO(node->get_logger(), "FK raw frame:\nRobotModel model frame = %s", model_frame.c_str());
  RCLCPP_INFO(node->get_logger(), "FK raw xyz / xyzw:\n%s",
              formatIso(t_model_tcp, model_frame).c_str());

  RCLCPP_INFO(node->get_logger(), "========== POSE CROSS CHECK ==========");
  RCLCPP_INFO(node->get_logger(), "A. Target PreGrasp in base_link\n%s",
              formatIso(t_base_target, "base_link").c_str());
  RCLCPP_INFO(node->get_logger(), "B. Final TCP raw FK in model frame\n%s",
              formatIso(t_model_tcp, model_frame).c_str());
  RCLCPP_INFO(node->get_logger(), "C. Final TCP transformed into base_link\n%s",
              formatIso(t_base_tcp, "base_link").c_str());
  RCLCPP_INFO(node->get_logger(), "D. Target PreGrasp in model frame\n%s",
              formatIso(t_model_target, model_frame).c_str());

  double ac_pos = 0.0;
  double ac_ori = 0.0;
  double bd_pos = 0.0;
  double bd_ori = 0.0;
  poseError(t_base_target, t_base_tcp, ac_pos, ac_ori);
  poseError(t_model_tcp, t_model_target, bd_pos, bd_ori);
  RCLCPP_INFO(node->get_logger(), "A vs C position error = %.6f m", ac_pos);
  RCLCPP_INFO(node->get_logger(), "A vs C orientation error = %.6f deg", ac_ori);
  RCLCPP_INFO(node->get_logger(), "B vs D position error = %.6f m", bd_pos);
  RCLCPP_INFO(node->get_logger(), "B vs D orientation error = %.6f deg", bd_ori);

  const bool goal_ok = ac_pos <= pos_tol && ac_ori <= ori_tol_deg;
  RCLCPP_INFO(node->get_logger(), "TCP position error: %.6f m (tol=%.6f)", ac_pos, pos_tol);
  RCLCPP_INFO(node->get_logger(), "TCP orientation error: %.6f deg (tol=%.6f)", ac_ori, ori_tol_deg);
  RCLCPP_INFO(node->get_logger(), "Goal verification: %s", goal_ok ? "PASS" : "FAIL");

  task.publishAllSolutions(false);
  task.enableIntrospection(true);

  const auto after_msg = waitForFreshJoints(node, std::chrono::seconds(5));
  if (after_msg)
  {
    const auto after = jointsFromMsg(*after_msg);
    const double drift = maxHomeError(after, actual);
    RCLCPP_INFO(node->get_logger(), "JOINT STATES BEFORE:\n%s", formatJoints(actual).c_str());
    RCLCPP_INFO(node->get_logger(), "JOINT STATES AFTER:\n%s", formatJoints(after).c_str());
    RCLCPP_INFO(node->get_logger(), "Max joint drift after plan: %.6f rad", drift);
    if (drift > 0.02)
    {
      RCLCPP_ERROR(node->get_logger(),
                   "Joints changed after plan-only node. Treat as unexpected motion.");
    }
    else
    {
      RCLCPP_INFO(node->get_logger(), "Robot moved because of this node? NO");
    }
  }

  RCLCPP_INFO(node->get_logger(),
              "Introspection is active for RViz. THIS STEP IS PLAN-ONLY. Ctrl+C to exit.");
  spinner.join();
  return goal_ok ? 0 : 4;
}
