#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <moveit/robot_model/robot_model.h>
#include <moveit/task_constructor/solvers/pipeline_planner.h>
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/task.h>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <rclcpp/rclcpp.hpp>

namespace
{
constexpr double kDegToRad = M_PI / 180.0;

// Copied from ~/fairino_ws/src/fr_control/config/stage4_config.yaml
// robot.initial_joint_positions. SOURCE UNIT: degree.
// Converted to radian before any MoveIt / MTC JointModelGroup call.
const std::map<std::string, double> kHomeJointPositionsDeg = {
  { "j1", -134.053 },
  { "j2", -123.046 },
  { "j3", -112.585 },
  { "j4", -24.971 },
  { "j5", -34.448 },
  { "j6", 47.587 },
};

// Wrist roll only. URDF j6 limits are +/-3.0543 rad (~+/-175 deg).
// 5 deg keeps the arm posture and avoids swinging toward the table/column.
constexpr double kGoalJ6OffsetDeg = 5.0;

const char* kPlanningGroup = "fairino3_v6_group";
const char* kPipelineName = "ompl";
constexpr size_t kMaxSolutions = 5;

std::map<std::string, double> makeGoalRadians()
{
  std::map<std::string, double> goal_rad;
  for (const auto& item : kHomeJointPositionsDeg)
  {
    goal_rad[item.first] = item.second * kDegToRad;
  }
  goal_rad["j6"] = (kHomeJointPositionsDeg.at("j6") + kGoalJ6OffsetDeg) * kDegToRad;
  return goal_rad;
}

std::string formatJoints(const std::map<std::string, double>& joints_rad)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(6);
  bool first = true;
  for (const auto& name : { "j1", "j2", "j3", "j4", "j5", "j6" })
  {
    const auto it = joints_rad.find(name);
    if (it == joints_rad.end())
    {
      continue;
    }
    if (!first)
    {
      oss << ", ";
    }
    first = false;
    oss << name << "=" << it->second << " rad (" << (it->second / kDegToRad) << " deg)";
  }
  return oss.str();
}

bool overlayRobotDescriptionFromMoveGroup(const rclcpp::Node::SharedPtr& node)
{
  auto client = std::make_shared<rclcpp::SyncParametersClient>(node, "move_group");
  if (!client->wait_for_service(std::chrono::seconds(60)))
  {
    RCLCPP_ERROR(node->get_logger(),
                 "Timed out waiting for /move_group. Start the existing FR3 MoveIt bringup first.");
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
    RCLCPP_INFO(node->get_logger(), "Overlaid %s from /move_group (%zu bytes)",
                parameter.get_name().c_str(), parameter.as_string().size());
  }
  return true;
}

std::map<std::string, double> queryCurrentArmJoints(const rclcpp::Node::SharedPtr& node)
{
  std::map<std::string, double> current;
  auto client = node->create_client<moveit_msgs::srv::GetPlanningScene>("get_planning_scene");
  if (!client->wait_for_service(std::chrono::seconds(30)))
  {
    RCLCPP_WARN(node->get_logger(), "get_planning_scene is not available; cannot log start joints.");
    return current;
  }

  auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
  request->components.components = moveit_msgs::msg::PlanningSceneComponents::ROBOT_STATE;
  auto future = client->async_send_request(request);
  if (future.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
  {
    RCLCPP_WARN(node->get_logger(), "get_planning_scene timed out; cannot log start joints.");
    return current;
  }

  const auto response = future.get();
  const auto& names = response->scene.robot_state.joint_state.name;
  const auto& positions = response->scene.robot_state.joint_state.position;
  const size_t n = std::min(names.size(), positions.size());
  for (size_t i = 0; i < n; ++i)
  {
    if (kHomeJointPositionsDeg.count(names[i]) != 0)
    {
      current[names[i]] = positions[i];
    }
  }
  return current;
}

void logRobotModel(const rclcpp::Node::SharedPtr& node, const moveit::core::RobotModelConstPtr& model)
{
  if (!model)
  {
    RCLCPP_ERROR(node->get_logger(), "Robot model not found");
    return;
  }
  RCLCPP_INFO(node->get_logger(), "Loaded robot model: %s", model->getName().c_str());
  if (!model->hasJointModelGroup(kPlanningGroup))
  {
    RCLCPP_ERROR(node->get_logger(), "Planning group not found: %s", kPlanningGroup);
    return;
  }
  const auto* group = model->getJointModelGroup(kPlanningGroup);
  std::ostringstream joints;
  for (const auto& name : group->getVariableNames())
  {
    if (!joints.str().empty())
    {
      joints << ", ";
    }
    joints << name;
  }
  const auto* tip = group->getOnlyOneEndEffectorTip();
  RCLCPP_INFO(node->get_logger(), "planning group: %s", kPlanningGroup);
  RCLCPP_INFO(node->get_logger(), "base frame: %s", model->getModelFrame().c_str());
  RCLCPP_INFO(node->get_logger(), "EE/TCP: %s", tip ? tip->getName().c_str() : "(none)");
  RCLCPP_INFO(node->get_logger(), "joint names: %s", joints.str().c_str());
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("fr3_mtc_smoke_test", options);

  RCLCPP_INFO(node->get_logger(), "THIS STEP IS PLAN-ONLY. NO TRAJECTORY EXECUTION IS PERFORMED.");

  if (!overlayRobotDescriptionFromMoveGroup(node))
  {
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  const auto goal_rad = makeGoalRadians();
  const auto start_rad = queryCurrentArmJoints(node);

  moveit::task_constructor::Task task("", true);
  task.setName("FR3 MTC Smoke Test");
  task.loadRobotModel(node);
  logRobotModel(node, task.getRobotModel());
  if (!task.getRobotModel() || !task.getRobotModel()->hasJointModelGroup(kPlanningGroup))
  {
    executor.cancel();
    spinner.join();
    rclcpp::shutdown();
    return 1;
  }

  auto current_state = std::make_unique<moveit::task_constructor::stages::CurrentState>("CurrentState");
  task.add(std::move(current_state));

  auto pipeline = std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kPipelineName);

  auto move_to = std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo", pipeline);
  move_to->setGroup(kPlanningGroup);
  move_to->setGoal(goal_rad);
  task.add(std::move(move_to));

  RCLCPP_INFO(node->get_logger(), "MTC Task: CurrentState -> MoveTo");
  RCLCPP_INFO(node->get_logger(), "Start (CurrentState, radian): %s",
              start_rad.empty() ? "(unavailable)" : formatJoints(start_rad).c_str());
  RCLCPP_INFO(node->get_logger(), "Goal (Home + j6 5 deg, radian): %s", formatJoints(goal_rad).c_str());
  RCLCPP_INFO(node->get_logger(), "Planner: %s (FR3 MoveIt default OMPL / RRTConnect)", kPipelineName);

  try
  {
    task.init();
  }
  catch (const std::exception& ex)
  {
    RCLCPP_ERROR(node->get_logger(), "task.init() failed: %s", ex.what());
    executor.cancel();
    spinner.join();
    rclcpp::shutdown();
    return 1;
  }

  const auto plan_result = task.plan(kMaxSolutions);
  const size_t num_solutions = task.numSolutions();
  std::ostringstream state;
  task.printState(state);
  RCLCPP_INFO(node->get_logger(), "planning result: %s", plan_result ? "SUCCESS" : "FAIL");
  RCLCPP_INFO(node->get_logger(), "solution 数量: %zu", num_solutions);
  RCLCPP_INFO(node->get_logger(), "目标 joint state: %s", formatJoints(goal_rad).c_str());
  RCLCPP_INFO(node->get_logger(), "MTC task state:\n%s", state.str().c_str());
  if (!plan_result)
  {
    std::ostringstream failure;
    task.explainFailure(failure);
    RCLCPP_ERROR(node->get_logger(), "planning failure:\n%s", failure.str().c_str());
  }

  task.publishAllSolutions(false);
  task.enableIntrospection(true);

  RCLCPP_INFO(node->get_logger(),
              "Introspection is active for RViz Motion Planning Tasks. "
              "THIS STEP IS PLAN-ONLY. NO TRAJECTORY EXECUTION IS PERFORMED. Ctrl+C to exit.");

  spinner.join();
  return plan_result && num_solutions >= 1 ? 0 : 2;
}
