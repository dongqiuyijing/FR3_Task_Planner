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
const char* kPilzPipeline = "pilz_industrial_motion_planner";
const char* kPlannerId = "LIN";
const std::vector<std::string> kArmJoints = { "j1", "j2", "j3", "j4", "j5", "j6" };
constexpr double kLinDistanceM = 0.005;
constexpr double kLateralTolM = 0.001;
constexpr double kPosTolM = 0.005;
constexpr double kOriTolDeg = 3.0;

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

geometry_msgs::msg::PoseStamped isoToPoseStamped(const Eigen::Isometry3d& transform,
                                                 const std::string& frame)
{
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = frame;
  pose.pose.position.x = transform.translation().x();
  pose.pose.position.y = transform.translation().y();
  pose.pose.position.z = transform.translation().z();
  const Eigen::Quaterniond q(transform.rotation());
  pose.pose.orientation.x = q.x();
  pose.pose.orientation.y = q.y();
  pose.pose.orientation.z = q.z();
  pose.pose.orientation.w = q.w();
  return pose;
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

double maxJointError(const std::map<std::string, double>& a, const std::map<std::string, double>& b)
{
  double max_error = 0.0;
  for (const auto& name : kArmJoints)
  {
    max_error = std::max(max_error, std::abs(a.at(name) - b.at(name)));
  }
  return max_error;
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

std::vector<std::map<std::string, double>> extractArmWaypoints(
    const moveit_task_constructor_msgs::msg::Solution& solution_msg)
{
  std::vector<std::map<std::string, double>> points;
  for (const auto& sub : solution_msg.sub_trajectory)
  {
    const auto& joint_trajectory = sub.trajectory.joint_trajectory;
    for (const auto& point : joint_trajectory.points)
    {
      std::map<std::string, double> joints;
      const size_t n = std::min(joint_trajectory.joint_names.size(), point.positions.size());
      for (size_t i = 0; i < n; ++i)
      {
        joints[joint_trajectory.joint_names[i]] = point.positions[i];
      }
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

void logSceneObjects(const rclcpp::Node::SharedPtr& node)
{
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
  }
  RCLCPP_INFO(node->get_logger(), "Known collision objects: %s",
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

struct LinCandidate
{
  std::string name;
  Eigen::Vector3d model_delta;
  bool in_tcp_frame;
};

Eigen::Isometry3d makeGoalModel(const Eigen::Isometry3d& t_model_tcp, const LinCandidate& candidate)
{
  if (candidate.in_tcp_frame)
  {
    return t_model_tcp * Eigen::Translation3d(candidate.model_delta);
  }
  Eigen::Isometry3d goal = t_model_tcp;
  goal.translation() += candidate.model_delta;
  return goal;
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("fr3_mtc_pilz_smoke_test", options);
  RCLCPP_INFO(node->get_logger(), "STEP 4A PLAN-ONLY. NO TRAJECTORY EXECUTION IS PERFORMED.");

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
    RCLCPP_ERROR(node->get_logger(), "/joint_states missing or incomplete. STEP 4A FAIL.");
    shutdownSpinner(executor, spinner);
    return 1;
  }
  const auto actual = jointsFromMsg(*before_msg);
  logSceneObjects(node);

  moveit::task_constructor::Task task("", true);
  task.setName("FR3 Pilz LIN Capability Test");
  task.loadRobotModel(node);
  const auto robot_model = task.getRobotModel();
  if (!robot_model || !robot_model->hasJointModelGroup(kPlanningGroup) ||
      !robot_model->hasLinkModel(kEeLink) || !robot_model->hasLinkModel("base_link"))
  {
    RCLCPP_ERROR(node->get_logger(), "Robot model / group / links missing");
    shutdownSpinner(executor, spinner);
    return 1;
  }

  moveit::core::RobotState probe(robot_model);
  probe.setToDefaultValues();
  applyJoints(probe, actual);
  const std::string model_frame = robot_model->getModelFrame();
  const Eigen::Isometry3d t_model_tcp = probe.getGlobalLinkTransform(kEeLink);
  const Eigen::Isometry3d t_model_base = probe.getGlobalLinkTransform("base_link");
  const Eigen::Isometry3d t_base_tcp = t_model_base.inverse() * t_model_tcp;

  RCLCPP_INFO(node->get_logger(), "========== STEP 4A PILZ DIAGNOSTICS ==========");
  RCLCPP_INFO(node->get_logger(), "RobotModel model frame:\n%s", model_frame.c_str());
  RCLCPP_INFO(node->get_logger(), "Target frame:\n%s", kPlanningFrame);
  RCLCPP_INFO(node->get_logger(), "IK frame:\n%s", kEeLink);
  RCLCPP_INFO(node->get_logger(), "Pipeline:\n%s", kPilzPipeline);
  RCLCPP_INFO(node->get_logger(), "planner_id:\n%s", kPlannerId);
  RCLCPP_INFO(node->get_logger(), "Current Home joints:\n%s", formatJoints(actual).c_str());
  RCLCPP_INFO(node->get_logger(), "Current TCP raw FK:\n%s", formatIso(t_model_tcp, model_frame).c_str());
  RCLCPP_INFO(node->get_logger(), "Current TCP in base_link:\n%s", formatIso(t_base_tcp, "base_link").c_str());

  const std::vector<LinCandidate> candidates = {
    { "world +Z 5mm (away from table)", Eigen::Vector3d(0.0, 0.0, kLinDistanceM), false },
    { "TCP +Z 5mm", Eigen::Vector3d(0.0, 0.0, kLinDistanceM), true },
    { "TCP -Z 5mm", Eigen::Vector3d(0.0, 0.0, -kLinDistanceM), true },
  };

  bool planned = false;
  size_t num_solutions = 0;
  Eigen::Isometry3d t_base_target = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d t_model_target = Eigen::Isometry3d::Identity();
  std::string chosen;
  for (const auto& candidate : candidates)
  {
    t_model_target = makeGoalModel(t_model_tcp, candidate);
    t_base_target = t_model_base.inverse() * t_model_target;
    const auto goal = isoToPoseStamped(t_base_target, kPlanningFrame);
    chosen = candidate.name;
    RCLCPP_INFO(node->get_logger(), "Trying LIN candidate: %s", chosen.c_str());
    RCLCPP_INFO(node->get_logger(), "Goal in base_link:\n%s", formatIso(t_base_target, "base_link").c_str());
    RCLCPP_INFO(node->get_logger(), "Goal in model frame:\n%s", formatIso(t_model_target, model_frame).c_str());

    moveit::task_constructor::Task attempt("", true);
    attempt.setName("FR3 Pilz LIN " + chosen);
    attempt.loadRobotModel(node);
    attempt.add(std::make_unique<moveit::task_constructor::stages::CurrentState>("CurrentState"));

    auto pipeline =
        std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kPilzPipeline);
    pipeline->setPlannerId(kPlannerId);
    pipeline->setTimeout(10.0);

    auto move_to =
        std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo Pilz LIN", pipeline);
    move_to->setGroup(kPlanningGroup);
    move_to->setIKFrame(kEeLink);
    move_to->setGoal(goal);
    move_to->setTimeout(10.0);
    attempt.add(std::move(move_to));

    try
    {
      attempt.init();
    }
    catch (const std::exception& ex)
    {
      RCLCPP_ERROR(node->get_logger(), "task.init() failed for %s: %s", chosen.c_str(), ex.what());
      continue;
    }

    const auto plan_result = attempt.plan(1);
    num_solutions = attempt.numSolutions();
    std::ostringstream state;
    attempt.printState(state);
    RCLCPP_INFO(node->get_logger(), "planning result: %s", plan_result ? "SUCCESS" : "FAIL");
    RCLCPP_INFO(node->get_logger(), "solution 数量: %zu", num_solutions);
    RCLCPP_INFO(node->get_logger(), "MTC task state:\n%s", state.str().c_str());
    if (!plan_result || num_solutions < 1)
    {
      std::ostringstream failure;
      attempt.explainFailure(failure);
      RCLCPP_WARN(node->get_logger(), "LIN candidate failed:\n%s", failure.str().c_str());
      continue;
    }

    const auto& solution = *attempt.solutions().front();
    moveit_task_constructor_msgs::msg::Solution solution_msg;
    solution.toMsg(solution_msg);
    const auto waypoints = extractArmWaypoints(solution_msg);
    RCLCPP_INFO(node->get_logger(), "trajectory points = %zu", waypoints.size());
    if (waypoints.size() < 2)
    {
      RCLCPP_WARN(node->get_logger(), "LIN candidate produced too few points");
      continue;
    }

    moveit::core::RobotState fk_state(probe);
    std::vector<Eigen::Isometry3d> tcp_model;
    tcp_model.reserve(waypoints.size());
    for (const auto& joints : waypoints)
    {
      applyJoints(fk_state, joints);
      tcp_model.push_back(fk_state.getGlobalLinkTransform(kEeLink));
    }

    const Eigen::Vector3d start_p = tcp_model.front().translation();
    const Eigen::Vector3d goal_p = t_model_target.translation();
    const Eigen::Vector3d line = goal_p - start_p;
    const double line_norm = line.norm();
    double max_lateral = 0.0;
    double max_ori = 0.0;
    const Eigen::Quaterniond start_q(tcp_model.front().rotation());
    for (const auto& tcp : tcp_model)
    {
      const Eigen::Vector3d rel = tcp.translation() - start_p;
      double lateral = rel.norm();
      if (line_norm > 1e-9)
      {
        lateral = (rel - line * (rel.dot(line) / (line_norm * line_norm))).norm();
      }
      max_lateral = std::max(max_lateral, lateral);
      double pos_unused = 0.0;
      double ori = 0.0;
      poseError(tcp_model.front(), tcp, pos_unused, ori);
      max_ori = std::max(max_ori, ori);
    }

    const Eigen::Isometry3d t_model_end = tcp_model.back();
    const Eigen::Isometry3d t_base_end = t_model_base.inverse() * t_model_end;
    double end_pos = 0.0;
    double end_ori = 0.0;
    poseError(t_base_target, t_base_end, end_pos, end_ori);

    RCLCPP_INFO(node->get_logger(), "first joints:\n%s", formatJoints(waypoints.front()).c_str());
    RCLCPP_INFO(node->get_logger(), "last joints:\n%s", formatJoints(waypoints.back()).c_str());
    RCLCPP_INFO(node->get_logger(), "max lateral deviation = %.6f m (tol=%.6f)", max_lateral,
                kLateralTolM);
    RCLCPP_INFO(node->get_logger(), "max orientation deviation = %.6f deg", max_ori);
    RCLCPP_INFO(node->get_logger(), "endpoint position error = %.6f m (tol=%.6f)", end_pos, kPosTolM);
    RCLCPP_INFO(node->get_logger(), "endpoint orientation error = %.6f deg (tol=%.6f)", end_ori,
                kOriTolDeg);
    RCLCPP_INFO(node->get_logger(), "Collision checking: ENABLED");

    const bool lin_ok = max_lateral <= kLateralTolM && end_pos <= kPosTolM && end_ori <= kOriTolDeg;
    attempt.publishAllSolutions(false);
    attempt.enableIntrospection(true);

    const auto after_msg = waitForFreshJoints(node, std::chrono::seconds(5));
    if (after_msg)
    {
      const auto after = jointsFromMsg(*after_msg);
      RCLCPP_INFO(node->get_logger(), "JOINT STATES BEFORE:\n%s", formatJoints(actual).c_str());
      RCLCPP_INFO(node->get_logger(), "JOINT STATES AFTER:\n%s", formatJoints(after).c_str());
      RCLCPP_INFO(node->get_logger(), "Max joint drift after plan: %.6f rad",
                  maxJointError(after, actual));
      RCLCPP_INFO(node->get_logger(), "Robot moved because of this node? %s",
                  maxJointError(after, actual) > 0.02 ? "YES" : "NO");
    }

    RCLCPP_INFO(node->get_logger(), "Goal verification: %s", lin_ok ? "PASS" : "FAIL");
    RCLCPP_INFO(node->get_logger(),
                "Introspection is active for RViz. THIS STEP IS PLAN-ONLY. Ctrl+C to exit.");
    planned = true;
    spinner.join();
    return lin_ok ? 0 : 4;
  }

  if (!planned)
  {
    RCLCPP_ERROR(node->get_logger(),
                 "All Pilz LIN candidates failed. Collision checking stayed ENABLED.");
  }
  shutdownSpinner(executor, spinner);
  return 3;
}
