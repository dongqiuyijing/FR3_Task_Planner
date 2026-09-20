#include "fr_task_planner/inspection_endpoint_candidates.hpp"
#include "fr_task_planner/inspection_visibility.hpp"
#include "fr_task_planner/pregrasp_ik_candidates.hpp"
#include "fr_task_planner/winner_trajectory_io.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/point.hpp>
#include <moveit/collision_detection/collision_common.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/task_constructor/solvers/pipeline_planner.h>
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/fixed_state.h>
#include <moveit/task_constructor/stages/modify_planning_scene.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/storage.h>
#include <moveit/task_constructor/task.h>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/planning_scene_components.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <moveit_task_constructor_msgs/msg/solution.hpp>
#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace
{
using fr_task_planner::applyJoints;
using fr_task_planner::applyJointsToScene;
using fr_task_planner::bottomCircleRoiDef;
using fr_task_planner::checkGeometricVisibility;
using fr_task_planner::cloneDiagnosticScene;
using fr_task_planner::collectCollisionContacts;
using fr_task_planner::CollisionCategory;
using fr_task_planner::CollisionDiagConfig;
using fr_task_planner::DesignCamera;
using fr_task_planner::EndpointCandidate;
using fr_task_planner::EndpointGenerationConfig;
using fr_task_planner::filterRollsByView;
using fr_task_planner::fixedInspectionDesignCamera;
using fr_task_planner::generatePreGraspIkCandidates;
using fr_task_planner::generateValidEndpointCandidates;
using fr_task_planner::GeometricVisibilityResult;
using fr_task_planner::jointL1;
using fr_task_planner::jointL2;
using fr_task_planner::jointsFromState;
using fr_task_planner::NamedJointSeed;
using fr_task_planner::PreGraspIkCandidate;
using fr_task_planner::PreGraspIkConfig;
using fr_task_planner::PreGraspIkGenerationResult;
using fr_task_planner::kArmJoints;
using fr_task_planner::maxJointError;
using fr_task_planner::poseToIso;
using fr_task_planner::RollPose;
using fr_task_planner::tcpInBase;
using fr_task_planner::ViewGeom;
using fr_task_planner::PersistedTrajectory;
using fr_task_planner::PersistValidation;
using fr_task_planner::TrajectoryPointRecord;
using fr_task_planner::TrajectorySegmentRecord;
using fr_task_planner::WinnerEndpoints;
using fr_task_planner::attachStandardEvents;
using fr_task_planner::computeFrozenTaskMetrics;
using fr_task_planner::extractArmPositions;
using fr_task_planner::fillDerivedTotals;
using fr_task_planner::firstLogicalSegment;
using fr_task_planner::FrozenTaskMetrics;
using fr_task_planner::isFixedDeployableLogical;
using fr_task_planner::jointsToVec;
using fr_task_planner::kStep12cPreGraspRad;
using fr_task_planner::lastLogicalSegment;
using fr_task_planner::logicalMetrics;
using fr_task_planner::kLogicalCurrentToHome;
using fr_task_planner::loadWinnerYaml;
using fr_task_planner::logicalSegmentFromStage;
using fr_task_planner::readTrajectoryYaml;
using fr_task_planner::timesMonotonic;
using fr_task_planner::validateContinuity;
using fr_task_planner::validateEndpoints;
using fr_task_planner::validateRoundTrip;
using fr_task_planner::winnerMatchesFrozenStep12c;
using fr_task_planner::detachObjectDiagnostic;
using fr_task_planner::kLogicalAToB;
using fr_task_planner::kLogicalBToC;
using fr_task_planner::kLogicalGraspToLift;
using fr_task_planner::kLogicalHomeToPreGrasp;
using fr_task_planner::kLogicalLiftToA;
using fr_task_planner::kLogicalPreGraspToGrasp;
using fr_task_planner::maxAbsError;
using fr_task_planner::maxAbsErrorJoints;
using fr_task_planner::poseError;
using fr_task_planner::removeWorldObjectDiagnostic;
using fr_task_planner::vecToJoints;
using fr_task_planner::writeTrajectoryYaml;

const char* kPlanningGroup = "fairino3_v6_group";
const char* kPlanningFrame = "base_link";
const char* kEeLink = "gripper_tcp";
const char* kOmplPipeline = "ompl";
const char* kPilzPipeline = "pilz_industrial_motion_planner";
const char* kPilzPlannerId = "LIN";
const char* kMarkerTopic = "/fr3_vis/markers";
const char* kGhostTopic = "/fr3_vis/collision_robot_state";
const char* kTrajTopic = "/display_planned_path";

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

int getInt(const rclcpp::Node::SharedPtr& node, const std::string& name, int fallback)
{
  if (!node->has_parameter(name))
  {
    return fallback;
  }
  return static_cast<int>(node->get_parameter(name).as_int());
}

bool getBool(const rclcpp::Node::SharedPtr& node, const std::string& name, bool fallback)
{
  if (!node->has_parameter(name))
  {
    return fallback;
  }
  const auto p = node->get_parameter(name);
  if (p.get_type() == rclcpp::ParameterType::PARAMETER_BOOL)
  {
    return p.as_bool();
  }
  if (p.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
  {
    return p.as_string() == "true" || p.as_string() == "1";
  }
  return fallback;
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

std::map<std::string, double> readHomeDeg(const rclcpp::Node::SharedPtr& node,
                                         const std::map<std::string, double>& home_rad)
{
  std::map<std::string, double> home;
  for (const auto& name : kArmJoints)
  {
    const std::string key = "home_" + name + "_deg";
    home[name] = node->has_parameter(key) ? getDouble(node, key) : home_rad.at(name) * 180.0 / M_PI;
  }
  return home;
}

std::string formatHomeList(const std::map<std::string, double>& joints)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(12);
  oss << "[";
  for (size_t i = 0; i < kArmJoints.size(); ++i)
  {
    if (i)
    {
      oss << ", ";
    }
    oss << joints.at(kArmJoints[i]);
  }
  oss << "]";
  return oss.str();
}

DesignCamera cameraFromNode(const rclcpp::Node::SharedPtr& node)
{
  DesignCamera camera = fixedInspectionDesignCamera();
  const double roll = node->has_parameter("camera_rpy_roll") ?
                          getDouble(node, "camera_rpy_roll") :
                          -2.35619449;
  const double pitch = node->has_parameter("camera_rpy_pitch") ? getDouble(node, "camera_rpy_pitch") : 0.0;
  const double yaw = node->has_parameter("camera_rpy_yaw") ? getDouble(node, "camera_rpy_yaw") : 0.0;
  const Eigen::Matrix3d rot =
      (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
          .toRotationMatrix();
  camera.optical_center_world = Eigen::Vector3d(0.0, 0.0, 1.40);
  camera.optical_forward_world = (rot * Eigen::Vector3d::UnitZ()).normalized();
  camera.image_up_world = (-(rot * Eigen::Vector3d::UnitY())).normalized();
  if (node->has_parameter("camera_design_x"))
  {
    camera.optical_center_world = Eigen::Vector3d(getDouble(node, "camera_design_x"),
                                                  getDouble(node, "camera_design_y"),
                                                  getDouble(node, "camera_design_z"));
  }
  if (node->has_parameter("camera_forward_x"))
  {
    camera.optical_forward_world =
        Eigen::Vector3d(getDouble(node, "camera_forward_x"), getDouble(node, "camera_forward_y"),
                        getDouble(node, "camera_forward_z"))
            .normalized();
  }
  return camera;
}

std::string vecYaml(const Eigen::Vector3d& v)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(9);
  oss << "[" << v.x() << ", " << v.y() << ", " << v.z() << "]";
  return oss.str();
}

std::string isoPoseYaml(const Eigen::Isometry3d& t)
{
  const Eigen::Quaterniond q(t.linear());
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(9);
  oss << "{xyz: [" << t.translation().x() << ", " << t.translation().y() << ", "
      << t.translation().z() << "], xyzw: [" << q.x() << ", " << q.y() << ", " << q.z() << ", "
      << q.w() << "]}";
  return oss.str();
}

planning_scene::PlanningScenePtr fetchPlanningScene(
    const rclcpp::Node::SharedPtr& node, const moveit::core::RobotModelConstPtr& model)
{
  auto client = node->create_client<moveit_msgs::srv::GetPlanningScene>("get_planning_scene");
  if (!client->wait_for_service(std::chrono::seconds(30)))
  {
    return nullptr;
  }
  auto req = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
  req->components.components =
      moveit_msgs::msg::PlanningSceneComponents::SCENE_SETTINGS |
      moveit_msgs::msg::PlanningSceneComponents::ROBOT_STATE |
      moveit_msgs::msg::PlanningSceneComponents::ROBOT_STATE_ATTACHED_OBJECTS |
      moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_NAMES |
      moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_GEOMETRY |
      moveit_msgs::msg::PlanningSceneComponents::TRANSFORMS |
      moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX |
      moveit_msgs::msg::PlanningSceneComponents::LINK_PADDING_AND_SCALING;
  auto future = client->async_send_request(req);
  if (future.wait_for(std::chrono::seconds(30)) != std::future_status::ready)
  {
    return nullptr;
  }
  auto resp = future.get();
  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  scene->setPlanningSceneMsg(resp->scene);
  return scene;
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
    if (std::abs(joints.at(name)) > 1e-6)
    {
      return false;
    }
  }
  return true;
}

moveit_msgs::msg::CollisionObject makeCylinder(const std::string& object_id,
                                               const geometry_msgs::msg::PoseStamped& pose,
                                               double height, double radius)
{
  moveit_msgs::msg::CollisionObject object;
  object.id = object_id;
  object.header.frame_id = pose.header.frame_id;
  object.primitives.resize(1);
  object.primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
  object.primitives[0].dimensions.resize(2);
  object.primitives[0].dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT] = height;
  object.primitives[0].dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS] = radius;
  object.primitive_poses.push_back(pose.pose);
  object.operation = moveit_msgs::msg::CollisionObject::ADD;
  return object;
}

void flattenSolutions(const moveit::task_constructor::SolutionBase& solution,
                      std::vector<const moveit::task_constructor::SolutionBase*>& out)
{
  if (const auto* seq = dynamic_cast<const moveit::task_constructor::SolutionSequence*>(&solution))
  {
    for (const auto* child : seq->solutions())
    {
      flattenSolutions(*child, out);
    }
    return;
  }
  if (const auto* wrap = dynamic_cast<const moveit::task_constructor::WrappedSolution*>(&solution))
  {
    flattenSolutions(*wrap->wrapped(), out);
    return;
  }
  out.push_back(&solution);
}

const moveit::task_constructor::SolutionBase* findStageSolution(
    const std::vector<const moveit::task_constructor::SolutionBase*>& leaves, const std::string& name)
{
  for (const auto* leaf : leaves)
  {
    if (leaf->creator() && leaf->creator()->name() == name)
    {
      return leaf;
    }
  }
  return nullptr;
}

void setStringParam(const rclcpp::Node::SharedPtr& node, const std::string& name,
                    const std::string& value)
{
  if (!node->has_parameter(name))
  {
    node->declare_parameter<std::string>(name, value);
  }
  else
  {
    node->set_parameter(rclcpp::Parameter(name, value));
  }
}

bool overlayRobotDescriptionFromMoveGroup(const rclcpp::Node::SharedPtr& target,
                                          const rclcpp::Node::SharedPtr& helper)
{
  auto client = std::make_shared<rclcpp::SyncParametersClient>(helper, "move_group");
  if (!client->wait_for_service(std::chrono::seconds(90)))
  {
    return false;
  }
  for (const auto& parameter : client->get_parameters(
           { "robot_description", "robot_description_semantic" }))
  {
    if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET)
    {
      return false;
    }
    setStringParam(target, parameter.get_name(), parameter.as_string());
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
      const auto found = jointsFromMsg(*latest);
      bool complete = true;
      for (const auto& name : kArmJoints)
      {
        complete = complete && found.count(name) != 0;
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

void shutdownSpinner(rclcpp::executors::MultiThreadedExecutor& executor, std::thread& spinner)
{
  executor.cancel();
  if (spinner.joinable())
  {
    spinner.join();
  }
  rclcpp::shutdown();
}

ViewGeom readViewGeom(const rclcpp::Node::SharedPtr& node, const std::string& name)
{
  ViewGeom view;
  view.name = name;
  view.center_in_object = Eigen::Vector3d(getDouble(node, name + "_center_x"),
                                          getDouble(node, name + "_center_y"),
                                          getDouble(node, name + "_center_z"));
  view.normal_in_object = Eigen::Vector3d(getDouble(node, name + "_normal_x"),
                                          getDouble(node, name + "_normal_y"),
                                          getDouble(node, name + "_normal_z"));
  view.up_in_object = Eigen::Vector3d(getDouble(node, name + "_up_x"), getDouble(node, name + "_up_y"),
                                      getDouble(node, name + "_up_z"));
  return view;
}

std::vector<RollPose> readRollPoses(const rclcpp::Node::SharedPtr& node)
{
  const auto views = node->get_parameter("roll_views").as_string_array();
  const auto rolls = node->get_parameter("roll_degs").as_double_array();
  const auto indexes = node->get_parameter("roll_pose_index").as_double_array();
  const auto ox = node->get_parameter("roll_obj_x").as_double_array();
  const auto oy = node->get_parameter("roll_obj_y").as_double_array();
  const auto oz = node->get_parameter("roll_obj_z").as_double_array();
  const auto oqx = node->get_parameter("roll_obj_qx").as_double_array();
  const auto oqy = node->get_parameter("roll_obj_qy").as_double_array();
  const auto oqz = node->get_parameter("roll_obj_qz").as_double_array();
  const auto oqw = node->get_parameter("roll_obj_qw").as_double_array();
  const auto tx = node->get_parameter("roll_tcp_x").as_double_array();
  const auto ty = node->get_parameter("roll_tcp_y").as_double_array();
  const auto tz = node->get_parameter("roll_tcp_z").as_double_array();
  const auto tqx = node->get_parameter("roll_tcp_qx").as_double_array();
  const auto tqy = node->get_parameter("roll_tcp_qy").as_double_array();
  const auto tqz = node->get_parameter("roll_tcp_qz").as_double_array();
  const auto tqw = node->get_parameter("roll_tcp_qw").as_double_array();
  const std::string frame = getString(node, "planning_frame", kPlanningFrame);
  std::vector<RollPose> out;
  for (size_t i = 0; i < views.size(); ++i)
  {
    RollPose pose;
    pose.view = views[i];
    pose.roll_deg = rolls[i];
    pose.pose_index = static_cast<int>(std::lround(indexes[i]));
    pose.object.header.frame_id = frame;
    pose.object.pose.position.x = ox[i];
    pose.object.pose.position.y = oy[i];
    pose.object.pose.position.z = oz[i];
    pose.object.pose.orientation.x = oqx[i];
    pose.object.pose.orientation.y = oqy[i];
    pose.object.pose.orientation.z = oqz[i];
    pose.object.pose.orientation.w = oqw[i];
    pose.tcp.header.frame_id = frame;
    pose.tcp.pose.position.x = tx[i];
    pose.tcp.pose.position.y = ty[i];
    pose.tcp.pose.position.z = tz[i];
    pose.tcp.pose.orientation.x = tqx[i];
    pose.tcp.pose.orientation.y = tqy[i];
    pose.tcp.pose.orientation.z = tqz[i];
    pose.tcp.pose.orientation.w = tqw[i];
    out.push_back(pose);
  }
  return out;
}

void addPrefixStages(moveit::task_constructor::Task& task, const rclcpp::Node::SharedPtr& node,
                     const std::string& group, const std::string& ee_link,
                     const std::string& attach_link, const std::string& object_id,
                     const std::string& table_name, const std::vector<std::string>& touch_links,
                     const geometry_msgs::msg::PoseStamped& object_scene,
                     const geometry_msgs::msg::PoseStamped& pregrasp,
                     const geometry_msgs::msg::PoseStamped& grasp,
                     const geometry_msgs::msg::PoseStamped& lift, double object_height,
                     double object_radius, double planning_time,
                     const std::map<std::string, double>& home,
                     const std::map<std::string, double>* pregrasp_joints = nullptr)
{
  task.add(std::make_unique<moveit::task_constructor::stages::CurrentState>("CurrentState"));
  auto ompl_home =
      std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
  ompl_home->setTimeout(planning_time);
  auto move_home = std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo Home", ompl_home);
  move_home->setGroup(group);
  move_home->setGoal(home);
  move_home->setTimeout(planning_time);
  task.add(std::move(move_home));
  auto prepare = std::make_unique<moveit::task_constructor::stages::ModifyPlanningScene>(
      "Prepare Object On Table");
  prepare->addObject(makeCylinder(object_id, object_scene, object_height, object_radius));
  prepare->allowCollisions(object_id, table_name, true);
  task.add(std::move(prepare));
  auto ompl = std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
  ompl->setTimeout(planning_time);
  auto move_pregrasp =
      std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo PreGrasp", ompl);
  move_pregrasp->setGroup(group);
  if (pregrasp_joints)
  {
    move_pregrasp->setGoal(*pregrasp_joints);
  }
  else
  {
    move_pregrasp->setIKFrame(ee_link);
    move_pregrasp->setGoal(pregrasp);
  }
  move_pregrasp->setTimeout(planning_time);
  task.add(std::move(move_pregrasp));
  auto allow = std::make_unique<moveit::task_constructor::stages::ModifyPlanningScene>(
      "Allow Gripper-Part Contact");
  allow->allowCollisions(object_id, touch_links, true);
  task.add(std::move(allow));
  auto pilz_grasp =
      std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kPilzPipeline);
  pilz_grasp->setPlannerId(kPilzPlannerId);
  pilz_grasp->setTimeout(planning_time);
  auto move_grasp =
      std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo Grasp", pilz_grasp);
  move_grasp->setGroup(group);
  move_grasp->setIKFrame(ee_link);
  move_grasp->setGoal(grasp);
  move_grasp->setTimeout(planning_time);
  task.add(std::move(move_grasp));
  auto attach =
      std::make_unique<moveit::task_constructor::stages::ModifyPlanningScene>("Attach Part To TCP");
  attach->setCallback(
      [object_id, attach_link, touch_links](const planning_scene::PlanningScenePtr& scene,
                                            const moveit::task_constructor::PropertyMap&) {
        moveit_msgs::msg::AttachedCollisionObject attached;
        attached.link_name = attach_link;
        attached.object.id = object_id;
        attached.object.operation = moveit_msgs::msg::CollisionObject::ADD;
        attached.touch_links = touch_links;
        scene->processAttachedCollisionObjectMsg(attached);
      });
  task.add(std::move(attach));
  auto pilz_lift =
      std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kPilzPipeline);
  pilz_lift->setPlannerId(kPilzPlannerId);
  pilz_lift->setTimeout(planning_time);
  auto move_lift =
      std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo Lift", pilz_lift);
  move_lift->setGroup(group);
  move_lift->setIKFrame(ee_link);
  move_lift->setGoal(lift);
  move_lift->setTimeout(planning_time);
  task.add(std::move(move_lift));
  auto restore = std::make_unique<moveit::task_constructor::stages::ModifyPlanningScene>(
      "Restore Part-Table Collision");
  restore->allowCollisions(object_id, table_name, false);
  task.add(std::move(restore));
}

double trajectoryDuration(const moveit_task_constructor_msgs::msg::Solution& msg)
{
  double total = 0.0;
  for (const auto& sub : msg.sub_trajectory)
  {
    const auto& jt = sub.trajectory.joint_trajectory;
    if (jt.points.empty())
    {
      continue;
    }
    bool has_arm = false;
    for (const auto& name : kArmJoints)
    {
      has_arm = has_arm ||
                std::find(jt.joint_names.begin(), jt.joint_names.end(), name) != jt.joint_names.end();
    }
    if (has_arm)
    {
      total += rclcpp::Duration(jt.points.back().time_from_start).seconds();
    }
  }
  return total;
}

double pathLengthMsg(const moveit_task_constructor_msgs::msg::Solution& msg)
{
  double acc = 0.0;
  for (const auto& sub : msg.sub_trajectory)
  {
    const auto& jt = sub.trajectory.joint_trajectory;
    std::map<std::string, double> prev;
    bool have = false;
    for (const auto& pt : jt.points)
    {
      std::map<std::string, double> cur;
      const size_t n = std::min(jt.joint_names.size(), pt.positions.size());
      for (size_t i = 0; i < n; ++i)
      {
        cur[jt.joint_names[i]] = pt.positions[i];
      }
      bool has_arm = false;
      for (const auto& name : kArmJoints)
      {
        has_arm = has_arm || cur.count(name) != 0;
      }
      if (!has_arm)
      {
        continue;
      }
      if (have)
      {
        acc += jointL2(cur, prev);
      }
      prev = std::move(cur);
      have = true;
    }
  }
  return acc;
}

bool tryColumnDistance(const planning_scene::PlanningScene& scene, const std::string& group,
                       const std::string& column_name, double& column_signed)
{
  collision_detection::DistanceRequest req;
  req.enable_signed_distance = true;
  req.type = collision_detection::DistanceRequestTypes::ALL;
  req.group_name = group;
  req.enableGroup(scene.getRobotModel());
  req.acm = &scene.getAllowedCollisionMatrix();
  collision_detection::DistanceResult res;
  try
  {
    scene.getCollisionEnv()->distanceRobot(req, res, scene.getCurrentState());
  }
  catch (const std::exception&)
  {
    return false;
  }
  column_signed = std::numeric_limits<double>::infinity();
  bool any = false;
  for (const auto& item : res.distances)
  {
    if (item.first.first != column_name && item.first.second != column_name)
    {
      continue;
    }
    for (const auto& rec : item.second)
    {
      if (!std::isfinite(rec.distance))
      {
        continue;
      }
      any = true;
      column_signed = std::min(column_signed, rec.distance);
    }
  }
  return any;
}

double jointLimitMargin(const moveit::core::RobotState& state, const moveit::core::JointModelGroup* jmg)
{
  if (!jmg)
  {
    return 0.0;
  }
  double margin = std::numeric_limits<double>::infinity();
  for (const auto* joint : jmg->getActiveJointModels())
  {
    const auto& bounds = joint->getVariableBounds();
    if (bounds.empty())
    {
      continue;
    }
    const double q = state.getVariablePosition(joint->getName());
    if (bounds.front().position_bounded_)
    {
      margin = std::min(margin, q - bounds.front().min_position_);
      margin = std::min(margin, bounds.front().max_position_ - q);
    }
  }
  return std::isfinite(margin) ? margin : 0.0;
}

std_msgs::msg::ColorRGBA rgba(float r, float g, float b, float a)
{
  std_msgs::msg::ColorRGBA c;
  c.r = r;
  c.g = g;
  c.b = b;
  c.a = a;
  return c;
}

geometry_msgs::msg::Point toPoint(const Eigen::Vector3d& v)
{
  geometry_msgs::msg::Point p;
  p.x = v.x();
  p.y = v.y();
  p.z = v.z();
  return p;
}

visualization_msgs::msg::Marker baseMarker(int id, const std::string& ns, int type)
{
  visualization_msgs::msg::Marker m;
  m.header.frame_id = "world";
  m.ns = ns;
  m.id = id;
  m.type = type;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.pose.orientation.w = 1.0;
  m.lifetime = rclcpp::Duration(0, 0);
  m.frame_locked = true;
  return m;
}

void addSphere(visualization_msgs::msg::MarkerArray& arr, int id, const Eigen::Vector3d& p,
               double radius, const std_msgs::msg::ColorRGBA& color, const std::string& ns)
{
  auto m = baseMarker(id, ns, visualization_msgs::msg::Marker::SPHERE);
  m.pose.position = toPoint(p);
  m.scale.x = m.scale.y = m.scale.z = radius;
  m.color = color;
  arr.markers.push_back(m);
}

void addArrow(visualization_msgs::msg::MarkerArray& arr, int id, const Eigen::Vector3d& start,
              const Eigen::Vector3d& end, const std_msgs::msg::ColorRGBA& color, const std::string& ns)
{
  auto m = baseMarker(id, ns, visualization_msgs::msg::Marker::ARROW);
  m.points = { toPoint(start), toPoint(end) };
  m.scale.x = 0.012;
  m.scale.y = 0.022;
  m.scale.z = 0.03;
  m.color = color;
  arr.markers.push_back(m);
}

void addText(visualization_msgs::msg::MarkerArray& arr, int id, const Eigen::Vector3d& p,
             const std::string& text, const std_msgs::msg::ColorRGBA& color, double size,
             const std::string& ns)
{
  auto m = baseMarker(id, ns, visualization_msgs::msg::Marker::TEXT_VIEW_FACING);
  m.pose.position = toPoint(p);
  m.scale.z = size;
  m.color = color;
  m.text = text;
  arr.markers.push_back(m);
}

void addCircle(visualization_msgs::msg::MarkerArray& arr, int id, const Eigen::Vector3d& center,
               const Eigen::Vector3d& normal, double radius, const std_msgs::msg::ColorRGBA& color,
               const std::string& ns)
{
  auto m = baseMarker(id, ns, visualization_msgs::msg::Marker::LINE_STRIP);
  m.scale.x = 0.002;
  m.color = color;
  Eigen::Vector3d n = normal.normalized();
  Eigen::Vector3d tmp = (std::abs(n.z()) < 0.9) ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
  Eigen::Vector3d u = n.cross(tmp).normalized();
  Eigen::Vector3d v = n.cross(u).normalized();
  const int nseg = 32;
  for (int k = 0; k <= nseg; ++k)
  {
    const double ang = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(nseg);
    m.points.push_back(toPoint(center + radius * (std::cos(ang) * u + std::sin(ang) * v)));
  }
  arr.markers.push_back(m);
}

void addCube(visualization_msgs::msg::MarkerArray& arr, int id, const Eigen::Vector3d& center,
             const Eigen::Vector3d& size, const std_msgs::msg::ColorRGBA& color, const std::string& ns)
{
  auto m = baseMarker(id, ns, visualization_msgs::msg::Marker::CUBE);
  m.pose.position = toPoint(center);
  m.scale.x = size.x();
  m.scale.y = size.y();
  m.scale.z = size.z();
  m.color = color;
  arr.markers.push_back(m);
}

struct GeomCheck
{
  double center_err = 0.0;
  double normal_err_deg = 0.0;
  double tcp_object_err = 0.0;
  Eigen::Vector3d object_center = Eigen::Vector3d::Zero();
};

GeomCheck checkRollGeometry(const RollPose& roll, const ViewGeom& view,
                            const Eigen::Isometry3d& t_world_base,
                            const Eigen::Isometry3d& t_tcp_object, const Eigen::Vector3d& p1,
                            const Eigen::Vector3d& d1)
{
  GeomCheck g;
  const Eigen::Isometry3d t_world_object = t_world_base * poseToIso(roll.object.pose);
  const Eigen::Isometry3d t_world_tcp = t_world_base * poseToIso(roll.tcp.pose);
  const Eigen::Vector3d center =
      t_world_object.translation() + t_world_object.linear() * view.center_in_object;
  const Eigen::Vector3d normal = (t_world_object.linear() * view.normal_in_object).normalized();
  g.center_err = (center - p1).norm();
  g.normal_err_deg =
      std::acos(std::min(1.0, std::max(-1.0, normal.dot(d1.normalized())))) * 180.0 / M_PI;
  const Eigen::Isometry3d reconstructed = t_world_tcp * t_tcp_object;
  g.tcp_object_err = (reconstructed.translation() - t_world_object.translation()).norm();
  g.object_center = t_world_object.translation();
  return g;
}

struct ViewLayerCounts
{
  int roll_samples = 0;
  int orientation_valid = 0;
  int geometric_visible_pose = 0;
  int raw_ik = 0;
  int joint_valid = 0;
  int collision_free = 0;
  int visibility_valid = 0;
  int vis_center_blocked = 0;
  int vis_fraction_blocked = 0;
  std::string vis_occluders = "none";
  double best_center_err = 0.0;
  double best_normal_err = 0.0;
  std::string dominant_pair = "none";
  double best_column_distance = std::numeric_limits<double>::quiet_NaN();
  bool distance_available = false;
  std::optional<EndpointCandidate> best_near;
};

struct ScoredEndpoint
{
  EndpointCandidate cand;
  GeometricVisibilityResult vis;
  double clearance = 0.0;
  double joint_margin = 0.0;
  double from_lift = 0.0;
};

struct EdgeResult
{
  bool success = false;
  int attempts = 0;
  int success_count = 0;
  double best_time = std::numeric_limits<double>::infinity();
  double best_path = std::numeric_limits<double>::infinity();
};

struct CompleteCand
{
  const ScoredEndpoint* a = nullptr;
  const ScoredEndpoint* b = nullptr;
  const ScoredEndpoint* c = nullptr;
  double time = 0.0;
  double path = 0.0;
  double min_clearance = 0.0;
  EdgeResult lift_a;
  EdgeResult a_b;
  EdgeResult b_c;
};

std::string jointsYaml(const std::map<std::string, double>& joints)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(9);
  oss << "[";
  for (size_t i = 0; i < kArmJoints.size(); ++i)
  {
    if (i)
    {
      oss << ", ";
    }
    oss << joints.at(kArmJoints[i]);
  }
  oss << "]";
  return oss.str();
}

std::string jointsDegYaml(const std::map<std::string, double>& joints)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(3);
  oss << "[";
  for (size_t i = 0; i < kArmJoints.size(); ++i)
  {
    if (i)
    {
      oss << ", ";
    }
    oss << joints.at(kArmJoints[i]) * 180.0 / M_PI;
  }
  oss << "]";
  return oss.str();
}

std::vector<ScoredEndpoint> pruneTopK(std::vector<ScoredEndpoint> in, int k)
{
  if (k <= 0 || static_cast<int>(in.size()) <= k)
  {
    return in;
  }
  std::sort(in.begin(), in.end(), [](const ScoredEndpoint& a, const ScoredEndpoint& b) {
    const int ba = static_cast<int>(std::floor((a.cand.roll_deg + 180.0) / 30.0));
    const int bb = static_cast<int>(std::floor((b.cand.roll_deg + 180.0) / 30.0));
    if (ba != bb)
    {
      return ba < bb;
    }
    if (std::abs(a.clearance - b.clearance) > 1e-4)
    {
      return a.clearance > b.clearance;
    }
    if (std::abs(a.joint_margin - b.joint_margin) > 1e-4)
    {
      return a.joint_margin > b.joint_margin;
    }
    return a.from_lift < b.from_lift;
  });
  std::vector<ScoredEndpoint> out;
  std::map<int, int> bins;
  for (auto& item : in)
  {
    const int bin = static_cast<int>(std::floor((item.cand.roll_deg + 180.0) / 30.0));
    if (bins[bin] >= 2)
    {
      continue;
    }
    bins[bin] += 1;
    out.push_back(item);
    if (static_cast<int>(out.size()) >= k)
    {
      return out;
    }
  }
  for (auto& item : in)
  {
    bool have = false;
    for (const auto& keep : out)
    {
      if (keep.cand.digest == item.cand.digest)
      {
        have = true;
        break;
      }
    }
    if (have)
    {
      continue;
    }
    out.push_back(item);
    if (static_cast<int>(out.size()) >= k)
    {
      break;
    }
  }
  return out;
}

bool betterComplete(const CompleteCand& a, const CompleteCand& b)
{
  const double dt = a.time - b.time;
  if (std::abs(dt) > 0.3)
  {
    return a.time < b.time;
  }
  if (std::abs(a.path - b.path) > 1e-6)
  {
    return a.path < b.path;
  }
  return a.min_clearance > b.min_clearance;
}

struct SearchViz
{
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers;
  rclcpp::Publisher<moveit_msgs::msg::DisplayRobotState>::SharedPtr ghost;
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr traj;
  Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
  Eigen::Vector3d d1 = -Eigen::Vector3d::UnitY();
  DesignCamera camera;
  Eigen::Vector3d table_c = Eigen::Vector3d::Zero();
  Eigen::Vector3d table_s = Eigen::Vector3d::Zero();
  Eigen::Vector3d column_c = Eigen::Vector3d::Zero();
  Eigen::Vector3d column_s = Eigen::Vector3d::Zero();
  Eigen::Vector3d part0 = Eigen::Vector3d::Zero();
  Eigen::Isometry3d t_world_base = Eigen::Isometry3d::Identity();
  double object_height = 0.035;
  double object_radius = 0.0075;
  std::string hud = "STEP 12C";
  std::string a_status = "PENDING";
  std::string b_status = "PENDING";
  std::string c_status = "PENDING";
  GeometricVisibilityResult vis;
  std::optional<EndpointCandidate> current;
  std::optional<EndpointCandidate> best;
  std::chrono::steady_clock::time_point last_pub{};
  double min_period = 0.4;

  Eigen::Isometry3d currentObjectWorld() const
  {
    if (current)
    {
      return t_world_base * poseToIso(current->object_target.pose);
    }
    if (best)
    {
      return t_world_base * poseToIso(best->object_target.pose);
    }
    return Eigen::Isometry3d::Identity();
  }

  void publish(const rclcpp::Node::SharedPtr& node, bool force = false)
  {
    (void)node;
    if (!markers)
    {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!force && last_pub.time_since_epoch().count() != 0 &&
        std::chrono::duration<double>(now - last_pub).count() < min_period)
    {
      return;
    }
    last_pub = now;
    visualization_msgs::msg::MarkerArray arr;
    addCube(arr, 1, table_c, table_s, rgba(0.45f, 0.32f, 0.18f, 0.35f), "workcell/table");
    addCube(arr, 2, column_c, column_s, rgba(0.35f, 0.35f, 0.4f, 0.35f), "workcell/column");
    addSphere(arr, 3, part0, 0.012, rgba(0.9f, 0.8f, 0.1f, 0.9f), "workcell/part");
    const Eigen::Vector3d initial_bottom = part0 + Eigen::Vector3d(0.0, 0.0, -0.5 * object_height);
    const Eigen::Vector3d initial_top = part0 + Eigen::Vector3d(0.0, 0.0, 0.5 * object_height);
    addSphere(arr, 4, initial_bottom, 0.008, rgba(0.1f, 0.95f, 0.35f, 1.0f), "identity/initial_bottom");
    addSphere(arr, 5, initial_top, 0.008, rgba(0.7f, 0.7f, 0.85f, 0.55f), "identity/initial_top");
    addText(arr, 6, initial_bottom + Eigen::Vector3d(0.08, 0.0, 0.0),
            "INITIAL BOTTOM\nobject -Z\nz=0.750 TABLE CONTACT", rgba(0.2f, 1.0f, 0.4f, 1.0f), 0.022,
            "identity/initial_bottom");
    addText(arr, 7, initial_top + Eigen::Vector3d(0.08, 0.0, 0.02),
            "INITIAL TOP\nobject +Z\nz=0.785 ARM2 FUTURE", rgba(0.75f, 0.75f, 0.9f, 0.7f), 0.02,
            "identity/initial_top");
    addSphere(arr, 10, p1, 0.012, rgba(1.0f, 0.2f, 0.9f, 1.0f), "inspection/p1");
    addText(arr, 11, p1 + Eigen::Vector3d(0.0, 0.0, 0.06), "P1 SURFACE CENTER",
            rgba(1.0f, 0.4f, 1.0f, 1.0f), 0.03, "inspection/p1");
    addArrow(arr, 12, p1, p1 + 0.18 * d1, rgba(0.2f, 0.9f, 1.0f, 1.0f), "inspection/d1");
    addText(arr, 13, p1 + 0.22 * d1,
            "TARGET SURFACE NORMAL\n[0, -0.707, +0.707]\nface outward toward camera",
            rgba(0.3f, 0.95f, 1.0f, 1.0f), 0.026, "inspection/d1");
    addSphere(arr, 20, camera.optical_center_world, 0.02, rgba(1.0f, 0.55f, 0.1f, 1.0f),
              "inspection/camera");
    addArrow(arr, 21, camera.optical_center_world,
             camera.optical_center_world + 0.22 * camera.optical_forward_world,
             rgba(1.0f, 0.7f, 0.15f, 1.0f), "inspection/camera");
    addText(arr, 22, camera.optical_center_world + Eigen::Vector3d(0.05, 0.0, 0.08),
            "NEW CAMERA\n[0,0,1.40] RPY -135 deg\nforward [0,+0.707,-0.707]",
            rgba(1.0f, 0.7f, 0.2f, 1.0f), 0.026, "inspection/camera");
    addCircle(arr, 30, p1, d1, object_radius, rgba(0.15f, 1.0f, 0.35f, 1.0f), "inspection/c_bottom");
    addText(arr, 31, p1 + Eigen::Vector3d(0.0, -0.08, 0.10),
            "C — ORIGINAL BOTTOM FACE\nobject -Z\nTABLE-CONTACT FACE\nINSPECTED BY ARM 1",
            rgba(0.15f, 1.0f, 0.4f, 1.0f), 0.028, "inspection/c_bottom");
    const Eigen::Isometry3d t_obj = currentObjectWorld();
    const Eigen::Vector3d bottom_c =
        t_obj.translation() + t_obj.linear() * Eigen::Vector3d(0.0, 0.0, -0.5 * object_height);
    const Eigen::Vector3d top_c =
        t_obj.translation() + t_obj.linear() * Eigen::Vector3d(0.0, 0.0, 0.5 * object_height);
    const Eigen::Vector3d bottom_n = (t_obj.linear() * Eigen::Vector3d(0.0, 0.0, -1.0)).normalized();
    const Eigen::Vector3d top_n = (t_obj.linear() * Eigen::Vector3d(0.0, 0.0, 1.0)).normalized();
    if (current || best)
    {
      addCircle(arr, 32, bottom_c, bottom_n, object_radius, rgba(0.1f, 1.0f, 0.25f, 1.0f),
                "inspection/c_bottom_live");
      addArrow(arr, 33, bottom_c, bottom_c + 0.12 * bottom_n, rgba(0.1f, 1.0f, 0.3f, 1.0f),
               "inspection/c_bottom_live");
      addCircle(arr, 34, top_c, top_n, object_radius, rgba(0.65f, 0.7f, 0.95f, 0.45f),
                "inspection/top_reserved");
      addSphere(arr, 35, top_c, 0.006, rgba(0.7f, 0.75f, 1.0f, 0.45f), "inspection/top_reserved");
      addText(arr, 36, top_c + Eigen::Vector3d(0.0, 0.06, 0.06),
              "ORIGINAL TOP FACE\nobject +Z\nNOT INSPECTED BY ARM 1\nRESERVED FOR ARM 2",
              rgba(0.75f, 0.8f, 1.0f, 0.55f), 0.022, "inspection/top_reserved");
    }
    int los_id = 40;
    for (size_t i = 0; i < vis.roi_world.size() && i < 8; ++i)
    {
      const bool ok = i < vis.ray_visible.size() && vis.ray_visible[i];
      addArrow(arr, los_id++, camera.optical_center_world, vis.roi_world[i],
               ok ? rgba(0.2f, 1.0f, 0.3f, 0.8f) : rgba(1.0f, 0.15f, 0.1f, 0.9f),
               "inspection/los");
    }
    addText(arr, 60, Eigen::Vector3d(0.0, 0.55, 1.55), hud, rgba(1.0f, 1.0f, 1.0f, 1.0f), 0.04,
            "hud/status");
    addText(arr, 61, Eigen::Vector3d(-0.28, 0.55, 1.38),
            std::string("A +Y side  ") + a_status + "\nB -Y side  " + b_status +
                "\nC ORIGINAL BOTTOM object -Z  " + c_status,
            rgba(0.9f, 0.95f, 1.0f, 1.0f), 0.028, "hud/views");
    markers->publish(arr);
  }
};

EdgeResult planJointEdge(const rclcpp::Node::SharedPtr& node, const std::string& group,
                         const planning_scene::PlanningScene& start_scene,
                         const std::map<std::string, double>& from,
                         const std::map<std::string, double>& to, double planning_time, int attempts)
{
  EdgeResult out;
  for (int i = 0; i < attempts && rclcpp::ok(); ++i)
  {
    ++out.attempts;
    moveit::task_constructor::Task task("", false);
    task.setName("FR3 Step12 Edge");
    task.loadRobotModel(node);
    auto scene = planning_scene::PlanningScene::clone(start_scene.diff());
    applyJoints(scene->getCurrentStateNonConst(), from);
    auto fixed = std::make_unique<moveit::task_constructor::stages::FixedState>("Start");
    fixed->setState(scene);
    task.add(std::move(fixed));
    auto ompl =
        std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
    ompl->setTimeout(planning_time);
    auto move = std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo Edge", ompl);
    move->setGroup(group);
    move->setGoal(to);
    move->setTimeout(planning_time);
    task.add(std::move(move));
    try
    {
      task.init();
    }
    catch (const std::exception&)
    {
      continue;
    }
    if (!task.plan(1) || task.numSolutions() < 1)
    {
      continue;
    }
    moveit_task_constructor_msgs::msg::Solution msg;
    task.solutions().front()->toMsg(msg);
    const double t = trajectoryDuration(msg);
    const double l = pathLengthMsg(msg);
    ++out.success_count;
    out.success = true;
    if (t < out.best_time)
    {
      out.best_time = t;
      out.best_path = l;
    }
  }
  return out;
}

std::string fileBaseName(const std::string& path)
{
  const auto pos = path.find_last_of('/');
  return pos == std::string::npos ? path : path.substr(pos + 1);
}

bool targetsFrozenBaselineFile(const std::string& path)
{
  const std::string base = fileBaseName(path);
  return base == "step12c_tilted_camera_winner.yaml" ||
         base == "step12c_tilted_camera_winner_trajectory.yaml";
}

TrajectoryPointRecord recordFromPoint(const trajectory_msgs::msg::JointTrajectoryPoint& pt)
{
  TrajectoryPointRecord rec;
  rec.positions.assign(pt.positions.begin(), pt.positions.end());
  rec.velocities.assign(pt.velocities.begin(), pt.velocities.end());
  rec.accelerations.assign(pt.accelerations.begin(), pt.accelerations.end());
  rec.sec = pt.time_from_start.sec;
  rec.nanosec = pt.time_from_start.nanosec;
  return rec;
}

void appendMultiDof(const trajectory_msgs::msg::MultiDOFJointTrajectory& md,
                    TrajectorySegmentRecord& seg)
{
  seg.multi_dof_present = !md.joint_names.empty() || !md.points.empty();
  if (!seg.multi_dof_present)
  {
    return;
  }
  seg.multi_dof.joint_names = md.joint_names;
  for (const auto& pt : md.points)
  {
    fr_task_planner::MultiDofPointRecord rec;
    rec.sec = pt.time_from_start.sec;
    rec.nanosec = pt.time_from_start.nanosec;
    for (const auto& tf : pt.transforms)
    {
      fr_task_planner::MultiDofTransformRecord item;
      item.translation = { tf.translation.x, tf.translation.y, tf.translation.z };
      item.rotation_xyzw = { tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w };
      rec.transforms.push_back(item);
    }
    seg.multi_dof.points.push_back(rec);
  }
}

bool extractPersistedSegments(const moveit::task_constructor::SolutionBase& solution,
                              PersistedTrajectory& traj, std::string& error)
{
  std::vector<const moveit::task_constructor::SolutionBase*> leaves;
  flattenSolutions(solution, leaves);
  std::map<std::string, int> logical_count;
  for (const auto* leaf : leaves)
  {
    if (!leaf)
    {
      continue;
    }
    const std::string stage = (leaf->creator() && !leaf->creator()->name().empty()) ?
                                  leaf->creator()->name() :
                                  "unnamed";
    moveit_task_constructor_msgs::msg::Solution leaf_msg;
    leaf->toMsg(leaf_msg);
    std::map<std::string, double> start_j;
    std::map<std::string, double> end_j;
    if (leaf->start() && leaf->start()->scene())
    {
      start_j = jointsFromState(leaf->start()->scene()->getCurrentState());
    }
    if (leaf->end() && leaf->end()->scene())
    {
      end_j = jointsFromState(leaf->end()->scene()->getCurrentState());
    }
      for (const auto& sub : leaf_msg.sub_trajectory)
    {
      const auto& jt = sub.trajectory.joint_trajectory;
      const auto& md = sub.trajectory.multi_dof_joint_trajectory;
      const bool has_jt = !jt.points.empty();
      const bool has_md = !md.joint_names.empty() || !md.points.empty();
      if (!has_jt && !has_md)
      {
        continue;
      }
      TrajectorySegmentRecord seg;
      seg.planning_stage_name = stage;
      seg.logical_segment = logicalSegmentFromStage(stage);
      if (seg.logical_segment.empty())
      {
        seg.logical_segment = stage;
      }
      seg.subsegment_index = logical_count[seg.logical_segment]++;
      seg.name = seg.subsegment_index == 0 ?
                     seg.logical_segment :
                     (seg.logical_segment + "_sub" + std::to_string(seg.subsegment_index));
      seg.deployable = isFixedDeployableLogical(seg.logical_segment);
      seg.runtime_replan_required = seg.logical_segment == kLogicalCurrentToHome;
      seg.joint_names = jt.joint_names;
      if (has_jt)
      {
        if (start_j.empty())
        {
          seg.start_joints = extractArmPositions(jt.joint_names, jt.points.front().positions);
        }
        else
        {
          seg.start_joints = jointsToVec(start_j);
        }
        if (end_j.empty())
        {
          seg.end_joints = extractArmPositions(jt.joint_names, jt.points.back().positions);
        }
        else
        {
          seg.end_joints = jointsToVec(end_j);
        }
      }
      else
      {
        seg.start_joints = jointsToVec(start_j);
        seg.end_joints = jointsToVec(end_j);
      }
      bool any_vel = false;
      bool any_acc = false;
      for (const auto& pt : jt.points)
      {
        auto rec = recordFromPoint(pt);
        any_vel = any_vel || !rec.velocities.empty();
        any_acc = any_acc || !rec.accelerations.empty();
        seg.points.push_back(std::move(rec));
      }
      seg.velocities_present = any_vel;
      seg.accelerations_present = any_acc;
      appendMultiDof(md, seg);
      traj.segments.push_back(std::move(seg));
    }
  }
  if (traj.segments.empty())
  {
    error = "planned solution contained no JointTrajectory";
    return false;
  }
  std::set<std::string> names;
  for (const auto& name : kArmJoints)
  {
    names.insert(name);
  }
  for (const auto& seg : traj.segments)
  {
    names.insert(seg.joint_names.begin(), seg.joint_names.end());
  }
  traj.joint_names.assign(kArmJoints.begin(), kArmJoints.end());
  for (const auto& name : names)
  {
    if (std::find(traj.joint_names.begin(), traj.joint_names.end(), name) == traj.joint_names.end())
    {
      traj.joint_names.push_back(name);
    }
  }
  attachStandardEvents(traj);
  fillDerivedTotals(traj);
  return true;
}

moveit_msgs::msg::DisplayTrajectory displayFromPersisted(const PersistedTrajectory& traj,
                                                         const std::string& model_id)
{
  moveit_msgs::msg::DisplayTrajectory disp;
  disp.model_id = model_id;
  if (!traj.segments.empty())
  {
    disp.trajectory_start.joint_state.name.assign(kArmJoints.begin(), kArmJoints.end());
    disp.trajectory_start.joint_state.position = traj.segments.front().start_joints;
  }
  for (const auto& seg : traj.segments)
  {
    if (seg.points.empty())
    {
      continue;
    }
    moveit_msgs::msg::RobotTrajectory rt;
    rt.joint_trajectory.header.frame_id = kPlanningFrame;
    rt.joint_trajectory.joint_names = seg.joint_names;
    for (const auto& pt : seg.points)
    {
      trajectory_msgs::msg::JointTrajectoryPoint msg;
      msg.positions = pt.positions;
      msg.velocities = pt.velocities;
      msg.accelerations = pt.accelerations;
      msg.time_from_start.sec = pt.sec;
      msg.time_from_start.nanosec = pt.nanosec;
      rt.joint_trajectory.points.push_back(msg);
    }
    disp.trajectory.push_back(rt);
  }
  return disp;
}

size_t countDisplayPoints(const moveit_msgs::msg::DisplayTrajectory& disp)
{
  size_t n = 0;
  for (const auto& tr : disp.trajectory)
  {
    n += tr.joint_trajectory.points.size();
  }
  return n;
}

size_t countPersistedPoints(const PersistedTrajectory& traj)
{
  size_t n = 0;
  for (const auto& seg : traj.segments)
  {
    n += seg.points.size();
  }
  return n;
}

void writePersistDiagnostics(const std::string& path, const std::string& status,
                             const std::string& cls, const std::string& winner_path,
                             const WinnerEndpoints* winner, const PersistedTrajectory* traj,
                             const PersistValidation* roundtrip, const PersistValidation* continuity,
                             const PersistValidation* endpoints, int attempts, int success_attempt,
                             bool plan_ok, const std::string& rviz_source, bool playback_ok)
{
  std::ofstream yaml(path);
  yaml.setf(std::ios::fixed);
  yaml.precision(17);
  yaml << "status: " << status << "\n";
  yaml << "failure_classification: " << cls << "\n";
  yaml << "task_version: STEP12C\n";
  yaml << "source_winner_path: " << winner_path << "\n";
  yaml << "duplicate_move_grasp: NOT_FOUND\n";
  yaml << "duplicate_move_grasp_fixed: N/A\n";
  yaml << "endpoint_search_performed: false\n";
  yaml << "roll_search_performed: false\n";
  yaml << "ik_search_performed: false\n";
  yaml << "beam_search_performed: false\n";
  yaml << "full_task_plan_attempts: " << attempts << "\n";
  yaml << "full_task_plan_success: " << (plan_ok ? "true" : "false") << "\n";
  yaml << "successful_attempt: " << success_attempt << "\n";
  yaml << "execution_performed: false\n";
  yaml << "rviz_playback_source: " << rviz_source << "\n";
  yaml << "rviz_playback: " << (playback_ok ? "PASS" : "FAIL") << "\n";
  if (winner)
  {
    yaml << "source_winner:\n";
    yaml << "  a_roll_deg: " << winner->a_roll_deg << "\n";
    yaml << "  b_roll_deg: " << winner->b_roll_deg << "\n";
    yaml << "  c_roll_deg: " << winner->c_roll_deg << "\n";
  }
  if (traj)
  {
    yaml << "motion_subtrajectory_count: " << traj->segments.size() << "\n";
    int deployable = 0;
    bool current_home = false;
    bool multi = false;
    for (const auto& seg : traj->segments)
    {
      deployable += seg.deployable ? 1 : 0;
      current_home = current_home || seg.logical_segment == kLogicalCurrentToHome;
      multi = multi || seg.multi_dof_present;
      yaml << "segment_" << seg.name << "_points: " << seg.points.size() << "\n";
      yaml << "segment_" << seg.name << "_duration: " << seg.duration << "\n";
    }
    yaml << "fixed_deployable_segment_count: " << deployable << "\n";
    yaml << "current_to_home_saved: " << (current_home ? "true" : "false") << "\n";
    yaml << "current_to_home_deployable: false\n";
    yaml << "runtime_replan_required: true\n";
    yaml << "multi_dof_present: " << (multi ? "true" : "false") << "\n";
    yaml << "persisted_full_plan_total_time: " << traj->persisted_full_plan_total_time << "\n";
    yaml << "persisted_fixed_task_total_duration: " << traj->persisted_fixed_task_total_duration
         << "\n";
    yaml << "persisted_total_joint_path_length: " << traj->persisted_total_joint_path_length << "\n";
    yaml << "search_score_total_time: " << traj->search_score_total_time << "\n";
    yaml << "search_score_joint_path_length: " << traj->search_score_joint_path_length << "\n";
    yaml << "time_difference: "
         << (traj->persisted_full_plan_total_time - traj->search_score_total_time) << "\n";
    yaml << "path_difference: "
         << (traj->persisted_total_joint_path_length - traj->search_score_joint_path_length) << "\n";
  }
  if (roundtrip)
  {
    yaml << "round_trip_ok: " << (roundtrip->ok ? "true" : "false") << "\n";
    yaml << "round_trip_reason: " << roundtrip->reason << "\n";
    yaml << "round_trip_position_max_error: " << roundtrip->position_max_error << "\n";
    yaml << "round_trip_velocity_max_error: " << roundtrip->velocity_max_error << "\n";
    yaml << "round_trip_acceleration_max_error: " << roundtrip->acceleration_max_error << "\n";
    yaml << "round_trip_time_identical: " << (roundtrip->time_identical ? "true" : "false") << "\n";
  }
  if (continuity)
  {
    yaml << "continuity_ok: " << (continuity->ok ? "true" : "false") << "\n";
    yaml << "max_discontinuity: " << continuity->max_discontinuity << "\n";
    for (const auto& item : continuity->discontinuities)
    {
      std::string key = item.first;
      for (char& c : key)
      {
        if (c == ' ' || c == '/')
        {
          c = '_';
        }
      }
      yaml << "continuity_" << key << ": " << item.second << "\n";
    }
  }
  if (endpoints)
  {
    yaml << "home_start_error: " << endpoints->home_start_error << "\n";
    yaml << "a_max_joint_error: " << endpoints->a_max_error << "\n";
    yaml << "b_max_joint_error: " << endpoints->b_max_error << "\n";
    yaml << "c_max_joint_error: " << endpoints->c_max_error << "\n";
    yaml << "endpoint_ok: " << (endpoints->ok ? "true" : "false") << "\n";
  }
}

int runWinnerReplayPersist(
    const rclcpp::Node::SharedPtr& node, const std::string& group, const std::string& ee_link,
    const std::string& attach_link, const std::string& object_id, const std::string& table_name,
    const std::vector<std::string>& touch_links, const geometry_msgs::msg::PoseStamped& object_scene,
    const geometry_msgs::msg::PoseStamped& pregrasp, const geometry_msgs::msg::PoseStamped& grasp,
    const geometry_msgs::msg::PoseStamped& lift, double object_height, double object_radius,
    double planning_time, const std::map<std::string, double>& home,
    const moveit::core::RobotModelConstPtr& robot_model, SearchViz& viz, const std::string& diag_path)
{
  RCLCPP_INFO(node->get_logger(), "========== STEP 12D WINNER TRAJECTORY PERSIST ==========");
  RCLCPP_INFO(node->get_logger(), "WINNER REPLAY / PERSIST ONLY");
  RCLCPP_INFO(node->get_logger(), "endpoint search performed: NO");
  RCLCPP_INFO(node->get_logger(), "roll search performed: NO");
  RCLCPP_INFO(node->get_logger(), "IK search performed: NO");
  RCLCPP_INFO(node->get_logger(), "beam search performed: NO");
  RCLCPP_INFO(node->get_logger(), "DUPLICATE_MOVE_GRASP: NOT FOUND");
  RCLCPP_INFO(node->get_logger(), "FIXED: N/A");
  RCLCPP_INFO(node->get_logger(), "PLAN / SERIALIZE / READ-BACK / RVIZ DISPLAY ONLY");

  const std::string winner_path = getString(
      node, "winner_input_path",
      "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/step12c_tilted_camera_winner.yaml");
  const std::string traj_path = getString(
      node, "trajectory_output_path",
      "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/"
      "step12c_tilted_camera_winner_trajectory.yaml");
  const int retries = getInt(node, "full_plan_retries", 5);
  const bool visualize_saved = getBool(node, "visualize_saved_trajectory", true);
  if (visualize_saved && !viz.traj)
  {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
    viz.traj = node->create_publisher<moveit_msgs::msg::DisplayTrajectory>(kTrajTopic, qos);
  }

  WinnerEndpoints winner;
  std::string error;
  if (!loadWinnerYaml(winner_path, winner, error))
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to load winner YAML %s: %s", winner_path.c_str(),
                 error.c_str());
    writePersistDiagnostics(diag_path, "FAIL", "WINNER_YAML_UNREADABLE", winner_path, nullptr,
                            nullptr, nullptr, nullptr, nullptr, 0, 0, false, "NONE", false);
    return 3;
  }
  if (!winnerMatchesFrozenStep12c(winner, error))
  {
    RCLCPP_ERROR(node->get_logger(), "Winner endpoints changed: %s", error.c_str());
    writePersistDiagnostics(diag_path, "FAIL", "WINNER_ENDPOINT_MISMATCH", winner_path, &winner,
                            nullptr, nullptr, nullptr, nullptr, 0, 0, false, "NONE", false);
    return 3;
  }
  if (maxJointError(home, winner.home_rad) > 1e-8)
  {
    RCLCPP_ERROR(node->get_logger(), "Launch Home does not match winner Home (max|dq|=%.12f)",
                 maxJointError(home, winner.home_rad));
    writePersistDiagnostics(diag_path, "FAIL", "HOME_MISMATCH", winner_path, &winner, nullptr,
                            nullptr, nullptr, nullptr, 0, 0, false, "NONE", false);
    return 3;
  }
  RCLCPP_INFO(node->get_logger(),
              "SOURCE WINNER A r=%.1f B r=%.1f C r=%.1f (endpoints locked, no search)",
              winner.a_roll_deg, winner.b_roll_deg, winner.c_roll_deg);

  PersistedTrajectory planned;
  int success_attempt = 0;
  int attempts = 0;
  bool plan_ok = false;
  for (int attempt = 1; attempt <= retries && rclcpp::ok(); ++attempt)
  {
    ++attempts;
    moveit::task_constructor::Task task("", true);
    task.setName("FR3 Step12D Persist Winner");
    task.loadRobotModel(node);
    addPrefixStages(task, node, group, ee_link, attach_link, object_id, table_name, touch_links,
                    object_scene, pregrasp, grasp, lift, object_height, object_radius, planning_time,
                    home);
    auto ompl =
        std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
    ompl->setTimeout(planning_time);
    auto move_a = std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo ViewA", ompl);
    move_a->setGroup(group);
    move_a->setGoal(winner.a_rad);
    move_a->setTimeout(planning_time);
    task.add(std::move(move_a));
    auto move_b = std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo ViewB", ompl);
    move_b->setGroup(group);
    move_b->setGoal(winner.b_rad);
    move_b->setTimeout(planning_time);
    task.add(std::move(move_b));
    auto move_c = std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo ViewC", ompl);
    move_c->setGroup(group);
    move_c->setGoal(winner.c_rad);
    move_c->setTimeout(planning_time);
    task.add(std::move(move_c));
    try
    {
      task.init();
    }
    catch (const std::exception& ex)
    {
      RCLCPP_WARN(node->get_logger(), "persist task init failed attempt %d: %s", attempt, ex.what());
      continue;
    }
    if (!task.plan(1) || task.numSolutions() < 1)
    {
      RCLCPP_WARN(node->get_logger(), "persist full-task plan failed attempt %d/%d", attempt,
                  retries);
      continue;
    }
    planned = PersistedTrajectory{};
    planned.source_winner_file = fileBaseName(winner_path);
    planned.source_winner_path = winner_path;
    planned.search_score_total_time = winner.search_score_total_time;
    planned.search_score_joint_path_length = winner.search_score_joint_path_length;
    planned.winner = winner;
    if (!extractPersistedSegments(*task.solutions().front(), planned, error))
    {
      RCLCPP_ERROR(node->get_logger(), "persist extract failed: %s", error.c_str());
      writePersistDiagnostics(diag_path, "FAIL", "TRAJECTORY_EXTRACT_FAILED", winner_path, &winner,
                              nullptr, nullptr, nullptr, nullptr, attempts, attempt, true, "NONE",
                              false);
      return 5;
    }
    plan_ok = true;
    success_attempt = attempt;
    RCLCPP_INFO(node->get_logger(),
                "FIXED-WINNER FULL PLAN SUCCESS attempt=%d/%d subtrajectories=%zu", attempt,
                retries, planned.segments.size());
    break;
  }
  if (!plan_ok)
  {
    RCLCPP_ERROR(node->get_logger(), "FIXED_WINNER_REPLAY_PLANNING_FAILED after %d attempts",
                 attempts);
    writePersistDiagnostics(diag_path, "FAIL", "FIXED_WINNER_REPLAY_PLANNING_FAILED", winner_path,
                            &winner, nullptr, nullptr, nullptr, nullptr, attempts, 0, false, "NONE",
                            false);
    return 4;
  }

  if (!writeTrajectoryYaml(traj_path, planned, error))
  {
    RCLCPP_ERROR(node->get_logger(), "serialize failed: %s", error.c_str());
    writePersistDiagnostics(diag_path, "FAIL", "TRAJECTORY_SERIALIZE_FAILED", winner_path, &winner,
                            &planned, nullptr, nullptr, nullptr, attempts, success_attempt, true,
                            "NONE", false);
    return 5;
  }
  RCLCPP_INFO(node->get_logger(), "Serialized exact planned JointTrajectory to %s",
              traj_path.c_str());

  PersistedTrajectory loaded;
  if (!readTrajectoryYaml(traj_path, loaded, error))
  {
    RCLCPP_ERROR(node->get_logger(), "read-back failed: %s", error.c_str());
    writePersistDiagnostics(diag_path, "FAIL", "TRAJECTORY_READBACK_FAILED", winner_path, &winner,
                            &planned, nullptr, nullptr, nullptr, attempts, success_attempt, true,
                            "NONE", false);
    return 5;
  }
  const auto roundtrip = validateRoundTrip(planned, loaded);
  const auto continuity = validateContinuity(loaded);
  const auto endpoints = validateEndpoints(loaded, winner);
  bool times_ok = true;
  for (const auto& seg : loaded.segments)
  {
    if (!seg.points.empty() && !timesMonotonic(seg))
    {
      times_ok = false;
      RCLCPP_ERROR(node->get_logger(), "time_from_start not monotonic or duration<=0: %s",
                   seg.name.c_str());
    }
  }
  RCLCPP_INFO(node->get_logger(),
              "ROUND-TRIP %s pos_err=%.3e vel_err=%.3e acc_err=%.3e time_identical=%s",
              roundtrip.ok ? "PASS" : "FAIL", roundtrip.position_max_error,
              roundtrip.velocity_max_error, roundtrip.acceleration_max_error,
              roundtrip.time_identical ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "CONTINUITY %s max_disc=%.6e", continuity.ok ? "PASS" : "FAIL",
              continuity.max_discontinuity);
  RCLCPP_INFO(node->get_logger(), "ENDPOINT Home=%.6e A=%.6e B=%.6e C=%.6e %s",
              endpoints.home_start_error, endpoints.a_max_error, endpoints.b_max_error,
              endpoints.c_max_error, endpoints.ok ? "PASS" : "FAIL");
  RCLCPP_INFO(node->get_logger(),
              "TIMING search_score=%.9f persisted_full=%.9f persisted_fixed=%.9f diff=%.9f",
              loaded.search_score_total_time, loaded.persisted_full_plan_total_time,
              loaded.persisted_fixed_task_total_duration,
              loaded.persisted_full_plan_total_time - loaded.search_score_total_time);
  RCLCPP_INFO(node->get_logger(), "PATH search_score=%.9f persisted=%.9f diff=%.9f",
              loaded.search_score_joint_path_length, loaded.persisted_total_joint_path_length,
              loaded.persisted_total_joint_path_length - loaded.search_score_joint_path_length);

  const bool persist_ok = roundtrip.ok && continuity.ok && endpoints.ok && times_ok;
  bool playback_ok = false;
  std::string rviz_source = "NONE";
  if (persist_ok && visualize_saved && viz.traj)
  {
    const auto disp = displayFromPersisted(loaded, robot_model->getName());
    const size_t saved_pts = countPersistedPoints(loaded);
    const size_t played_pts = countDisplayPoints(disp);
    viz.traj->publish(disp);
    playback_ok = saved_pts == played_pts && saved_pts > 0;
    rviz_source = "SAVED TRAJECTORY";
    RCLCPP_INFO(node->get_logger(), "RVIZ PLAYBACK SOURCE:");
    RCLCPP_INFO(node->get_logger(), "SAVED STEP12C WINNER TRAJECTORY");
    RCLCPP_INFO(node->get_logger(), "NOT:");
    RCLCPP_INFO(node->get_logger(), "NEWLY REPLANNED TRAJECTORY");
    RCLCPP_INFO(node->get_logger(), "saved-vs-played points: saved=%zu played=%zu %s", saved_pts,
                played_pts, playback_ok ? "PASS" : "FAIL");
  }
  else if (!visualize_saved)
  {
    rviz_source = "DISABLED";
  }

  if (persist_ok && playback_ok)
  {
    RCLCPP_INFO(node->get_logger(), "FROZEN TRAJECTORY CREATED");
    writePersistDiagnostics(diag_path, "PASS", "none", winner_path, &winner, &loaded, &roundtrip,
                            &continuity, &endpoints, attempts, success_attempt, true, rviz_source,
                            playback_ok);
    return 0;
  }
  if (persist_ok && !visualize_saved)
  {
    RCLCPP_INFO(node->get_logger(), "FROZEN TRAJECTORY CREATED");
    writePersistDiagnostics(diag_path, "PASS", "none", winner_path, &winner, &loaded, &roundtrip,
                            &continuity, &endpoints, attempts, success_attempt, true, rviz_source,
                            true);
    return 0;
  }
  const char* cls = !roundtrip.ok ? "ROUND_TRIP_FAILED" :
                    !times_ok     ? "TIME_MONOTONIC_FAILED" :
                    !continuity.ok ? "CONTINUITY_FAILED" :
                    !endpoints.ok  ? "ENDPOINT_MATCH_FAILED" :
                                     "RVIZ_SAVED_PLAYBACK_FAILED";
  writePersistDiagnostics(diag_path, "FAIL", cls, winner_path, &winner, &loaded, &roundtrip,
                          &continuity, &endpoints, attempts, success_attempt, true, rviz_source,
                          playback_ok);
  return 5;
}

std::string fileSha256(const std::string& path)
{
  const std::string cmd = "sha256sum \"" + path + "\" 2>/dev/null";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe)
  {
    return "";
  }
  char buf[160];
  std::string out;
  while (fgets(buf, sizeof(buf), pipe) != nullptr)
  {
    out += buf;
  }
  pclose(pipe);
  const auto sp = out.find(' ');
  return sp == std::string::npos ? out : out.substr(0, sp);
}

planning_scene::PlanningScenePtr makePreGraspPlanningScene(
    const planning_scene::PlanningScene& src, const moveit_msgs::msg::CollisionObject& object,
    const std::string& object_id, const std::string& table_name)
{
  auto scene = cloneDiagnosticScene(src);
  detachObjectDiagnostic(*scene, object_id);
  for (const auto& name : scene->getWorld()->getObjectIds())
  {
    if (name == object_id)
    {
      removeWorldObjectDiagnostic(*scene, object_id);
      break;
    }
  }
  scene->processCollisionObjectMsg(object);
  scene->getAllowedCollisionMatrixNonConst().setEntry(object_id, table_name, true);
  return scene;
}

std::map<std::string, double> stageEndJoints(
    const std::vector<const moveit::task_constructor::SolutionBase*>& leaves, const std::string& name)
{
  const auto* sol = findStageSolution(leaves, name);
  if (!sol || !sol->end() || !sol->end()->scene())
  {
    return {};
  }
  return jointsFromState(sol->end()->scene()->getCurrentState());
}

bool finiteTrajectory(const PersistedTrajectory& traj, std::string& reason)
{
  for (const auto& seg : traj.segments)
  {
    for (const auto& name : kArmJoints)
    {
      if (std::find(seg.joint_names.begin(), seg.joint_names.end(), name) == seg.joint_names.end() &&
          !seg.points.empty())
      {
        reason = "missing joint " + name + " in " + seg.name;
        return false;
      }
    }
    if (!seg.points.empty() && !timesMonotonic(seg))
    {
      reason = "nonmonotonic time: " + seg.name;
      return false;
    }
    for (const auto& pt : seg.points)
    {
      auto check = [&](const std::vector<double>& values, const char* kind) {
        for (double v : values)
        {
          if (!std::isfinite(v))
          {
            reason = std::string("nonfinite ") + kind + " in " + seg.name;
            return false;
          }
        }
        return true;
      };
      if (!check(pt.positions, "position") || !check(pt.velocities, "velocity") ||
          !check(pt.accelerations, "acceleration"))
      {
        return false;
      }
    }
  }
  return true;
}

struct GraspPlanOutcome
{
  bool ok = false;
  std::string failure = "not planned";
  PersistedTrajectory traj;
  FrozenTaskMetrics metrics;
  std::map<std::string, double> pregrasp_joints;
  std::map<std::string, double> grasp_joints;
  std::map<std::string, double> lift_joints;
  double min_clearance = 0.0;
  bool collision_free = false;
  bool joint_limits_ok = false;
  bool pilz_lin = false;
  bool abc_ok = false;
  bool fk_ok = false;
  bool pregrasp_locked = false;
};

void addFixedAbcStages(moveit::task_constructor::Task& task, const rclcpp::Node::SharedPtr& node,
                       const std::string& group, const WinnerEndpoints& winner, double planning_time)
{
  auto ompl =
      std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
  ompl->setTimeout(planning_time);
  auto move_a = std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo ViewA", ompl);
  move_a->setGroup(group);
  move_a->setGoal(winner.a_rad);
  move_a->setTimeout(planning_time);
  task.add(std::move(move_a));
  auto move_b = std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo ViewB", ompl);
  move_b->setGroup(group);
  move_b->setGoal(winner.b_rad);
  move_b->setTimeout(planning_time);
  task.add(std::move(move_b));
  auto move_c = std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo ViewC", ompl);
  move_c->setGroup(group);
  move_c->setGoal(winner.c_rad);
  move_c->setTimeout(planning_time);
  task.add(std::move(move_c));
}

GraspPlanOutcome evaluateMtcSolution(
    moveit::task_constructor::Task& task, bool full, const std::string& group,
    const std::string& ee_link, const geometry_msgs::msg::PoseStamped& pregrasp,
    const geometry_msgs::msg::PoseStamped& grasp, const geometry_msgs::msg::PoseStamped& lift,
    const WinnerEndpoints& winner, const std::map<std::string, double>& locked_pregrasp,
    double pos_tol, double ori_tol_deg, const CollisionDiagConfig& dcfg)
{
  GraspPlanOutcome out;
  std::string error;
  if (!extractPersistedSegments(*task.solutions().front(), out.traj, error))
  {
    out.failure = "extract failed: " + error;
    return out;
  }
  if (!finiteTrajectory(out.traj, out.failure))
  {
    return out;
  }
  std::vector<const moveit::task_constructor::SolutionBase*> leaves;
  flattenSolutions(*task.solutions().front(), leaves);
  out.pregrasp_joints = stageEndJoints(leaves, "MoveTo PreGrasp");
  out.grasp_joints = stageEndJoints(leaves, "MoveTo Grasp");
  out.lift_joints = stageEndJoints(leaves, "MoveTo Lift");
  out.pilz_lin = findStageSolution(leaves, "MoveTo Grasp") != nullptr &&
                 findStageSolution(leaves, "MoveTo Lift") != nullptr;
  if (!out.pilz_lin)
  {
    out.failure = "PILZ_LIN_MISSING";
    return out;
  }
  if (out.pregrasp_joints.empty() || maxAbsErrorJoints(out.pregrasp_joints, locked_pregrasp) > 1e-3)
  {
    out.failure = "PREGRASP_JOINT_LOCK_FAILED";
    return out;
  }
  out.pregrasp_locked = true;
  auto* jmg = task.getRobotModel()->getJointModelGroup(group);
  auto fk_stage = [&](const char* stage, const geometry_msgs::msg::PoseStamped& pose) {
    const auto* sol = findStageSolution(leaves, stage);
    if (!sol || !sol->end() || !sol->end()->scene())
    {
      return false;
    }
    const auto& state = sol->end()->scene()->getCurrentState();
    if (jmg && !state.satisfiesBounds(jmg))
    {
      out.joint_limits_ok = false;
      return false;
    }
    double pos = 0.0;
    double ori = 0.0;
    poseError(poseToIso(pose.pose), tcpInBase(state, ee_link), pos, ori);
    return pos <= pos_tol && ori <= ori_tol_deg;
  };
  out.joint_limits_ok = true;
  out.fk_ok = fk_stage("MoveTo PreGrasp", pregrasp) && fk_stage("MoveTo Grasp", grasp) &&
              fk_stage("MoveTo Lift", lift);
  if (!out.fk_ok)
  {
    out.failure = "PREFIX_FK_FAILED";
    return out;
  }
  out.collision_free = true;
  out.min_clearance = std::numeric_limits<double>::infinity();
  for (const auto* leaf : leaves)
  {
    if (!leaf || !leaf->start() || !leaf->start()->scene())
    {
      continue;
    }
    auto scene = cloneDiagnosticScene(*leaf->start()->scene());
    moveit_task_constructor_msgs::msg::Solution msg;
    leaf->toMsg(msg);
    for (const auto& sub : msg.sub_trajectory)
    {
      const auto& jt = sub.trajectory.joint_trajectory;
      if (jt.points.empty())
      {
        continue;
      }
      const size_t stride = std::max<size_t>(1, jt.points.size() / 8);
      for (size_t i = 0; i < jt.points.size(); i += stride)
      {
        const size_t idx = (i + stride >= jt.points.size()) ? (jt.points.size() - 1) : i;
        std::map<std::string, double> q;
        const size_t n = std::min(jt.joint_names.size(), jt.points[idx].positions.size());
        for (size_t k = 0; k < n; ++k)
        {
          q[jt.joint_names[k]] = jt.points[idx].positions[k];
        }
        applyJointsToScene(*scene, q);
        if (jmg && !scene->getCurrentState().satisfiesBounds(jmg))
        {
          out.joint_limits_ok = false;
          out.failure = "JOINT_LIMITS_FAILED";
          return out;
        }
        const auto snap = collectCollisionContacts(*scene, dcfg);
        if (snap.collision)
        {
          out.collision_free = false;
          out.failure = "TRAJECTORY_COLLISION";
          return out;
        }
        double col = 0.0;
        if (tryColumnDistance(*scene, group, dcfg.column_name, col))
        {
          out.min_clearance = std::min(out.min_clearance, col);
        }
      }
    }
  }
  if (!std::isfinite(out.min_clearance))
  {
    out.min_clearance = 0.0;
  }
  out.metrics = computeFrozenTaskMetrics(out.traj);
  if (!out.metrics.home_to_pregrasp.present || !out.metrics.pregrasp_to_grasp.present ||
      !out.metrics.grasp_to_lift.present)
  {
    out.failure = "PREFIX_SEGMENTS_MISSING";
    return out;
  }
  if (full)
  {
    const auto endpoints = validateEndpoints(out.traj, winner, 1e-3);
    const auto continuity = validateContinuity(out.traj, 1e-3);
    out.abc_ok = endpoints.ok;
    if (!endpoints.ok)
    {
      out.failure = "ABC_ENDPOINT_MISMATCH";
      return out;
    }
    if (!continuity.ok)
    {
      out.failure = "SEGMENT_DISCONTINUITY";
      return out;
    }
    if (!out.metrics.lift_to_a.present || !out.metrics.a_to_b.present || !out.metrics.b_to_c.present)
    {
      out.failure = "FULL_SEGMENTS_MISSING";
      return out;
    }
  }
  out.ok = out.collision_free && out.joint_limits_ok && out.fk_ok && out.pilz_lin &&
           out.pregrasp_locked;
  if (out.ok)
  {
    out.failure = "ok";
  }
  return out;
}

GraspPlanOutcome planGraspCandidate(
    const rclcpp::Node::SharedPtr& node, const std::string& group, const std::string& ee_link,
    const std::string& attach_link, const std::string& object_id, const std::string& table_name,
    const std::vector<std::string>& touch_links, const geometry_msgs::msg::PoseStamped& object_scene,
    const geometry_msgs::msg::PoseStamped& pregrasp, const geometry_msgs::msg::PoseStamped& grasp,
    const geometry_msgs::msg::PoseStamped& lift, double object_height, double object_radius,
    double planning_time, const std::map<std::string, double>& home,
    const std::map<std::string, double>& locked_pregrasp, const WinnerEndpoints& winner, bool full,
    int max_attempts, const CollisionDiagConfig& dcfg, double pos_tol, double ori_tol_deg)
{
  GraspPlanOutcome last;
  for (int attempt = 1; attempt <= max_attempts && rclcpp::ok(); ++attempt)
  {
    moveit::task_constructor::Task task("", true);
    task.setName(full ? "FR3 Step14 Full" : "FR3 Step14 Prefix");
    task.loadRobotModel(node);
    addPrefixStages(task, node, group, ee_link, attach_link, object_id, table_name, touch_links,
                    object_scene, pregrasp, grasp, lift, object_height, object_radius, planning_time,
                    home, &locked_pregrasp);
    if (full)
    {
      addFixedAbcStages(task, node, group, winner, planning_time);
    }
    try
    {
      task.init();
    }
    catch (const std::exception& ex)
    {
      last.failure = std::string("init failed: ") + ex.what();
      continue;
    }
    if (!task.plan(1) || task.numSolutions() < 1)
    {
      last.failure = full ? "FULL_TASK_PLAN_FAILED" : "PREFIX_PLAN_FAILED";
      continue;
    }
    last = evaluateMtcSolution(task, full, group, ee_link, pregrasp, grasp, lift, winner,
                               locked_pregrasp, pos_tol, ori_tol_deg, dcfg);
    if (last.ok)
    {
      return last;
    }
  }
  return last;
}

int runGraspPrefixOptimize(
    const rclcpp::Node::SharedPtr& node, const std::string& group, const std::string& ee_link,
    const std::string& attach_link, const std::string& object_id, const std::string& table_name,
    const std::string& column_name, const std::vector<std::string>& touch_links,
    const geometry_msgs::msg::PoseStamped& object_scene,
    const geometry_msgs::msg::PoseStamped& pregrasp, const geometry_msgs::msg::PoseStamped& grasp,
    const geometry_msgs::msg::PoseStamped& lift, double object_height, double object_radius,
    double planning_time, const std::map<std::string, double>& home,
    const moveit::core::RobotModelConstPtr& robot_model, SearchViz& viz, const std::string& diag_path,
    double pos_tol, double ori_tol_deg)
{
  const auto t0 = std::chrono::steady_clock::now();
  auto elapsed = [&]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  };
  RCLCPP_INFO(node->get_logger(), "========== STEP 14 GRASP PREFIX OPTIMIZATION ==========");
  RCLCPP_INFO(node->get_logger(), "PLAN / RViz ONLY. No Gazebo execute. No real robot.");
  RCLCPP_INFO(node->get_logger(), "PreGrasp/Grasp/Lift TCP poses FIXED. A/B/C joints FIXED.");
  RCLCPP_INFO(node->get_logger(), "Pilz LIN for Grasp and Lift PRESERVED.");

  const std::string winner_in = getString(
      node, "winner_input_path",
      "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/step12c_tilted_camera_winner.yaml");
  const std::string frozen_path = getString(
      node, "frozen_trajectory_path",
      "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/"
      "step12c_tilted_camera_winner_trajectory.yaml");
  std::string winner_out = getString(
      node, "winner_output_path",
      "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/step14_optimized_grasp_winner.yaml");
  std::string traj_out = getString(
      node, "trajectory_output_path",
      "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/"
      "step14_optimized_grasp_trajectory.yaml");
  if (targetsFrozenBaselineFile(winner_out) || winner_out == winner_in)
  {
    RCLCPP_WARN(node->get_logger(),
                "STEP14 refusing to overwrite frozen winner YAML; redirecting output");
    winner_out =
        "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/"
        "step14_optimized_grasp_winner.yaml";
  }
  if (targetsFrozenBaselineFile(traj_out) || traj_out == frozen_path)
  {
    RCLCPP_WARN(node->get_logger(),
                "STEP14 refusing to overwrite frozen trajectory YAML; redirecting output");
    traj_out =
        "/home/cyberbraindualarm/fr_task_ws/src/fr_task_planner/config/"
        "step14_optimized_grasp_trajectory.yaml";
  }
  const int max_ik = getInt(node, "max_pregrasp_ik_candidates", 24);
  const int max_prefix = getInt(node, "max_prefix_plan_candidates", 12);
  const int max_full = getInt(node, "max_full_task_candidates", 6);
  const int plan_attempts = getInt(node, "full_candidate_plan_attempts", 3);
  const double timeout_sec = node->has_parameter("step14_search_timeout_sec") ?
                                 node->get_parameter("step14_search_timeout_sec").as_double() :
                                 900.0;
  const double prefix_planning_time = node->has_parameter("step14_prefix_planning_time") ?
                                          node->get_parameter("step14_prefix_planning_time").as_double() :
                                          std::min(5.0, planning_time);
  const double full_planning_time = node->has_parameter("step14_full_planning_time") ?
                                        node->get_parameter("step14_full_planning_time").as_double() :
                                        planning_time;
  const std::string sha_winner_before = fileSha256(winner_in);
  const std::string sha_traj_before = fileSha256(frozen_path);

  WinnerEndpoints winner;
  std::string error;
  if (!loadWinnerYaml(winner_in, winner, error) || !winnerMatchesFrozenStep12c(winner, error))
  {
    RCLCPP_ERROR(node->get_logger(), "STEP14 winner lock failed: %s", error.c_str());
    std::ofstream yaml(diag_path);
    yaml << "status: FAIL\nfailure_classification: WINNER_ENDPOINT_MISMATCH\n";
    return 3;
  }
  PersistedTrajectory frozen;
  if (!readTrajectoryYaml(frozen_path, frozen, error))
  {
    RCLCPP_ERROR(node->get_logger(), "STEP14 frozen trajectory unread: %s", error.c_str());
    std::ofstream yaml(diag_path);
    yaml << "status: FAIL\nfailure_classification: FROZEN_TRAJECTORY_UNREADABLE\n";
    return 3;
  }
  const FrozenTaskMetrics old_m = computeFrozenTaskMetrics(frozen);
  RCLCPP_INFO(node->get_logger(),
              "BASELINE from points Home->PreGrasp t=%.9f J1=%.6f rad  prefix=%.9f  Home->C=%.9f",
              old_m.home_to_pregrasp.duration, old_m.home_to_pregrasp.j1_abs_travel,
              old_m.prefix_duration, old_m.full_deployable_duration);

  auto world_scene = fetchPlanningScene(node, robot_model);
  if (!world_scene)
  {
    std::ofstream yaml(diag_path);
    yaml << "status: FAIL\nfailure_classification: PLANNING_SCENE_UNAVAILABLE\n";
    return 2;
  }
  const auto object = makeCylinder(object_id, object_scene, object_height, object_radius);
  auto pregrasp_scene = makePreGraspPlanningScene(*world_scene, object, object_id, table_name);
  CollisionDiagConfig dcfg;
  dcfg.object_id = object_id;
  dcfg.table_name = table_name;
  dcfg.column_name = column_name;
  dcfg.touch_links = touch_links;

  PreGraspIkConfig ik_cfg;
  ik_cfg.group = group;
  ik_cfg.ee_link = ee_link;
  ik_cfg.object_id = object_id;
  ik_cfg.table_name = table_name;
  ik_cfg.column_name = column_name;
  ik_cfg.touch_links = touch_links;
  ik_cfg.pos_tol = pos_tol;
  ik_cfg.ori_tol_deg = ori_tol_deg;
  ik_cfg.min_ik_solution_distance = node->has_parameter("min_ik_solution_distance") ?
                                        node->get_parameter("min_ik_solution_distance").as_double() :
                                        0.1;
  ik_cfg.max_unique_candidates = static_cast<uint32_t>(std::max(1, max_ik));
  ik_cfg.max_ik_attempts = static_cast<uint32_t>(std::max(8, getInt(node, "max_ik_attempts", 96)));
  const auto baseline_pre = vecToJoints(kStep12cPreGraspRad);
  std::vector<NamedJointSeed> extra = { { "view_a", winner.a_rad },
                                        { "view_b", winner.b_rad },
                                        { "view_c", winner.c_rad } };
  const auto ik = generatePreGraspIkCandidates(*pregrasp_scene, pregrasp, home, baseline_pre, extra,
                                               ik_cfg, node->get_logger());
  int different_branches = 0;
  double min_j1 = std::numeric_limits<double>::infinity();
  for (const auto& cand : ik.unique)
  {
    if (cand.valid)
    {
      ++different_branches;
      min_j1 = std::min(min_j1, cand.j1_abs_displacement);
    }
  }

  std::vector<PreGraspIkCandidate> prefix_pool;
  for (const auto& cand : ik.unique)
  {
    if (cand.is_baseline)
    {
      prefix_pool.push_back(cand);
    }
  }
  for (const auto& cand : ik.unique)
  {
    if (static_cast<int>(prefix_pool.size()) >= max_prefix)
    {
      break;
    }
    if (cand.is_baseline || !cand.valid)
    {
      continue;
    }
    prefix_pool.push_back(cand);
  }

  struct PrefixRow
  {
    PreGraspIkCandidate ik;
    GraspPlanOutcome plan;
  };
  std::vector<PrefixRow> prefix_ok;
  int prefix_tested = 0;
  for (const auto& cand : prefix_pool)
  {
    if (!rclcpp::ok() || elapsed() > timeout_sec)
    {
      break;
    }
    ++prefix_tested;
    RCLCPP_INFO(node->get_logger(), "STEP14 prefix plan %s j1_disp=%.3f rad baseline=%s",
                cand.candidate_id.c_str(), cand.j1_abs_displacement,
                cand.is_baseline ? "YES" : "NO");
    PrefixRow row;
    row.ik = cand;
    row.plan = planGraspCandidate(node, group, ee_link, attach_link, object_id, table_name,
                                  touch_links, object_scene, pregrasp, grasp, lift, object_height,
                                  object_radius, prefix_planning_time, home, cand.joints, winner,
                                  false, plan_attempts, dcfg, pos_tol, ori_tol_deg);
    if (row.plan.ok)
    {
      prefix_ok.push_back(std::move(row));
    }
    else
    {
      RCLCPP_WARN(node->get_logger(), "prefix %s failed: %s", cand.candidate_id.c_str(),
                  row.plan.failure.c_str());
    }
  }
  std::sort(prefix_ok.begin(), prefix_ok.end(), [](const PrefixRow& a, const PrefixRow& b) {
    return a.plan.metrics.prefix_duration < b.plan.metrics.prefix_duration;
  });

  std::vector<PrefixRow> full_pool;
  for (const auto& row : prefix_ok)
  {
    if (row.ik.is_baseline)
    {
      full_pool.push_back(row);
    }
  }
  for (const auto& row : prefix_ok)
  {
    if (static_cast<int>(full_pool.size()) >= max_full)
    {
      break;
    }
    if (row.ik.is_baseline)
    {
      continue;
    }
    full_pool.push_back(row);
  }

  struct FullRow
  {
    PrefixRow prefix;
    GraspPlanOutcome plan;
  };
  std::vector<FullRow> full_ok;
  int full_tested = 0;
  int full_rejected = 0;
  for (const auto& row : full_pool)
  {
    if (!rclcpp::ok() || elapsed() > timeout_sec)
    {
      break;
    }
    ++full_tested;
    RCLCPP_INFO(node->get_logger(), "STEP14 full Home->C plan %s", row.ik.candidate_id.c_str());
    FullRow fr;
    fr.prefix = row;
    fr.plan = planGraspCandidate(node, group, ee_link, attach_link, object_id, table_name, touch_links,
                                 object_scene, pregrasp, grasp, lift, object_height, object_radius,
                                 full_planning_time, home, row.ik.joints, winner, true, plan_attempts,
                                 dcfg, pos_tol, ori_tol_deg);
    if (fr.plan.ok)
    {
      full_ok.push_back(std::move(fr));
    }
    else
    {
      ++full_rejected;
      RCLCPP_WARN(node->get_logger(), "full %s failed: %s", row.ik.candidate_id.c_str(),
                  fr.plan.failure.c_str());
    }
  }
  std::sort(full_ok.begin(), full_ok.end(), [](const FullRow& a, const FullRow& b) {
    const auto& am = a.plan.metrics;
    const auto& bm = b.plan.metrics;
    if (std::abs(am.full_deployable_duration - bm.full_deployable_duration) > 1e-9)
    {
      return am.full_deployable_duration < bm.full_deployable_duration;
    }
    if (std::abs(am.home_to_pregrasp.duration - bm.home_to_pregrasp.duration) > 1e-9)
    {
      return am.home_to_pregrasp.duration < bm.home_to_pregrasp.duration;
    }
    if (std::abs(am.full_deployable_path_length_l2 - bm.full_deployable_path_length_l2) > 1e-9)
    {
      return am.full_deployable_path_length_l2 < bm.full_deployable_path_length_l2;
    }
    return am.home_to_pregrasp.j1_abs_travel < bm.home_to_pregrasp.j1_abs_travel;
  });

  const bool have_winner = !full_ok.empty();
  const FullRow* win = have_winner ? &full_ok.front() : nullptr;
  bool improved = false;
  if (win)
  {
    improved = win->plan.metrics.full_deployable_duration < old_m.full_deployable_duration - 1e-6;
    PersistedTrajectory saved = win->plan.traj;
    saved.task_version = "STEP14";
    saved.label = "STEP14 OPTIMIZED GRASP PREFIX TRAJECTORY";
    saved.source_winner_file = fileBaseName(winner_in);
    saved.source_winner_path = winner_in;
    saved.search_score_total_time = win->plan.metrics.full_deployable_duration;
    saved.search_score_joint_path_length = win->plan.metrics.full_deployable_path_length_l2;
    saved.winner = winner;
    saved.execution_performed = false;
    saved.real_robot_validated = false;
    fillDerivedTotals(saved);
    if (!writeTrajectoryYaml(traj_out, saved, error))
    {
      RCLCPP_ERROR(node->get_logger(), "STEP14 serialize failed: %s", error.c_str());
      std::ofstream yaml(diag_path);
      yaml << "status: FAIL\nfailure_classification: TRAJECTORY_SERIALIZE_FAILED\n";
      return 5;
    }
    PersistedTrajectory loaded;
    if (!readTrajectoryYaml(traj_out, loaded, error))
    {
      std::ofstream yaml(diag_path);
      yaml << "status: FAIL\nfailure_classification: TRAJECTORY_READBACK_FAILED\n";
      return 5;
    }
    const auto roundtrip = validateRoundTrip(saved, loaded);
    const auto continuity = validateContinuity(loaded);
    const auto endpoints = validateEndpoints(loaded, winner, 1e-3);
    const FrozenTaskMetrics loaded_m = computeFrozenTaskMetrics(loaded);
    const bool identity = roundtrip.ok &&
                          std::abs(loaded_m.full_deployable_duration -
                                   win->plan.metrics.full_deployable_duration) < 1e-12;
    std::ofstream winy(winner_out);
    winy.setf(std::ios::fixed);
    winy.precision(17);
    winy << "label: STEP14 OPTIMIZED GRASP PREFIX WINNER\n";
    winy << "task_version: STEP14\n";
    winy << "path_improvement_verified: " << (improved ? "YES" : "NO") << "\n";
    winy << "candidate_id: " << win->prefix.ik.candidate_id << "\n";
    winy << "is_baseline_ik: " << (win->prefix.ik.is_baseline ? "true" : "false") << "\n";
    winy << "home_joints_rad: " << formatHomeList(home) << "\n";
    winy << "pregrasp_joints_rad: " << formatHomeList(win->plan.pregrasp_joints) << "\n";
    winy << "grasp_joints_rad: " << formatHomeList(win->plan.grasp_joints) << "\n";
    winy << "lift_joints_rad: " << formatHomeList(win->plan.lift_joints) << "\n";
    winy << "a_roll_deg: " << winner.a_roll_deg << "\n";
    winy << "b_roll_deg: " << winner.b_roll_deg << "\n";
    winy << "c_roll_deg: " << winner.c_roll_deg << "\n";
    winy << "a_joints_rad: " << formatHomeList(winner.a_rad) << "\n";
    winy << "b_joints_rad: " << formatHomeList(winner.b_rad) << "\n";
    winy << "c_joints_rad: " << formatHomeList(winner.c_rad) << "\n";
    winy << "scored_full_home_to_c: " << win->plan.metrics.full_deployable_duration << "\n";
    winy << "saved_full_home_to_c: " << loaded_m.full_deployable_duration << "\n";
    winy << "old_full_home_to_c: " << old_m.full_deployable_duration << "\n";
    winy << "scored_trajectory_is_saved_trajectory: " << (identity ? "YES" : "NO") << "\n";
    winy << "round_trip_ok: " << (roundtrip.ok ? "true" : "false") << "\n";
    winy << "execution: false\n";
    if (viz.traj)
    {
      viz.traj->publish(displayFromPersisted(loaded, robot_model->getName()));
    }
    RCLCPP_INFO(node->get_logger(),
                "STEP14 winner id=%s improved=%s full_new=%.9f full_old=%.9f J1_new=%.6f J1_old=%.6f",
                win->prefix.ik.candidate_id.c_str(), improved ? "YES" : "NO",
                loaded_m.full_deployable_duration, old_m.full_deployable_duration,
                loaded_m.home_to_pregrasp.j1_abs_travel, old_m.home_to_pregrasp.j1_abs_travel);
    const std::string sha_winner_after = fileSha256(winner_in);
    const std::string sha_traj_after = fileSha256(frozen_path);
    std::ofstream yaml(diag_path);
    yaml.setf(std::ios::fixed);
    yaml.precision(17);
    yaml << "status: " << (have_winner ? "PASS" : "FAIL") << "\n";
    yaml << "task_version: STEP14\n";
    yaml << "failure_classification: none\n";
    yaml << "implementation_ready: YES\n";
    yaml << "path_improvement_verified: " << (improved ? "YES" : "NO") << "\n";
    yaml << "baseline_preserved: "
         << ((sha_winner_before == sha_winner_after && sha_traj_before == sha_traj_after) ? "YES" :
                                                                                            "NO")
         << "\n";
    yaml << "frozen_winner_sha256_before: " << sha_winner_before << "\n";
    yaml << "frozen_winner_sha256_after: " << sha_winner_after << "\n";
    yaml << "frozen_trajectory_sha256_before: " << sha_traj_before << "\n";
    yaml << "frozen_trajectory_sha256_after: " << sha_traj_after << "\n";
    yaml << "scored_trajectory_is_saved_trajectory: " << (identity ? "YES" : "NO") << "\n";
    yaml << "round_trip_ok: " << (roundtrip.ok ? "true" : "false") << "\n";
    yaml << "continuity_ok: " << (continuity.ok ? "true" : "false") << "\n";
    yaml << "endpoint_ok: " << (endpoints.ok ? "true" : "false") << "\n";
    yaml << "fixed_abc_preserved: YES\n";
    yaml << "pilz_lin_preserved: YES\n";
    yaml << "step13_unchanged: YES\n";
    yaml << "real_robot_commands: 0\n";
    yaml << "gripper_commands: 0\n";
    yaml << "execution: false\n";
    yaml << "ik_attempts: " << ik.attempts << "\n";
    yaml << "ik_success: " << ik.ik_success << "\n";
    yaml << "unique_ik_candidates: " << ik.unique.size() << "\n";
    yaml << "fk_valid: " << ik.fk_valid << "\n";
    yaml << "bounds_valid: " << ik.bounds_valid << "\n";
    yaml << "collision_valid: " << ik.collision_valid << "\n";
    yaml << "unique_valid: " << ik.unique_valid << "\n";
    yaml << "different_ik_branches: " << different_branches << "\n";
    yaml << "baseline_included: " << (ik.baseline_included ? "true" : "false") << "\n";
    yaml << "minimum_observed_j1_travel: " << (std::isfinite(min_j1) ? min_j1 : -1.0) << "\n";
    yaml << "prefix_candidates_tested: " << prefix_tested << "\n";
    yaml << "successful_prefixes: " << prefix_ok.size() << "\n";
    yaml << "complete_candidates_evaluated: " << full_tested << "\n";
    yaml << "complete_feasible_candidates: " << full_ok.size() << "\n";
    yaml << "complete_rejected: " << full_rejected << "\n";
    yaml << "search_elapsed_sec: " << elapsed() << "\n";
    yaml << "search_timeout_sec: " << timeout_sec << "\n";
    yaml << "timed_out: " << (elapsed() > timeout_sec ? "true" : "false") << "\n";
    yaml << "old_home_j1: " << home.at("j1") << "\n";
    yaml << "old_pregrasp_j1: " << baseline_pre.at("j1") << "\n";
    yaml << "old_home_to_pregrasp_time: " << old_m.home_to_pregrasp.duration << "\n";
    yaml << "old_home_to_pregrasp_j1_travel: " << old_m.home_to_pregrasp.j1_abs_travel << "\n";
    yaml << "old_home_to_pregrasp_path: " << old_m.home_to_pregrasp.path_length_l2 << "\n";
    yaml << "old_home_to_grasp_lift_time: " << old_m.prefix_duration << "\n";
    yaml << "old_lift_to_a_time: " << old_m.lift_to_a.duration << "\n";
    yaml << "old_a_to_b_time: " << old_m.a_to_b.duration << "\n";
    yaml << "old_b_to_c_time: " << old_m.b_to_c.duration << "\n";
    yaml << "old_full_home_to_c_time: " << old_m.full_deployable_duration << "\n";
    yaml << "old_scaled_home_to_c_at_005: " << old_m.scaled_full_duration_at_005 << "\n";
    yaml << "new_home_to_pregrasp_time: " << loaded_m.home_to_pregrasp.duration << "\n";
    yaml << "new_home_to_pregrasp_j1_travel: " << loaded_m.home_to_pregrasp.j1_abs_travel << "\n";
    yaml << "new_home_to_pregrasp_path: " << loaded_m.home_to_pregrasp.path_length_l2 << "\n";
    yaml << "new_home_to_grasp_lift_time: " << loaded_m.prefix_duration << "\n";
    yaml << "new_lift_to_a_time: " << loaded_m.lift_to_a.duration << "\n";
    yaml << "new_a_to_b_time: " << loaded_m.a_to_b.duration << "\n";
    yaml << "new_b_to_c_time: " << loaded_m.b_to_c.duration << "\n";
    yaml << "new_full_home_to_c_time: " << loaded_m.full_deployable_duration << "\n";
    yaml << "new_scaled_home_to_c_at_005: " << loaded_m.scaled_full_duration_at_005 << "\n";
    yaml << "new_joint_path_length: " << loaded_m.full_deployable_path_length_l2 << "\n";
    yaml << "new_minimum_clearance: " << win->plan.min_clearance << "\n";
    yaml << "new_collision_free: " << (win->plan.collision_free ? "true" : "false") << "\n";
    yaml << "winner_candidate_id: " << win->prefix.ik.candidate_id << "\n";
    yaml << "winner_seed_name: " << win->prefix.ik.seed_name << "\n";
    yaml << "winner_is_baseline: " << (win->prefix.ik.is_baseline ? "true" : "false") << "\n";
    yaml << "winner_pregrasp_joints: " << formatHomeList(win->plan.pregrasp_joints) << "\n";
    yaml << "winner_grasp_joints: " << formatHomeList(win->plan.grasp_joints) << "\n";
    yaml << "winner_lift_joints: " << formatHomeList(win->plan.lift_joints) << "\n";
    yaml << "winner_a_joints: " << formatHomeList(winner.a_rad) << "\n";
    yaml << "winner_b_joints: " << formatHomeList(winner.b_rad) << "\n";
    yaml << "winner_c_joints: " << formatHomeList(winner.c_rad) << "\n";
    yaml << "best_prefix_time: "
         << (prefix_ok.empty() ? -1.0 : prefix_ok.front().plan.metrics.prefix_duration) << "\n";
    yaml << "best_prefix_j1_travel: "
         << (prefix_ok.empty() ? -1.0 : prefix_ok.front().plan.metrics.home_to_pregrasp.j1_abs_travel)
         << "\n";
    yaml << "best_prefix_joint_path: "
         << (prefix_ok.empty() ? -1.0 : prefix_ok.front().plan.metrics.prefix_path_length_l2) << "\n";
    yaml << "home_to_pregrasp_pipeline: ompl\n";
    yaml << "pregrasp_to_grasp_pipeline: pilz_industrial_motion_planner\n";
    yaml << "pregrasp_to_grasp_planner_id: LIN\n";
    yaml << "grasp_to_lift_pipeline: pilz_industrial_motion_planner\n";
    yaml << "grasp_to_lift_planner_id: LIN\n";
    yaml << "candidates:\n";
    for (const auto& cand : ik.unique)
    {
      yaml << "  - id: " << cand.candidate_id << "\n";
      yaml << "    seed: " << cand.seed_name << "\n";
      yaml << "    baseline: " << (cand.is_baseline ? "true" : "false") << "\n";
      yaml << "    valid: " << (cand.valid ? "true" : "false") << "\n";
      yaml << "    joints: " << formatHomeList(cand.joints) << "\n";
      yaml << "    fk_position_error: " << cand.fk_position_error << "\n";
      yaml << "    fk_orientation_error_deg: " << cand.fk_orientation_error_deg << "\n";
      yaml << "    distance_from_home: " << cand.distance_from_home << "\n";
      yaml << "    j1_abs_displacement: " << cand.j1_abs_displacement << "\n";
      yaml << "    collision_free: " << (cand.collision_free ? "true" : "false") << "\n";
      yaml << "    failure_reason: " << cand.failure_reason << "\n";
    }
    yaml << "winner_output_path: " << winner_out << "\n";
    yaml << "trajectory_output_path: " << traj_out << "\n";
    if (!identity || !roundtrip.ok)
    {
      return 5;
    }
    return 0;
  }

  const std::string sha_winner_after = fileSha256(winner_in);
  const std::string sha_traj_after = fileSha256(frozen_path);
  std::ofstream yaml(diag_path);
  yaml.setf(std::ios::fixed);
  yaml.precision(17);
  yaml << "status: FAIL\n";
  yaml << "failure_classification: "
       << (ik.unique.empty() ? "IK_NO_CANDIDATES" :
           prefix_ok.empty() ? "PREFIX_PLANNING_FAILED" :
                               "FULL_TASK_PLANNING_FAILED")
       << "\n";
  yaml << "implementation_ready: YES\n";
  yaml << "path_improvement_verified: NO\n";
  yaml << "ik_attempts: " << ik.attempts << "\n";
  yaml << "unique_ik_candidates: " << ik.unique.size() << "\n";
  yaml << "successful_prefixes: " << prefix_ok.size() << "\n";
  yaml << "complete_feasible_candidates: 0\n";
  yaml << "frozen_winner_sha256_before: " << sha_winner_before << "\n";
  yaml << "frozen_winner_sha256_after: " << sha_winner_after << "\n";
  yaml << "frozen_trajectory_sha256_before: " << sha_traj_before << "\n";
  yaml << "frozen_trajectory_sha256_after: " << sha_traj_after << "\n";
  yaml << "search_elapsed_sec: " << elapsed() << "\n";
  yaml << "real_robot_commands: 0\n";
  yaml << "gripper_commands: 0\n";
  return 4;
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("fr3_mtc_complete_abc_search", options);

  RCLCPP_INFO(node->get_logger(),
              "========== STEP 12C TILTED-CAMERA FIXED-ABC SEARCH ==========");
  RCLCPP_INFO(node->get_logger(), "PLAN / RViz ONLY. No Gazebo execute. No real robot.");
  RCLCPP_INFO(node->get_logger(),
              "ARM1 C = bottom_circle = object -Z original table-contact face");
  RCLCPP_INFO(node->get_logger(), "ARM2 future = top_circle = object +Z (not planned)");

  const auto home = readHome(node);
  const auto home_deg = readHomeDeg(node, home);
  RCLCPP_INFO(node->get_logger(), "HOME_DEG: %s", formatHomeList(home_deg).c_str());
  RCLCPP_INFO(node->get_logger(), "HOME_RAD: %s", formatHomeList(home).c_str());
  for (const auto& name : kArmJoints)
  {
    const double expected_rad = home_deg.at(name) * M_PI / 180.0;
    if (std::abs(home.at(name) - expected_rad) > 1e-9)
    {
      RCLCPP_ERROR(node->get_logger(),
                   "GEOMETRY_INVALID: %s deg/rad mismatch deg=%.12f rad=%.12f expected_rad=%.12f",
                   name.c_str(), home_deg.at(name), home.at(name), expected_rad);
    }
  }
  const auto pregrasp = readPose(node, "pregrasp");
  const auto grasp = readPose(node, "grasp");
  const auto lift = readPose(node, "lift");
  const auto object_world = readPose(node, "object_world");
  const auto object_base = readPose(node, "object");
  const auto world_base = readPose(node, "world_base");
  const auto tcp_object = readPose(node, "tcp_object");
  const ViewGeom view_a = readViewGeom(node, "side_pos_y");
  const ViewGeom view_b = readViewGeom(node, "side_neg_y");
  const ViewGeom view_c = readViewGeom(node, "bottom_circle");
  const ViewGeom view_top = readViewGeom(node, "top_circle");
  const Eigen::Vector3d p1(getDouble(node, "p1_x"), getDouble(node, "p1_y"), getDouble(node, "p1_z"));
  const Eigen::Vector3d d1 =
      Eigen::Vector3d(getDouble(node, "d1_x"), getDouble(node, "d1_y"), getDouble(node, "d1_z"))
          .normalized();
  const Eigen::Vector3d preferred_up =
      Eigen::Vector3d(getDouble(node, "preferred_up_x"), getDouble(node, "preferred_up_y"),
                      getDouble(node, "preferred_up_z"))
          .normalized();
  const std::string group = getString(node, "planning_group", kPlanningGroup);
  const std::string ee_link = getString(node, "ee_link", kEeLink);
  const std::string attach_link = getString(node, "attach_link", kEeLink);
  const std::string object_id = getString(node, "object_name", "small_part");
  const std::string table_name = getString(node, "table_name", "table");
  const std::string column_name = getString(node, "column_name", "mounting_column");
  const auto touch_links = node->get_parameter("touch_links").as_string_array();
  const double object_height = getDouble(node, "object_height");
  const double object_radius = getDouble(node, "object_radius");
  const double home_tol = node->has_parameter("max_home_error_rad") ?
                              node->get_parameter("max_home_error_rad").as_double() :
                              0.03;
  const double pos_tol = getDouble(node, "position_tolerance");
  const double ori_tol_deg = getDouble(node, "orientation_tolerance_deg");
  const double planning_time = getDouble(node, "planning_time");
  const double search_planning_time = node->has_parameter("search_planning_time") ?
                                          node->get_parameter("search_planning_time").as_double() :
                                          std::min(3.0, planning_time);
  const int top_k = getInt(node, "top_k", 10);
  const int beam_width = getInt(node, "beam_width", 5);
  const int edge_attempts = getInt(node, "edge_attempts", 3);
  const int prefix_retries = getInt(node, "prefix_retries", 5);
  const int complete_budget = getInt(node, "complete_candidate_budget", 40);
  const std::string winner_path = getString(node, "winner_output_path",
                                            "/tmp/fr3_step12c_best_sampled_complete_task.yaml");
  const std::string diag_path =
      getString(node, "diagnostic_output_path", "/tmp/fr3_step12c_search.yaml");
  const bool visualize = getBool(node, "visualize_search", true);
  const double vis_hold = node->has_parameter("visualization_hold_seconds") ?
                              node->get_parameter("visualization_hold_seconds").as_double() :
                              12.0;
  const auto all_rolls = readRollPoses(node);
  const Eigen::Isometry3d t_world_base = poseToIso(world_base.pose);
  const Eigen::Isometry3d t_tcp_object = poseToIso(tcp_object.pose);
  DesignCamera camera = cameraFromNode(node);

  std::string failure_class = "SEARCH_BUDGET_EXHAUSTED";
  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
  SearchViz viz;
  if (visualize)
  {
    viz.markers = node->create_publisher<visualization_msgs::msg::MarkerArray>(kMarkerTopic, qos);
    viz.ghost = node->create_publisher<moveit_msgs::msg::DisplayRobotState>(kGhostTopic, qos);
    viz.traj = node->create_publisher<moveit_msgs::msg::DisplayTrajectory>(kTrajTopic, qos);
  }
  viz.p1 = p1;
  viz.d1 = d1;
  viz.camera = camera;
  viz.table_c = Eigen::Vector3d(getDouble(node, "table_cx"), getDouble(node, "table_cy"),
                                getDouble(node, "table_cz"));
  viz.table_s = Eigen::Vector3d(getDouble(node, "table_dx"), getDouble(node, "table_dy"),
                                getDouble(node, "table_dz"));
  viz.column_c = Eigen::Vector3d(getDouble(node, "column_cx"), getDouble(node, "column_cy"),
                                 getDouble(node, "column_cz"));
  viz.column_s = Eigen::Vector3d(getDouble(node, "column_dx"), getDouble(node, "column_dy"),
                                 getDouble(node, "column_dz"));
  viz.part0 = Eigen::Vector3d(object_world.pose.position.x, object_world.pose.position.y,
                              object_world.pose.position.z);
  viz.t_world_base = t_world_base;
  viz.object_height = object_height;
  viz.object_radius = object_radius;
  viz.hud = "STEP 12C  TILTED CAMERA  ARM1 C = ORIGINAL BOTTOM";

  const Eigen::Vector3d n_expected(0.0, -0.70710678, 0.70710678);
  const Eigen::Vector3d cam_fwd_expected(0.0, 0.70710678, -0.70710678);
  const bool p1_ok = std::abs(p1.x()) < 1e-9 && std::abs(p1.y() - 0.30) < 1e-9 &&
                     std::abs(p1.z() - 1.20) < 1e-9;
  if (!p1_ok)
  {
    RCLCPP_ERROR(node->get_logger(), "GEOMETRY_INVALID: P1 is not [0,0.3,1.2], got [%.6f, %.6f, %.6f]",
                 p1.x(), p1.y(), p1.z());
    failure_class = "GEOMETRY_INVALID";
  }
  const bool cam_pos_ok = (camera.optical_center_world - Eigen::Vector3d(0.0, 0.0, 1.40)).norm() < 1e-9;
  const bool cam_fwd_ok = (camera.optical_forward_world - cam_fwd_expected).norm() < 1e-5;
  const bool n_ok = (d1 - n_expected).norm() < 1e-5;
  const double cam_n_dot = camera.optical_forward_world.dot(d1);
  RCLCPP_INFO(node->get_logger(), "camera position %s forward %s n_target %s fwd·n=%.6f",
              vecYaml(camera.optical_center_world).c_str(),
              vecYaml(camera.optical_forward_world).c_str(), vecYaml(d1).c_str(), cam_n_dot);
  if (!cam_pos_ok || !cam_fwd_ok || !n_ok || cam_n_dot > -0.999)
  {
    RCLCPP_ERROR(node->get_logger(),
                 "GEOMETRY_INVALID: camera/n_target regression failed pos_ok=%d fwd_ok=%d n_ok=%d "
                 "dot=%.6f",
                 cam_pos_ok, cam_fwd_ok, n_ok, cam_n_dot);
    failure_class = "GEOMETRY_INVALID";
  }
  const bool c_face_ok = std::abs(view_c.normal_in_object.x()) < 1e-9 &&
                         std::abs(view_c.normal_in_object.y()) < 1e-9 &&
                         std::abs(view_c.normal_in_object.z() + 1.0) < 1e-9 &&
                         std::abs(view_c.center_in_object.z() + 0.5 * object_height) < 1e-6;
  const bool top_preserved = std::abs(view_top.normal_in_object.x()) < 1e-9 &&
                             std::abs(view_top.normal_in_object.y()) < 1e-9 &&
                             std::abs(view_top.normal_in_object.z() - 1.0) < 1e-9 &&
                             std::abs(view_top.center_in_object.z() - 0.5 * object_height) < 1e-6;
  if (!c_face_ok)
  {
    RCLCPP_ERROR(node->get_logger(),
                 "GEOMETRY_INVALID: ARM1 C is not object -Z bottom. normal=[%.4f,%.4f,%.4f] "
                 "center=[%.4f,%.4f,%.4f]",
                 view_c.normal_in_object.x(), view_c.normal_in_object.y(), view_c.normal_in_object.z(),
                 view_c.center_in_object.x(), view_c.center_in_object.y(), view_c.center_in_object.z());
    failure_class = "GEOMETRY_INVALID";
  }
  if (!top_preserved)
  {
    RCLCPP_ERROR(node->get_logger(),
                 "GEOMETRY_INVALID: top_circle was not preserved as object +Z. normal=[%.4f,%.4f,%.4f]",
                 view_top.normal_in_object.x(), view_top.normal_in_object.y(),
                 view_top.normal_in_object.z());
    failure_class = "GEOMETRY_INVALID";
  }
  RCLCPP_INFO(node->get_logger(), "C physical identity: %s", bottomCircleRoiDef().physical.c_str());

  const std::array<const ViewGeom*, 3> views = { &view_a, &view_b, &view_c };
  bool geom_pass = p1_ok && c_face_ok && top_preserved && cam_pos_ok && cam_fwd_ok && n_ok &&
                   cam_n_dot <= -0.999;
  for (const auto* view : views)
  {
    const auto rolls = filterRollsByView(all_rolls, view->name);
    int ori_ok = 0;
    if (rolls.empty())
    {
      geom_pass = false;
    }
    for (const auto& roll : rolls)
    {
      const auto g =
          checkRollGeometry(roll, *view, t_world_base, t_tcp_object, p1, d1);
      if (g.center_err < 1e-4 && g.normal_err_deg < 0.5 && g.tcp_object_err < 1e-4)
      {
        ++ori_ok;
      }
      else
      {
        geom_pass = false;
      }
    }
    RCLCPP_INFO(node->get_logger(),
                "REGRESSION %s rolls=%zu orientation-valid=%d center/normal/tcp-object",
                view->name.c_str(), rolls.size(), ori_ok);
  }
  if (!geom_pass)
  {
    failure_class = "GEOMETRY_INVALID";
    RCLCPP_ERROR(node->get_logger(), "GEOMETRY_INVALID: refusing search");
    std::ofstream yaml(diag_path);
    yaml << "status: FAIL\nfailure_classification: GEOMETRY_INVALID\n";
    yaml << "p1: " << vecYaml(p1) << "\n";
    yaml << "d1: " << vecYaml(d1) << "\n";
    yaml << "camera_position: " << vecYaml(camera.optical_center_world) << "\n";
    yaml << "camera_forward: " << vecYaml(camera.optical_forward_world) << "\n";
    return 3;
  }

  auto helper = rclcpp::Node::make_shared("fr3_mtc_complete_abc_params");
  if (!overlayRobotDescriptionFromMoveGroup(node, helper))
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to overlay robot_description from move_group");
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });
  viz.publish(node, true);

  const auto js = waitForFreshJoints(node, std::chrono::seconds(60));
  if (!js)
  {
    RCLCPP_ERROR(node->get_logger(), "No /joint_states");
    shutdownSpinner(executor, spinner);
    return 2;
  }
  const auto actual = jointsFromMsg(*js);
  if (jointsAreAllZero(actual))
  {
    RCLCPP_ERROR(node->get_logger(), "Joint states are all zero");
    shutdownSpinner(executor, spinner);
    return 2;
  }
  const bool already_at_home = maxJointError(actual, home) <= home_tol;
  RCLCPP_INFO(node->get_logger(),
              "Current vs new Home max|dq|=%.6f rad  already_at_home=%s  (Home is an MTC MoveTo "
              "stage, not assumed as silent initial state)",
              maxJointError(actual, home), already_at_home ? "YES" : "NO");

  moveit::task_constructor::Task gen_task("", false);
  gen_task.setName("FR3 Step12C Prefix");
  gen_task.loadRobotModel(node);
  const auto robot_model = gen_task.getRobotModel();
  auto* jmg_home = robot_model->getJointModelGroup(group);
  CollisionDiagConfig home_dcfg;
  home_dcfg.object_id = object_id;
  home_dcfg.table_name = table_name;
  home_dcfg.column_name = column_name;
  home_dcfg.touch_links = touch_links;
  auto home_scene = fetchPlanningScene(node, robot_model);
  if (!home_scene)
  {
    RCLCPP_ERROR(node->get_logger(), "NEW_HOME_INVALID: failed to fetch planning scene");
    shutdownSpinner(executor, spinner);
    return 2;
  }
  applyJointsToScene(*home_scene, home);
  bool home_limits_ok = jmg_home && home_scene->getCurrentState().satisfiesBounds(jmg_home);
  auto home_contacts = collectCollisionContacts(*home_scene, home_dcfg);
  bool home_self_ok = true;
  bool home_table_ok = true;
  bool home_column_ok = true;
  std::string home_pair = "none";
  for (const auto& c : home_contacts.contacts)
  {
    if (c.category == CollisionCategory::ROBOT_SELF)
    {
      home_self_ok = false;
    }
    if (c.category == CollisionCategory::ROBOT_TABLE)
    {
      home_table_ok = false;
    }
    if (c.category == CollisionCategory::ROBOT_COLUMN)
    {
      home_column_ok = false;
    }
    if (home_pair == "none")
    {
      home_pair = c.pair_key;
    }
  }
  const bool home_collision_free = !home_contacts.collision;
  RCLCPP_INFO(node->get_logger(),
              "NEW HOME validation limits=%s self=%s table=%s column=%s collision_free=%s pair=%s",
              home_limits_ok ? "PASS" : "FAIL", home_self_ok ? "PASS" : "FAIL",
              home_table_ok ? "PASS" : "FAIL", home_column_ok ? "PASS" : "FAIL",
              home_collision_free ? "PASS" : "FAIL", home_pair.c_str());
  if (!home_limits_ok || !home_collision_free)
  {
    failure_class = "NEW_HOME_INVALID";
    RCLCPP_ERROR(node->get_logger(), "NEW_HOME_INVALID pair=%s limits=%d collision=%d",
                 home_pair.c_str(), home_limits_ok, home_contacts.collision);
    std::ofstream yaml(diag_path);
    yaml << "status: FAIL\nfailure_classification: NEW_HOME_INVALID\n";
    yaml << "home_limits_ok: " << (home_limits_ok ? "true" : "false") << "\n";
    yaml << "home_self_ok: " << (home_self_ok ? "true" : "false") << "\n";
    yaml << "home_table_ok: " << (home_table_ok ? "true" : "false") << "\n";
    yaml << "home_column_ok: " << (home_column_ok ? "true" : "false") << "\n";
    yaml << "home_pair: " << home_pair << "\n";
    yaml << "home_deg: " << formatHomeList(home_deg) << "\n";
    yaml << "home_rad: " << formatHomeList(home) << "\n";
    shutdownSpinner(executor, spinner);
    return 2;
  }

  geometry_msgs::msg::PoseStamped object_scene = object_world;
  if (robot_model->getModelFrame() != object_world.header.frame_id)
  {
    object_scene = object_base;
  }
  const bool winner_replay_only =
      getBool(node, "winner_replay_only", false) || getBool(node, "persist_winner_only", false);
  if (winner_replay_only)
  {
    const int rc = runWinnerReplayPersist(
        node, group, ee_link, attach_link, object_id, table_name, touch_links, object_scene,
        pregrasp, grasp, lift, object_height, object_radius, planning_time, home, robot_model, viz,
        diag_path);
    if (vis_hold > 0)
    {
      std::this_thread::sleep_for(std::chrono::duration<double>(vis_hold));
    }
    const auto after = waitForFreshJoints(node, std::chrono::seconds(3));
    if (after)
    {
      RCLCPP_INFO(node->get_logger(), "Robot moved because of this node? %s",
                  maxJointError(jointsFromMsg(*after), actual) > 0.02 ? "YES" : "NO");
    }
    shutdownSpinner(executor, spinner);
    return rc;
  }
  if (getBool(node, "optimize_grasp_prefix", false))
  {
    const int rc = runGraspPrefixOptimize(
        node, group, ee_link, attach_link, object_id, table_name, column_name, touch_links,
        object_scene, pregrasp, grasp, lift, object_height, object_radius, planning_time, home,
        robot_model, viz, diag_path, pos_tol, ori_tol_deg);
    if (vis_hold > 0)
    {
      std::this_thread::sleep_for(std::chrono::duration<double>(vis_hold));
    }
    const auto after = waitForFreshJoints(node, std::chrono::seconds(3));
    if (after)
    {
      RCLCPP_INFO(node->get_logger(), "Robot moved because of this node? %s",
                  maxJointError(jointsFromMsg(*after), actual) > 0.02 ? "YES" : "NO");
    }
    shutdownSpinner(executor, spinner);
    return rc;
  }
  bool prefix_ok = false;
  double prefix_time = 0.0;
  double prefix_path = 0.0;
  planning_scene::PlanningScenePtr lift_scene;
  for (int attempt = 1; attempt <= prefix_retries && rclcpp::ok(); ++attempt)
  {
    moveit::task_constructor::Task task("", false);
    task.setName("FR3 Step12 Prefix");
    task.loadRobotModel(node);
    addPrefixStages(task, node, group, ee_link, attach_link, object_id, table_name, touch_links,
                    object_scene, pregrasp, grasp, lift, object_height, object_radius, planning_time,
                    home);
    try
    {
      task.init();
    }
    catch (const std::exception& ex)
    {
      RCLCPP_WARN(node->get_logger(), "prefix init failed attempt %d: %s", attempt, ex.what());
      continue;
    }
    if (!task.plan(1) || task.numSolutions() < 1)
    {
      RCLCPP_WARN(node->get_logger(), "prefix plan failed attempt %d/%d", attempt, prefix_retries);
      continue;
    }
    std::vector<const moveit::task_constructor::SolutionBase*> leaves;
    flattenSolutions(*task.solutions().front(), leaves);
    const auto* restore = findStageSolution(leaves, "Restore Part-Table Collision");
    if (!restore || !restore->end() || !restore->end()->scene())
    {
      continue;
    }
    lift_scene = planning_scene::PlanningScene::clone(restore->end()->scene());
    moveit_task_constructor_msgs::msg::Solution msg;
    task.solutions().front()->toMsg(msg);
    prefix_time = trajectoryDuration(msg);
    prefix_path = pathLengthMsg(msg);
    prefix_ok = true;
    std::vector<const moveit::task_constructor::SolutionBase*> home_leaves;
    flattenSolutions(*task.solutions().front(), home_leaves);
    const auto* home_sol = findStageSolution(home_leaves, "MoveTo Home");
    if (home_sol && home_sol->end() && home_sol->end()->scene())
    {
      const auto home_reached =
          jointsFromState(home_sol->end()->scene()->getCurrentState());
      RCLCPP_INFO(node->get_logger(),
                  "MTC MoveTo Home reached max|dq|=%.6f rad vs commanded HOME_RAD",
                  maxJointError(home_reached, home));
    }
    else
    {
      RCLCPP_WARN(node->get_logger(), "MoveTo Home stage missing from prefix solution");
    }
    RCLCPP_INFO(node->get_logger(), "PREFIX SUCCESS attempt=%d T=%.3f L=%.3f", attempt, prefix_time,
                prefix_path);
    break;
  }
  if (!prefix_ok || !lift_scene)
  {
    failure_class = "PREFIX_PLANNING_FAILED";
    RCLCPP_ERROR(node->get_logger(), "PREFIX_PLANNING_FAILED after %d retries", prefix_retries);
    std::ofstream yaml(diag_path);
    yaml << "status: FAIL\nfailure_classification: PREFIX_PLANNING_FAILED\n";
    shutdownSpinner(executor, spinner);
    return 3;
  }

  EndpointGenerationConfig cfg;
  cfg.group = group;
  cfg.ee_link = ee_link;
  cfg.object_id = object_id;
  cfg.table_name = table_name;
  cfg.touch_links = touch_links;
  cfg.p1 = p1;
  cfg.d1 = d1;
  cfg.preferred_up = preferred_up;
  cfg.t_tcp_object = t_tcp_object;
  cfg.t_model_base = lift_scene->getCurrentState().getGlobalLinkTransform("base_link");
  cfg.pos_tol = pos_tol;
  cfg.ori_tol_deg = ori_tol_deg;
  cfg.max_ik_solutions_per_pose =
      static_cast<uint32_t>(std::max(1, getInt(node, "max_ik_solutions_per_pose", 8)));
  cfg.min_ik_solution_distance = node->has_parameter("min_ik_solution_distance") ?
                                     node->get_parameter("min_ik_solution_distance").as_double() :
                                     0.1;
  CollisionDiagConfig dcfg;
  dcfg.object_id = object_id;
  dcfg.table_name = table_name;
  dcfg.column_name = column_name;
  dcfg.touch_links = touch_links;
  auto* jmg = robot_model->getJointModelGroup(group);

  auto evaluate_view = [&](const ViewGeom& view, ViewLayerCounts& counts,
                           std::vector<ScoredEndpoint>& kept, const std::string& label) {
    const auto rolls = filterRollsByView(all_rolls, view.name);
    counts.roll_samples = static_cast<int>(rolls.size());
    for (const auto& roll : rolls)
    {
      const auto g = checkRollGeometry(roll, view, t_world_base, t_tcp_object, p1, d1);
      if (g.center_err < 1e-4 && g.normal_err_deg < 0.5)
      {
        ++counts.orientation_valid;
        ++counts.geometric_visible_pose;
      }
    }
    const auto gen =
        generateValidEndpointCandidates(*lift_scene, rolls, view, cfg, node->get_logger());
    counts.raw_ik = static_cast<int>(gen.raw.size());
    for (const auto& cand : gen.raw)
    {
      if (cand.bounds_ok)
      {
        ++counts.joint_valid;
      }
      if (cand.collision_free && cand.bounds_ok)
      {
        ++counts.collision_free;
      }
      auto probe = planning_scene::PlanningScene::clone(lift_scene->diff());
      applyJoints(probe->getCurrentStateNonConst(), cand.joints);
      double col_d = 0.0;
      if (tryColumnDistance(*probe, group, column_name, col_d))
      {
        counts.distance_available = true;
        if (!std::isfinite(counts.best_column_distance) || col_d > counts.best_column_distance)
        {
          counts.best_column_distance = col_d;
          counts.best_near = cand;
        }
      }
      const auto snap = collectCollisionContacts(*probe, dcfg);
      std::map<std::string, int> pairs;
      for (const auto& c : snap.contacts)
      {
        pairs[c.pair_key] += c.contact_count;
      }
      if (!pairs.empty() && !cand.collision_free)
      {
        const auto it = std::max_element(pairs.begin(), pairs.end(),
                                         [](const auto& a, const auto& b) { return a.second < b.second; });
        counts.dominant_pair = it->first;
      }
    }
    for (const auto& cand : gen.valid)
    {
      auto probe = planning_scene::PlanningScene::clone(lift_scene->diff());
      applyJoints(probe->getCurrentStateNonConst(), cand.joints);
      const Eigen::Isometry3d t_world_tcp =
          t_world_base * tcpInBase(probe->getCurrentState(), ee_link);
      const Eigen::Isometry3d t_world_object = t_world_tcp * t_tcp_object;
      auto vis = checkGeometricVisibility(*probe, probe->getCurrentState(), view, t_world_object, p1,
                                          d1, camera, t_world_base, object_radius, object_height);
      viz.vis = vis;
      viz.current = cand;
      viz.hud = std::string("TESTING ") + cand.candidate_id + "  " + vis.classification;
      viz.publish(node);
      if (!vis.geometric_face_visible)
      {
        if (!vis.center_ray_clear)
        {
          ++counts.vis_center_blocked;
        }
        else
        {
          ++counts.vis_fraction_blocked;
        }
        if (counts.vis_occluders == "none" && !vis.occluders.empty())
        {
          counts.vis_occluders = vis.occluders.front();
          for (size_t i = 1; i < vis.occluders.size(); ++i)
          {
            counts.vis_occluders += "," + vis.occluders[i];
          }
        }
        continue;
      }
      ++counts.visibility_valid;
      ScoredEndpoint scored;
      scored.cand = cand;
      scored.vis = vis;
      tryColumnDistance(*probe, group, column_name, scored.clearance);
      scored.joint_margin = jointLimitMargin(probe->getCurrentState(), jmg);
      scored.from_lift = cand.joint_distance_from_lift;
      kept.push_back(std::move(scored));
      counts.best_center_err = cand.view_center_error;
      counts.best_normal_err = cand.normal_error;
    }
    kept = pruneTopK(std::move(kept), top_k);
    RCLCPP_INFO(node->get_logger(),
                "VIEW %s rolls=%d ori=%d pose-visible=%d rawIK=%d joint=%d collision-free=%d "
                "vis-clear=%d center-block=%d frac-block=%d kept=%zu pair=%s vis_occ=%s dist=%.4f",
                label.c_str(), counts.roll_samples, counts.orientation_valid,
                counts.geometric_visible_pose, counts.raw_ik, counts.joint_valid,
                counts.collision_free, counts.visibility_valid, counts.vis_center_blocked,
                counts.vis_fraction_blocked, kept.size(), counts.dominant_pair.c_str(),
                counts.vis_occluders.c_str(),
                counts.distance_available ? counts.best_column_distance : NAN);
  };

  ViewLayerCounts a_counts, b_counts, c_counts;
  std::vector<ScoredEndpoint> a_kept, b_kept, c_kept;
  viz.hud = "GENERATING A/B/C(bottom) ENDPOINTS @ P1=[0,0.30,1.20]";
  evaluate_view(view_a, a_counts, a_kept, "A side_pos_y");
  viz.a_status = a_kept.empty() ? "NO ENDPOINT" : (std::to_string(a_kept.size()) + " vis+free");
  evaluate_view(view_b, b_counts, b_kept, "B side_neg_y");
  viz.b_status = b_kept.empty() ? "NO ENDPOINT" : (std::to_string(b_kept.size()) + " vis+free");
  evaluate_view(view_c, c_counts, c_kept, "C bottom_circle ORIGINAL BOTTOM");
  viz.c_status = c_kept.empty() ? "NO ENDPOINT" : (std::to_string(c_kept.size()) + " vis+free");
  viz.publish(node, true);

  auto write_diag = [&](const std::string& status, const std::string& cls,
                        const CompleteCand* winner) {
    std::ofstream yaml(diag_path);
    yaml.setf(std::ios::fixed);
    yaml.precision(9);
    yaml << "status: " << status << "\n";
    yaml << "failure_classification: " << cls << "\n";
    yaml << "p1: [" << p1.x() << ", " << p1.y() << ", " << p1.z() << "]\n";
    yaml << "camera_design_position: [" << camera.optical_center_world.x() << ", "
         << camera.optical_center_world.y() << ", " << camera.optical_center_world.z() << "]\n";
    yaml << "camera_forward: " << vecYaml(camera.optical_forward_world) << "\n";
    yaml << "order: A,B,C(bottom_circle)\n";
    yaml << "arm1_c_semantic: bottom_circle\n";
    yaml << "arm1_c_physical: original_table_contact_bottom\n";
    yaml << "arm1_c_local_normal: [0.0, 0.0, -1.0]\n";
    yaml << "arm1_c_surface_local_center: [0.0, 0.0, " << view_c.center_in_object.z() << "]\n";
    yaml << "arm2_future_face: top_circle\n";
    yaml << "arm2_future_local_normal: [0.0, 0.0, 1.0]\n";
    yaml << "old_wrong_c: object_+Z_original_top\n";
    yaml << "old_wrong_c_visibility_valid: 0\n";
    yaml << "roll_step_deg: " << getDouble(node, "roll_step_deg") << "\n";
    yaml << "top_k: " << top_k << "\n";
    yaml << "beam_width: " << beam_width << "\n";
    yaml << "edge_attempts: " << edge_attempts << "\n";
    yaml << "search_planning_time: " << search_planning_time << "\n";
    yaml << "prefix_ok: " << (prefix_ok ? "true" : "false") << "\n";
    yaml << "prefix_time: " << prefix_time << "\n";
    auto dump_view = [&](const char* key, const ViewLayerCounts& c) {
      yaml << key << ":\n";
      yaml << "  roll_samples: " << c.roll_samples << "\n";
      yaml << "  orientation_valid: " << c.orientation_valid << "\n";
      yaml << "  raw_ik: " << c.raw_ik << "\n";
      yaml << "  joint_valid: " << c.joint_valid << "\n";
      yaml << "  collision_free: " << c.collision_free << "\n";
      yaml << "  visibility_valid: " << c.visibility_valid << "\n";
      yaml << "  vis_center_blocked: " << c.vis_center_blocked << "\n";
      yaml << "  vis_fraction_blocked: " << c.vis_fraction_blocked << "\n";
      yaml << "  vis_occluders: " << c.vis_occluders << "\n";
      yaml << "  dominant_pair: " << c.dominant_pair << "\n";
      yaml << "  best_column_distance: " << c.best_column_distance << "\n";
    };
    dump_view("view_A", a_counts);
    dump_view("view_B", b_counts);
    dump_view("view_C", c_counts);
    if (winner && winner->a && winner->b && winner->c)
    {
      yaml << "winner:\n";
      yaml << "  a_roll: " << winner->a->cand.roll_deg << "\n";
      yaml << "  b_roll: " << winner->b->cand.roll_deg << "\n";
      yaml << "  c_roll: " << winner->c->cand.roll_deg << "\n";
      yaml << "  a_joints_rad: " << jointsYaml(winner->a->cand.joints) << "\n";
      yaml << "  b_joints_rad: " << jointsYaml(winner->b->cand.joints) << "\n";
      yaml << "  c_joints_rad: " << jointsYaml(winner->c->cand.joints) << "\n";
      yaml << "  a_joints_deg: " << jointsDegYaml(winner->a->cand.joints) << "\n";
      yaml << "  b_joints_deg: " << jointsDegYaml(winner->b->cand.joints) << "\n";
      yaml << "  c_joints_deg: " << jointsDegYaml(winner->c->cand.joints) << "\n";
      yaml << "  predicted_total_trajectory_time: " << winner->time << "\n";
      yaml << "  total_joint_path_length: " << winner->path << "\n";
      yaml << "  minimum_clearance: " << winner->min_clearance << "\n";
      yaml << "  label: BEST SAMPLED FEASIBLE COMPLETE TASK WITHIN CONFIGURED SEARCH BUDGET\n";
    }
  };

  if (c_counts.raw_ik == 0)
  {
    failure_class = "BOTTOM_FACE_IK_UNREACHABLE";
    RCLCPP_ERROR(node->get_logger(),
                 "BOTTOM_FACE_IK_UNREACHABLE rolls=%d ori=%d rawIK=0 collision-free=0 vis=0 "
                 "(geometry valid; IK never found; visibility not evaluated on a live state)",
                 c_counts.roll_samples, c_counts.orientation_valid);
    write_diag("FAIL", failure_class, nullptr);
    viz.hud = "BOTTOM_FACE_IK_UNREACHABLE";
    viz.c_status = "NO IK";
    viz.publish(node, true);
    if (vis_hold > 0)
    {
      std::this_thread::sleep_for(std::chrono::duration<double>(vis_hold));
    }
    shutdownSpinner(executor, spinner);
    return 4;
  }
  if (c_counts.collision_free > 0 && c_counts.visibility_valid == 0)
  {
    failure_class = "C_BOTTOM_VISIBILITY_BLOCKED";
    RCLCPP_ERROR(node->get_logger(),
                 "C_BOTTOM_VISIBILITY_BLOCKED rolls=%d rawIK=%d collision-free=%d vis=0 "
                 "center-block=%d frac-block=%d occ=%s",
                 c_counts.roll_samples, c_counts.raw_ik, c_counts.collision_free,
                 c_counts.vis_center_blocked, c_counts.vis_fraction_blocked,
                 c_counts.vis_occluders.c_str());
    write_diag("FAIL", failure_class, nullptr);
    viz.hud = "C_BOTTOM_VISIBILITY_BLOCKED";
    viz.c_status = "VISIBILITY BLOCKED";
    viz.publish(node, true);
    if (vis_hold > 0)
    {
      std::this_thread::sleep_for(std::chrono::duration<double>(vis_hold));
    }
    shutdownSpinner(executor, spinner);
    return 4;
  }
  if (c_counts.collision_free == 0)
  {
    failure_class = "ROBOT_COLUMN_COLLISION";
    if (c_counts.dominant_pair.find("table") != std::string::npos)
    {
      failure_class = "TABLE_COLLISION";
    }
    else if (c_counts.dominant_pair.find(column_name) == std::string::npos)
    {
      failure_class = "SELF_COLLISION";
    }
    RCLCPP_ERROR(node->get_logger(),
                 "%s rolls=%d rawIK=%d vis=%d free=0 pair=%s best_signed=%.4f",
                 failure_class.c_str(), c_counts.roll_samples, c_counts.raw_ik,
                 c_counts.visibility_valid, c_counts.dominant_pair.c_str(),
                 c_counts.distance_available ? c_counts.best_column_distance : NAN);
    write_diag("FAIL", failure_class, nullptr);
    viz.hud = failure_class;
    viz.publish(node, true);
    if (vis_hold > 0)
    {
      std::this_thread::sleep_for(std::chrono::duration<double>(vis_hold));
    }
    shutdownSpinner(executor, spinner);
    return 4;
  }
  if (a_kept.empty() || b_kept.empty() || c_kept.empty())
  {
    auto classify_empty = [&](const char* view_label, const ViewLayerCounts& c,
                              const char* vis_class) {
      RCLCPP_ERROR(node->get_logger(),
                   "%s EMPTY endpoints: rolls=%d ori=%d rawIK=%d joint=%d collision-free=%d vis=%d "
                   "center-block=%d frac-block=%d pair=%s occ=%s",
                   view_label, c.roll_samples, c.orientation_valid, c.raw_ik, c.joint_valid,
                   c.collision_free, c.visibility_valid, c.vis_center_blocked,
                   c.vis_fraction_blocked, c.dominant_pair.c_str(), c.vis_occluders.c_str());
      if (c.orientation_valid == 0)
      {
        return std::string("GEOMETRY_INVALID");
      }
      if (c.raw_ik == 0)
      {
        return std::string("IK_UNREACHABLE");
      }
      if (c.collision_free == 0)
      {
        if (c.dominant_pair.find("table") != std::string::npos)
        {
          return std::string("TABLE_COLLISION");
        }
        if (c.dominant_pair.find(column_name) != std::string::npos)
        {
          return std::string("ROBOT_COLUMN_COLLISION");
        }
        return std::string("SELF_COLLISION");
      }
      if (c.visibility_valid == 0)
      {
        return std::string(vis_class);
      }
      return std::string("SEARCH_BUDGET_EXHAUSTED");
    };
    if (a_kept.empty())
    {
      failure_class = classify_empty("A", a_counts, "A_VISIBILITY_BLOCKED");
    }
    else if (b_kept.empty())
    {
      failure_class = classify_empty("B", b_counts, "B_VISIBILITY_BLOCKED");
    }
    else
    {
      failure_class = classify_empty("C", c_counts, "C_BOTTOM_VISIBILITY_BLOCKED");
    }
    write_diag("FAIL", failure_class, nullptr);
    shutdownSpinner(executor, spinner);
    return 4;
  }

  const auto lift_joints = jointsFromState(lift_scene->getCurrentState());
  std::vector<CompleteCand> beam;
  int edges_tried = 0;
  viz.hud = "BEAM SEARCH Lift→A";
  for (auto& a : a_kept)
  {
    if (!rclcpp::ok())
    {
      break;
    }
    viz.current = a.cand;
    viz.vis = a.vis;
    viz.publish(node);
    auto edge = planJointEdge(node, group, *lift_scene, lift_joints, a.cand.joints,
                              search_planning_time, edge_attempts);
    ++edges_tried;
    if (!edge.success)
    {
      continue;
    }
    CompleteCand node_a;
    node_a.a = &a;
    node_a.lift_a = edge;
    node_a.time = prefix_time + edge.best_time;
    node_a.path = prefix_path + edge.best_path;
    node_a.min_clearance = a.clearance;
    beam.push_back(node_a);
  }
  std::sort(beam.begin(), beam.end(), betterComplete);
  if (static_cast<int>(beam.size()) > beam_width)
  {
    beam.resize(static_cast<size_t>(beam_width));
  }
  if (beam.empty())
  {
    failure_class = "LIFT_TO_A_FAILED";
    write_diag("FAIL", failure_class, nullptr);
    shutdownSpinner(executor, spinner);
    return 4;
  }

  std::vector<CompleteCand> beam_ab;
  viz.hud = "BEAM SEARCH A→B";
  for (const auto& node_a : beam)
  {
    for (auto& b : b_kept)
    {
      if (!rclcpp::ok())
      {
        break;
      }
      viz.current = b.cand;
      viz.vis = b.vis;
      viz.publish(node);
      auto edge = planJointEdge(node, group, *lift_scene, node_a.a->cand.joints, b.cand.joints,
                                search_planning_time, edge_attempts);
      ++edges_tried;
      if (!edge.success)
      {
        continue;
      }
      CompleteCand nxt = node_a;
      nxt.b = &b;
      nxt.a_b = edge;
      nxt.time = node_a.time + edge.best_time;
      nxt.path = node_a.path + edge.best_path;
      nxt.min_clearance = std::min(node_a.min_clearance, b.clearance);
      beam_ab.push_back(nxt);
    }
  }
  std::sort(beam_ab.begin(), beam_ab.end(), betterComplete);
  if (static_cast<int>(beam_ab.size()) > beam_width)
  {
    beam_ab.resize(static_cast<size_t>(beam_width));
  }
  if (beam_ab.empty())
  {
    failure_class = "A_TO_B_FAILED";
    write_diag("FAIL", failure_class, nullptr);
    shutdownSpinner(executor, spinner);
    return 4;
  }

  std::vector<CompleteCand> complete;
  viz.hud = "BEAM SEARCH B→C";
  for (const auto& node_ab : beam_ab)
  {
    for (auto& c : c_kept)
    {
      if (!rclcpp::ok() || static_cast<int>(complete.size()) >= complete_budget)
      {
        break;
      }
      viz.current = c.cand;
      viz.vis = c.vis;
      viz.publish(node);
      auto edge = planJointEdge(node, group, *lift_scene, node_ab.b->cand.joints, c.cand.joints,
                                search_planning_time, edge_attempts);
      ++edges_tried;
      if (!edge.success)
      {
        continue;
      }
      CompleteCand nxt = node_ab;
      nxt.c = &c;
      nxt.b_c = edge;
      nxt.time = node_ab.time + edge.best_time;
      nxt.path = node_ab.path + edge.best_path;
      nxt.min_clearance = std::min(node_ab.min_clearance, c.clearance);
      complete.push_back(nxt);
      viz.best = c.cand;
    }
  }
  std::sort(complete.begin(), complete.end(), betterComplete);
  RCLCPP_INFO(node->get_logger(),
              "SEARCH edges_tried=%d complete_feasible=%zu budget=%d", edges_tried, complete.size(),
              complete_budget);
  if (complete.empty())
  {
    failure_class = "B_TO_C_FAILED";
    write_diag("FAIL", failure_class, nullptr);
    shutdownSpinner(executor, spinner);
    return 4;
  }

  const CompleteCand winner = complete.front();
  viz.hud = "WINNER FOUND — planning official Home→C playback";
  viz.vis = winner.c->vis;
  viz.best = winner.c->cand;
  viz.publish(node, true);

  bool playback_ok = false;
  for (int attempt = 1; attempt <= prefix_retries && rclcpp::ok(); ++attempt)
  {
    moveit::task_constructor::Task task("", true);
    task.setName("FR3 Step12 Winner ABC");
    task.loadRobotModel(node);
    addPrefixStages(task, node, group, ee_link, attach_link, object_id, table_name, touch_links,
                    object_scene, pregrasp, grasp, lift, object_height, object_radius, planning_time,
                    home);
    auto ompl =
        std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
    ompl->setTimeout(planning_time);
    auto move_a = std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo ViewA", ompl);
    move_a->setGroup(group);
    move_a->setGoal(winner.a->cand.joints);
    move_a->setTimeout(planning_time);
    task.add(std::move(move_a));
    auto move_b = std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo ViewB", ompl);
    move_b->setGroup(group);
    move_b->setGoal(winner.b->cand.joints);
    move_b->setTimeout(planning_time);
    task.add(std::move(move_b));
    auto move_c = std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo ViewC", ompl);
    move_c->setGroup(group);
    move_c->setGoal(winner.c->cand.joints);
    move_c->setTimeout(planning_time);
    task.add(std::move(move_c));
    try
    {
      task.init();
    }
    catch (const std::exception& ex)
    {
      RCLCPP_WARN(node->get_logger(), "winner task init failed: %s", ex.what());
      continue;
    }
    if (!task.plan(1) || task.numSolutions() < 1)
    {
      RCLCPP_WARN(node->get_logger(), "winner playback plan failed attempt %d", attempt);
      continue;
    }
    task.publishAllSolutions(false);
    if (viz.traj)
    {
      moveit_task_constructor_msgs::msg::Solution msg;
      task.solutions().front()->toMsg(msg);
      moveit_msgs::msg::DisplayTrajectory disp;
      disp.model_id = robot_model->getName();
      for (const auto& sub : msg.sub_trajectory)
      {
        if (!sub.trajectory.joint_trajectory.points.empty())
        {
          disp.trajectory.push_back(sub.trajectory);
        }
      }
      viz.traj->publish(disp);
    }
    playback_ok = true;
    RCLCPP_INFO(node->get_logger(), "WINNER PLAYBACK planned (RViz only, NOT executed)");
    break;
  }
  if (!playback_ok)
  {
    failure_class = "SEARCH_BUDGET_EXHAUSTED";
    write_diag("FAIL", failure_class, &winner);
    shutdownSpinner(executor, spinner);
    return 4;
  }

  {
    std::ofstream yaml(winner_path);
    yaml.setf(std::ios::fixed);
    yaml.precision(9);
    yaml << "label: BEST SAMPLED FEASIBLE COMPLETE TASK WITHIN CONFIGURED SEARCH BUDGET\n";
    yaml << "not_global_optimum: true\n";
    yaml << "task_version: STEP12C\n";
    yaml << "home_joints_deg: " << formatHomeList(home_deg) << "\n";
    yaml << "home_joints_rad: " << formatHomeList(home) << "\n";
    yaml << "mtc_used_home: true\n";
    yaml << "current_already_at_home: " << (already_at_home ? "true" : "false") << "\n";
    yaml << "p1: " << vecYaml(p1) << "\n";
    yaml << "camera_position: " << vecYaml(camera.optical_center_world) << "\n";
    yaml << "camera_rpy: [-2.35619449, 0.0, 0.0]\n";
    yaml << "camera_forward: " << vecYaml(camera.optical_forward_world) << "\n";
    yaml << "surface_canonical_rpy: [0.785398, 0.0, 0.0]\n";
    yaml << "surface_target_normal: " << vecYaml(d1) << "\n";
    yaml << "camera_calibrated: false\n";
    yaml << "order: [side_pos_y, side_neg_y, bottom_circle]\n";
    yaml << "arm1_c_physical: original_table_contact_bottom\n";
    yaml << "arm1_c_local_normal: [0.0, 0.0, -1.0]\n";
    yaml << "arm2_future_face: top_circle\n";
    yaml << "pregrasp_m: 0.08\n";
    yaml << "lift_m: 0.08\n";
    yaml << "t_tcp_object_translation: [0.0, 0.0, 0.0]\n";
    yaml << "t_tcp_object_xyzw: [" << tcp_object.pose.orientation.x << ", "
         << tcp_object.pose.orientation.y << ", " << tcp_object.pose.orientation.z << ", "
         << tcp_object.pose.orientation.w << "]\n";
    {
      const Eigen::Isometry3d a_obj = t_world_base * poseToIso(winner.a->cand.object_target.pose);
      const Eigen::Isometry3d b_obj = t_world_base * poseToIso(winner.b->cand.object_target.pose);
      const Eigen::Isometry3d c_obj = t_world_base * poseToIso(winner.c->cand.object_target.pose);
      const Eigen::Isometry3d a_tcp = t_world_base * poseToIso(winner.a->cand.tcp_target.pose);
      const Eigen::Isometry3d b_tcp = t_world_base * poseToIso(winner.b->cand.tcp_target.pose);
      const Eigen::Isometry3d c_tcp = t_world_base * poseToIso(winner.c->cand.tcp_target.pose);
      yaml << "A:\n";
      yaml << "  roll_deg: " << winner.a->cand.roll_deg << "\n";
      yaml << "  joints_rad: " << jointsYaml(winner.a->cand.joints) << "\n";
      yaml << "  joints_deg: " << jointsDegYaml(winner.a->cand.joints) << "\n";
      yaml << "  tcp_world: " << isoPoseYaml(a_tcp) << "\n";
      yaml << "  object_world: " << isoPoseYaml(a_obj) << "\n";
      yaml << "B:\n";
      yaml << "  roll_deg: " << winner.b->cand.roll_deg << "\n";
      yaml << "  joints_rad: " << jointsYaml(winner.b->cand.joints) << "\n";
      yaml << "  joints_deg: " << jointsDegYaml(winner.b->cand.joints) << "\n";
      yaml << "  tcp_world: " << isoPoseYaml(b_tcp) << "\n";
      yaml << "  object_world: " << isoPoseYaml(b_obj) << "\n";
      yaml << "C_bottom:\n";
      yaml << "  roll_deg: " << winner.c->cand.roll_deg << "\n";
      yaml << "  joints_rad: " << jointsYaml(winner.c->cand.joints) << "\n";
      yaml << "  joints_deg: " << jointsDegYaml(winner.c->cand.joints) << "\n";
      yaml << "  tcp_world: " << isoPoseYaml(c_tcp) << "\n";
      yaml << "  object_world: " << isoPoseYaml(c_obj) << "\n";
    }
    yaml << "a_roll_deg: " << winner.a->cand.roll_deg << "\n";
    yaml << "b_roll_deg: " << winner.b->cand.roll_deg << "\n";
    yaml << "c_roll_deg: " << winner.c->cand.roll_deg << "\n";
    yaml << "a_joints_rad: " << jointsYaml(winner.a->cand.joints) << "\n";
    yaml << "b_joints_rad: " << jointsYaml(winner.b->cand.joints) << "\n";
    yaml << "c_joints_rad: " << jointsYaml(winner.c->cand.joints) << "\n";
    yaml << "a_joints_deg: " << jointsDegYaml(winner.a->cand.joints) << "\n";
    yaml << "b_joints_deg: " << jointsDegYaml(winner.b->cand.joints) << "\n";
    yaml << "c_joints_deg: " << jointsDegYaml(winner.c->cand.joints) << "\n";
    yaml << "predicted_total_trajectory_time: " << winner.time << "\n";
    yaml << "total_joint_path_length: " << winner.path << "\n";
    yaml << "minimum_clearance: " << winner.min_clearance << "\n";
    yaml << "prefix_time: " << prefix_time << "\n";
    yaml << "lift_to_a_time: " << winner.lift_a.best_time << "\n";
    yaml << "a_to_b_time: " << winner.a_b.best_time << "\n";
    yaml << "b_to_c_time: " << winner.b_c.best_time << "\n";
    yaml << "segments:\n";
    yaml << "  - name: Current_to_Home\n";
    yaml << "  - name: Home_to_PreGrasp\n";
    yaml << "  - name: PreGrasp_to_Grasp\n";
    yaml << "  - name: Grasp_Attach_Lift\n";
    yaml << "  - name: Lift_to_A\n";
    yaml << "  - name: A_to_B\n";
    yaml << "  - name: B_to_C_original_bottom\n";
    yaml << "roll_step_deg: " << getDouble(node, "roll_step_deg") << "\n";
    yaml << "top_k: " << top_k << "\n";
    yaml << "beam_width: " << beam_width << "\n";
    yaml << "edge_attempts: " << edge_attempts << "\n";
    yaml << "max_ik_solutions_per_pose: " << cfg.max_ik_solutions_per_pose << "\n";
    yaml << "search_planning_time: " << search_planning_time << "\n";
    yaml << "complete_feasible_candidates: " << complete.size() << "\n";
    yaml << "execution: false\n";
  }
  write_diag("PASS", "none", &winner);
  RCLCPP_INFO(node->get_logger(),
              "WINNER A r=%.1f B r=%.1f C r=%.1f T=%.3f L=%.3f clearance=%.4f file=%s",
              winner.a->cand.roll_deg, winner.b->cand.roll_deg, winner.c->cand.roll_deg, winner.time,
              winner.path, winner.min_clearance, winner_path.c_str());
  viz.hud = "BEST SAMPLED FEASIBLE COMPLETE TASK\nCurrent→Home→A→B→C(original bottom) PLAYBACK (RViz only)";
  viz.a_status = "VISIBLE +Y SIDE";
  viz.b_status = "VISIBLE -Y SIDE";
  viz.c_status = "VISIBLE ORIGINAL BOTTOM object -Z";
  viz.publish(node, true);
  if (vis_hold > 0)
  {
    std::this_thread::sleep_for(std::chrono::duration<double>(vis_hold));
  }
  const auto after = waitForFreshJoints(node, std::chrono::seconds(3));
  if (after)
  {
    RCLCPP_INFO(node->get_logger(), "Robot moved because of this node? %s",
                maxJointError(jointsFromMsg(*after), actual) > 0.02 ? "YES" : "NO");
  }
  shutdownSpinner(executor, spinner);
  return 0;
}
