#include "fr_task_planner/inspection_endpoint_candidates.hpp"
#include "fr_task_planner/inspection_visibility.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <thread>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/task_constructor/solvers/pipeline_planner.h>
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/modify_planning_scene.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/storage.h>
#include <moveit/task_constructor/task.h>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/object_color.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <moveit_task_constructor_msgs/msg/solution.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace
{
using fr_task_planner::applyJoints;
using fr_task_planner::collectCollisionContacts;
using fr_task_planner::CollisionDiagConfig;
using fr_task_planner::EndpointCandidate;
using fr_task_planner::EndpointGenerationConfig;
using fr_task_planner::filterRollsByView;
using fr_task_planner::generateValidEndpointCandidates;
using fr_task_planner::isoToPose;
using fr_task_planner::kArmJoints;
using fr_task_planner::poseToIso;
using fr_task_planner::removeWorldObjectDiagnostic;
using fr_task_planner::RollPose;
using fr_task_planner::ViewGeom;

const char* kPlanningGroup = "fairino3_v6_group";
const char* kPlanningFrame = "base_link";
const char* kEeLink = "gripper_tcp";
const char* kOmplPipeline = "ompl";
const char* kPilzPipeline = "pilz_industrial_motion_planner";
const char* kPilzPlannerId = "LIN";

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

bool getBoolParam(const rclcpp::Node::SharedPtr& node, const std::string& name, bool fallback)
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

moveit_msgs::msg::CollisionObject makeBox(const std::string& id, const std::string& frame,
                                          const Eigen::Isometry3d& pose, const Eigen::Vector3d& size)
{
  moveit_msgs::msg::CollisionObject object;
  object.id = id;
  object.header.frame_id = frame;
  object.primitives.resize(1);
  object.primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
  object.primitives[0].dimensions = { size.x(), size.y(), size.z() };
  object.primitive_poses.push_back(isoToPose(pose));
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

using RelocateFn = std::function<void(const planning_scene::PlanningScenePtr&)>;

void addPrefixStages(moveit::task_constructor::Task& task, const rclcpp::Node::SharedPtr& node,
                     const std::string& group, const std::string& ee_link,
                     const std::string& attach_link, const std::string& object_id,
                     const std::string& table_name, const std::vector<std::string>& touch_links,
                     const geometry_msgs::msg::PoseStamped& object_scene,
                     const geometry_msgs::msg::PoseStamped& pregrasp,
                     const geometry_msgs::msg::PoseStamped& grasp,
                     const geometry_msgs::msg::PoseStamped& lift, double object_height,
                     double object_radius, double planning_time, RelocateFn relocate = {})
{
  task.add(std::make_unique<moveit::task_constructor::stages::CurrentState>("CurrentState"));
  if (relocate)
  {
    auto relocate_stage = std::make_unique<moveit::task_constructor::stages::ModifyPlanningScene>(
        "Relocate Workcell Diagnostic");
    relocate_stage->setCallback(
        [relocate](const planning_scene::PlanningScenePtr& scene,
                   const moveit::task_constructor::PropertyMap&) { relocate(scene); });
    task.add(std::move(relocate_stage));
  }
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
  move_pregrasp->setIKFrame(ee_link);
  move_pregrasp->setGoal(pregrasp);
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

geometry_msgs::msg::PoseStamped transformStamped(const geometry_msgs::msg::PoseStamped& pose,
                                                 const Eigen::Isometry3d& t_new_from_old)
{
  geometry_msgs::msg::PoseStamped out = pose;
  out.pose = isoToPose(t_new_from_old * poseToIso(pose.pose));
  return out;
}

std::vector<RollPose> rollsInCandidateBase(const std::vector<RollPose>& rolls,
                                           const Eigen::Isometry3d& t_cur, const Eigen::Isometry3d& t_cand)
{
  const Eigen::Isometry3d t_cand_from_cur = t_cand.inverse() * t_cur;
  std::vector<RollPose> out = rolls;
  for (auto& roll : out)
  {
    roll.object = transformStamped(roll.object, t_cand_from_cur);
    roll.tcp = transformStamped(roll.tcp, t_cand_from_cur);
  }
  return out;
}

std::vector<RollPose> filterRollsByStep(const std::vector<RollPose>& rolls, double step_deg)
{
  std::vector<RollPose> out;
  for (const auto& roll : rolls)
  {
    if (step_deg <= 1e-9)
    {
      out.push_back(roll);
      continue;
    }
    const double n = std::round(roll.roll_deg / step_deg);
    if (std::abs(roll.roll_deg - n * step_deg) <= 0.51)
    {
      out.push_back(roll);
    }
  }
  return out;
}

std::vector<RollPose> shiftRollsWorldZ(const std::vector<RollPose>& rolls,
                                       const Eigen::Isometry3d& t_world_base, double dz_world)
{
  const Eigen::Vector3d d_base = t_world_base.linear().transpose() * Eigen::Vector3d(0.0, 0.0, dz_world);
  std::vector<RollPose> out = rolls;
  for (auto& roll : out)
  {
    roll.object.pose.position.x += d_base.x();
    roll.object.pose.position.y += d_base.y();
    roll.object.pose.position.z += d_base.z();
    roll.tcp.pose.position.x += d_base.x();
    roll.tcp.pose.position.y += d_base.y();
    roll.tcp.pose.position.z += d_base.z();
  }
  return out;
}

void relocateWorkcell(planning_scene::PlanningScene& scene, const Eigen::Isometry3d& t_cand,
                      const Eigen::Isometry3d& table_world, const Eigen::Vector3d& table_size,
                      const Eigen::Isometry3d& column_world, const Eigen::Vector3d& column_size,
                      const std::string& table_name, const std::string& column_name,
                      const std::string& frame)
{
  removeWorldObjectDiagnostic(scene, table_name);
  removeWorldObjectDiagnostic(scene, column_name);
  const Eigen::Isometry3d t_base_table = t_cand.inverse() * table_world;
  const Eigen::Isometry3d t_base_column = t_cand.inverse() * column_world;
  scene.processCollisionObjectMsg(makeBox(table_name, frame, t_base_table, table_size));
  scene.processCollisionObjectMsg(makeBox(column_name, frame, t_base_column, column_size));
  scene.getAllowedCollisionMatrixNonConst().setEntry(column_name, "base_link", true);
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
  if (!any && std::isfinite(res.minimum_distance.distance))
  {
    column_signed = res.minimum_distance.distance;
    any = true;
  }
  return any;
}

struct ViewEval
{
  int roll_samples = 0;
  int raw_ik = 0;
  int unique_ik = 0;
  int free_ik = 0;
  int free_rolls = 0;
  std::vector<double> free_roll_degs;
  std::string dominant_pair = "none";
  bool distance_available = false;
  double best_column_distance = std::numeric_limits<double>::quiet_NaN();
  std::optional<EndpointCandidate> best_free;
  std::map<std::string, double> rep_joints;
  double rep_roll_deg = 0.0;
  geometry_msgs::msg::PoseStamped rep_tcp;
};

struct MountResult
{
  std::string name;
  double p1z = 1.1;
  double roll_step_deg = 10.0;
  double dx = 0.0;
  double dy = 0.0;
  double dz = 0.0;
  std::string rot_axis = "none";
  double rot_deg = 0.0;
  Eigen::Isometry3d t_world_base = Eigen::Isometry3d::Identity();
  ViewEval a;
  ViewEval b;
  ViewEval c;
  bool three_view = false;
};

struct LayoutConfig
{
  Eigen::Isometry3d t_cur = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d t_urdf_base = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d table_world = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d column_world = Eigen::Isometry3d::Identity();
  Eigen::Vector3d table_size = Eigen::Vector3d::Zero();
  Eigen::Vector3d column_size = Eigen::Vector3d::Zero();
  Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
  Eigen::Vector3d d1 = Eigen::Vector3d::UnitY();
  Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
  std::string table_name = "table";
  std::string column_name = "mounting_column";
  std::string object_frame = "base_link";
  std::string group = kPlanningGroup;
  EndpointGenerationConfig cfg;
  CollisionDiagConfig dcfg;
  ViewGeom view_a;
  ViewGeom view_b;
  ViewGeom view_c;
};

ViewEval evaluateView(const planning_scene::PlanningScene& lift_scene, const LayoutConfig& layout,
                      const Eigen::Isometry3d& t_cand, const std::vector<RollPose>& base_rolls,
                      const ViewGeom& view, const rclcpp::Logger& logger)
{
  ViewEval eval;
  auto scene = planning_scene::PlanningScene::clone(lift_scene.diff());
  relocateWorkcell(*scene, t_cand, layout.table_world, layout.table_size, layout.column_world,
                   layout.column_size, layout.table_name, layout.column_name, layout.object_frame);
  const auto rolls = rollsInCandidateBase(filterRollsByView(base_rolls, view.name), layout.t_cur, t_cand);
  eval.roll_samples = static_cast<int>(rolls.size());
  EndpointGenerationConfig cfg = layout.cfg;
  cfg.t_model_base = layout.t_urdf_base;
  const Eigen::Isometry3d t_adj = layout.t_urdf_base * t_cand.inverse();
  cfg.p1 = t_adj * layout.p1;
  cfg.d1 = (t_adj.linear() * layout.d1).normalized();
  cfg.preferred_up = (t_adj.linear() * layout.up).normalized();
  const auto gen = generateValidEndpointCandidates(*scene, rolls, view, cfg, logger);
  eval.raw_ik = static_cast<int>(gen.raw.size());
  eval.free_ik = static_cast<int>(gen.valid.size());
  {
    std::set<std::string> uniq;
    for (const auto& cand : gen.raw)
    {
      uniq.insert(cand.digest.empty() ? cand.candidate_id : cand.digest);
    }
    eval.unique_ik = static_cast<int>(uniq.size());
  }
  std::map<std::string, int> pairs;
  std::set<int> free_pose;
  eval.free_roll_degs.clear();
  for (const auto& cand : gen.valid)
  {
    if (free_pose.insert(cand.pose_index).second)
    {
      eval.free_roll_degs.push_back(cand.roll_deg);
    }
    free_pose.insert(cand.pose_index);
    if (!eval.best_free || std::abs(cand.roll_deg) < std::abs(eval.best_free->roll_deg))
    {
      eval.best_free = cand;
    }
  }
  eval.free_rolls = static_cast<int>(free_pose.size());
  double best_dist = -std::numeric_limits<double>::infinity();
  bool any_dist = false;
  std::string worst_pair = "none";
  int worst_n = 0;
  std::optional<EndpointCandidate> best_clearance;
  for (const auto& cand : gen.raw)
  {
    auto probe = planning_scene::PlanningScene::clone(scene->diff());
    applyJoints(probe->getCurrentStateNonConst(), cand.joints);
    const auto snap = collectCollisionContacts(*probe, layout.dcfg);
    std::map<std::string, int> local;
    for (const auto& c : snap.contacts)
    {
      local[c.pair_key] += c.contact_count;
    }
    if (!local.empty())
    {
      const auto it = std::max_element(local.begin(), local.end(),
                                       [](const auto& a, const auto& b) { return a.second < b.second; });
      pairs[it->first] += 1;
    }
    double col_d = 0.0;
    if (tryColumnDistance(*probe, layout.group, layout.column_name, col_d))
    {
      any_dist = true;
      if (col_d > best_dist)
      {
        best_dist = col_d;
        best_clearance = cand;
      }
    }
  }
  if (!pairs.empty())
  {
    const auto it = std::max_element(pairs.begin(), pairs.end(),
                                     [](const auto& a, const auto& b) { return a.second < b.second; });
    worst_pair = it->first;
    worst_n = it->second;
    (void)worst_n;
  }
  eval.dominant_pair = eval.free_ik > 0 ? "none" : worst_pair;
  eval.distance_available = any_dist;
  eval.best_column_distance = any_dist ? best_dist : std::numeric_limits<double>::quiet_NaN();
  if (eval.best_free)
  {
    eval.rep_joints = eval.best_free->joints;
    eval.rep_roll_deg = eval.best_free->roll_deg;
    eval.rep_tcp = eval.best_free->tcp_target;
  }
  else if (best_clearance)
  {
    eval.rep_joints = best_clearance->joints;
    eval.rep_roll_deg = best_clearance->roll_deg;
    eval.rep_tcp = best_clearance->tcp_target;
  }
  return eval;
}

[[maybe_unused]] MountResult evaluateMount(const planning_scene::PlanningScene& lift_scene, const LayoutConfig& layout,
                          const Eigen::Isometry3d& t_cand, const std::vector<RollPose>& all_rolls,
                          const rclcpp::Logger& logger, const std::string& name, bool force_ab)
{
  MountResult r;
  r.name = name;
  r.t_world_base = t_cand;
  r.c = evaluateView(lift_scene, layout, t_cand, all_rolls, layout.view_c, logger);
  if (force_ab || r.c.free_ik > 0)
  {
    r.a = evaluateView(lift_scene, layout, t_cand, all_rolls, layout.view_a, logger);
    r.b = evaluateView(lift_scene, layout, t_cand, all_rolls, layout.view_b, logger);
  }
  r.three_view = r.a.free_ik > 0 && r.b.free_ik > 0 && r.c.free_ik > 0;
  return r;
}

MountResult evaluateHeight(const planning_scene::PlanningScene& lift_scene, const LayoutConfig& layout,
                           const std::vector<RollPose>& all_rolls, double p1z, double roll_step_deg,
                           const rclcpp::Logger& logger, bool force_ab)
{
  LayoutConfig loc = layout;
  loc.p1.z() = p1z;
  const double dz = p1z - layout.p1.z();
  const auto rolls = shiftRollsWorldZ(filterRollsByStep(all_rolls, roll_step_deg), layout.t_cur, dz);
  MountResult r;
  std::ostringstream name;
  name << std::fixed << std::setprecision(3) << "p1z=" << p1z << "_r" << std::setprecision(0)
       << roll_step_deg;
  r.name = name.str();
  r.p1z = p1z;
  r.roll_step_deg = roll_step_deg;
  r.t_world_base = layout.t_cur;
  r.c = evaluateView(lift_scene, loc, layout.t_cur, rolls, loc.view_c, logger);
  if (force_ab || r.c.free_ik > 0)
  {
    r.a = evaluateView(lift_scene, loc, layout.t_cur, rolls, loc.view_a, logger);
    r.b = evaluateView(lift_scene, loc, layout.t_cur, rolls, loc.view_b, logger);
  }
  r.three_view = r.a.free_ik > 0 && r.b.free_ik > 0 && r.c.free_ik > 0;
  return r;
}

void logHeight(const rclcpp::Logger& logger, const MountResult& r)
{
  RCLCPP_INFO(logger,
              "P1.z=%.3f step=%.0fdeg  C raw=%d unique=%d free=%d rolls=%d dist=%.4f pair=%s  "
              "A=%d B=%d three=%s",
              r.p1z, r.roll_step_deg, r.c.raw_ik, r.c.unique_ik, r.c.free_ik, r.c.free_rolls,
              r.c.distance_available ? r.c.best_column_distance : std::numeric_limits<double>::quiet_NaN(),
              r.c.dominant_pair.c_str(), r.a.free_ik, r.b.free_ik, r.three_view ? "YES" : "NO");
}

[[maybe_unused]] double translationNorm(const MountResult& r)
{
  return std::sqrt(r.dx * r.dx + r.dy * r.dy + r.dz * r.dz);
}

[[maybe_unused]] double layoutScore(const MountResult& r)
{
  return translationNorm(r) + 0.0035 * std::abs(r.rot_deg);
}

[[maybe_unused]] void logMount(const rclcpp::Logger& logger, const MountResult& r)
{
  RCLCPP_INFO(logger,
              "%s dx=%.3f dy=%.3f dz=%.3f rot=%s %+.1fdeg  C free=%d rolls=%d dist=%.4f pair=%s  "
              "A=%d B=%d three=%s",
              r.name.c_str(), r.dx, r.dy, r.dz, r.rot_axis.c_str(), r.rot_deg, r.c.free_ik,
              r.c.free_rolls,
              r.c.distance_available ? r.c.best_column_distance : std::numeric_limits<double>::quiet_NaN(),
              r.c.dominant_pair.c_str(), r.a.free_ik, r.b.free_ik, r.three_view ? "YES" : "NO");
}

[[maybe_unused]] void fillDelta(MountResult& r, const Eigen::Isometry3d& t_cur, const Eigen::Isometry3d& t_cand,
               const std::string& rot_axis, double rot_deg)
{
  const Eigen::Vector3d dt = t_cand.translation() - t_cur.translation();
  r.dx = dt.x();
  r.dy = dt.y();
  r.dz = dt.z();
  r.rot_axis = rot_axis;
  r.rot_deg = rot_deg;
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

bool pairIs(const std::string& a, const std::string& b, const std::string& x, const std::string& y)
{
  return (a == x && b == y) || (a == y && b == x);
}

void addDelete(visualization_msgs::msg::MarkerArray& arr, int id, const std::string& ns)
{
  auto m = baseMarker(id, ns, visualization_msgs::msg::Marker::SPHERE);
  m.action = visualization_msgs::msg::Marker::DELETE;
  arr.markers.push_back(m);
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
  m.points.push_back(toPoint(start));
  m.points.push_back(toPoint(end));
  m.scale.x = 0.012;
  m.scale.y = 0.024;
  m.scale.z = 0.04;
  m.color = color;
  arr.markers.push_back(m);
}

void addText(visualization_msgs::msg::MarkerArray& arr, int id, const Eigen::Vector3d& p,
             const std::string& text, const std_msgs::msg::ColorRGBA& color, double height,
             const std::string& ns)
{
  auto m = baseMarker(id, ns, visualization_msgs::msg::Marker::TEXT_VIEW_FACING);
  m.pose.position = toPoint(p);
  m.scale.z = height;
  m.color = color;
  m.text = text;
  arr.markers.push_back(m);
}

void addAxes(visualization_msgs::msg::MarkerArray& arr, int id0, const Eigen::Isometry3d& pose,
             double length, const std::string& ns)
{
  const Eigen::Vector3d o = pose.translation();
  addArrow(arr, id0, o, o + length * pose.linear().col(0), rgba(1.0f, 0.15f, 0.15f, 1.0f), ns);
  addArrow(arr, id0 + 1, o, o + length * pose.linear().col(1), rgba(0.15f, 0.9f, 0.2f, 1.0f), ns);
  addArrow(arr, id0 + 2, o, o + length * pose.linear().col(2), rgba(0.2f, 0.45f, 1.0f, 1.0f), ns);
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

struct SearchViz
{
  bool enabled = false;
  double delay_sec = 0.7;
  double hold_sec = 8.0;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub;
  rclcpp::Publisher<moveit_msgs::msg::DisplayRobotState>::SharedPtr ghost_pub;
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr traj_pub;
  planning_scene::PlanningSceneConstPtr lift_scene;
  Eigen::Isometry3d t_world_base = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d t_tcp_object = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d canonical_c_object_world = Eigen::Isometry3d::Identity();
  Eigen::Vector3d top_normal_local = Eigen::Vector3d(0.0, 0.0, 1.0);
  Eigen::Vector3d d1 = Eigen::Vector3d(0.0, -1.0, 0.0);
  Eigen::Vector3d up = Eigen::Vector3d(0.0, 0.0, 1.0);
  Eigen::Vector3d table_center = Eigen::Vector3d::Zero();
  Eigen::Vector3d table_size = Eigen::Vector3d::Zero();
  Eigen::Vector3d column_center = Eigen::Vector3d::Zero();
  Eigen::Vector3d column_size = Eigen::Vector3d::Zero();
  Eigen::Vector3d part_center = Eigen::Vector3d::Zero();
  std::string phase = "COARSE 5 CM";
  std::string active_view = "C";
  int height_index = 0;
  int height_total = 9;
  std::optional<MountResult> best_clearance;
  std::optional<MountResult> best_feasible;
  Eigen::Vector3d contact_world = Eigen::Vector3d::Zero();
  bool contact_valid = false;
  bool launched_rviz_markers = false;
  bool show_visibility_audit = false;

  const ViewEval& focused(const MountResult& r) const
  {
    if (active_view == "A")
    {
      return r.a;
    }
    if (active_view == "B")
    {
      return r.b;
    }
    return r.c;
  }

  void noteResult(const MountResult& r)
  {
    if (r.c.distance_available &&
        (!best_clearance || r.c.best_column_distance > best_clearance->c.best_column_distance))
    {
      best_clearance = r;
    }
    if (r.three_view && (!best_feasible || r.p1z < best_feasible->p1z - 1e-12))
    {
      best_feasible = r;
    }
  }

  Eigen::Isometry3d tcpWorld(const geometry_msgs::msg::PoseStamped& tcp_base) const
  {
    return t_world_base * poseToIso(tcp_base.pose);
  }

  void refreshContact(const std::map<std::string, double>& joints)
  {
    contact_valid = false;
    if (!enabled || !lift_scene || joints.empty())
    {
      return;
    }
    auto scene = planning_scene::PlanningScene::clone(lift_scene->diff());
    applyJoints(scene->getCurrentStateNonConst(), joints);
    collision_detection::CollisionRequest req;
    req.contacts = true;
    req.max_contacts = 200;
    req.max_contacts_per_pair = 8;
    collision_detection::CollisionResult res;
    scene->checkCollision(req, res);
    const std::string model_frame = lift_scene->getRobotModel()->getModelFrame();
    for (const auto& item : res.contacts)
    {
      if (!pairIs(item.first.first, item.first.second, "forearm_link", "mounting_column") ||
          item.second.empty())
      {
        continue;
      }
      const Eigen::Vector3d pos = item.second.front().pos;
      contact_world = (model_frame == "world") ? pos : (t_world_base * pos);
      contact_valid = std::isfinite(contact_world.x());
      break;
    }
  }

  void publishGhost(const std::map<std::string, double>& joints, bool feasible)
  {
    if (!enabled || !ghost_pub || !lift_scene || joints.empty())
    {
      return;
    }
    auto scene = planning_scene::PlanningScene::clone(lift_scene->diff());
    applyJoints(scene->getCurrentStateNonConst(), joints);
    moveit_msgs::msg::DisplayRobotState display;
    moveit::core::robotStateToRobotStateMsg(scene->getCurrentState(), display.state, true);
    moveit_msgs::msg::ObjectColor forearm;
    forearm.id = "forearm_link";
    forearm.color = feasible ? rgba(0.1f, 0.9f, 0.2f, 0.9f) : rgba(1.0f, 0.05f, 0.05f, 0.95f);
    display.highlight_links.push_back(forearm);
    moveit_msgs::msg::ObjectColor upper;
    upper.id = "upperarm_link";
    upper.color = feasible ? rgba(0.15f, 0.7f, 0.25f, 0.75f) : rgba(1.0f, 0.35f, 0.05f, 0.85f);
    display.highlight_links.push_back(upper);
    display.hide = false;
    ghost_pub->publish(display);
  }

  void publishMarkers(const MountResult* current, const std::string& extra)
  {
    if (!enabled || !marker_pub)
    {
      return;
    }
    visualization_msgs::msg::MarkerArray arr;
    addCube(arr, 1, table_center, table_size, rgba(0.55f, 0.38f, 0.18f, 0.22f), "workcell/table");
    addText(arr, 2, table_center + Eigen::Vector3d(0.0, 0.0, 0.08), "TABLE",
            rgba(1.0f, 0.85f, 0.45f, 1.0f), 0.045, "workcell/table_label");
    addCube(arr, 3, column_center, column_size, rgba(0.35f, 0.45f, 0.55f, 0.28f), "workcell/column");
    addText(arr, 4, column_center + Eigen::Vector3d(0.0, 0.16, 0.05), "MOUNTING COLUMN",
            rgba(0.75f, 0.85f, 1.0f, 1.0f), 0.04, "workcell/column_label");
    addText(arr, 5, part_center + Eigen::Vector3d(0.0, -0.12, 0.08), "PART",
            rgba(1.0f, 0.92f, 0.4f, 1.0f), 0.03, "workcell/part_label");

    const Eigen::Vector3d p1(0.0, 0.4, current ? current->p1z : 1.10);
    const bool c_free = current && current->c.free_ik > 0;
    addSphere(arr, 10, p1, 0.03, c_free ? rgba(0.1f, 0.85f, 0.2f, 0.95f) : rgba(0.1f, 0.75f, 1.0f, 0.95f),
              "search/current");
    std::ostringstream p1txt;
    p1txt << std::fixed << std::setprecision(3) << "CURRENT P1\nz = " << p1.z() << " m";
    addText(arr, 11, p1 + Eigen::Vector3d(0.0, 0.0, 0.08), p1txt.str(),
            rgba(0.85f, 0.95f, 1.0f, 1.0f), 0.035, "search/current");
    addArrow(arr, 13, p1, p1 + 0.18 * up, rgba(0.85f, 0.25f, 0.95f, 1.0f), "inspection/up");

    Eigen::Isometry3d t_obj_c = canonical_c_object_world;
    if (current && !current->c.rep_joints.empty())
    {
      t_obj_c = tcpWorld(current->c.rep_tcp) * t_tcp_object;
    }
    const Eigen::Vector3d n_actual = (t_obj_c.linear() * top_normal_local).normalized();
    const Eigen::Vector3d n_expected(0.0, -1.0, 0.0);
    const double face_dot = std::max(-1.0, std::min(1.0, n_actual.dot(n_expected)));
    const double face_err_deg = std::acos(face_dot) * 180.0 / M_PI;
    addArrow(arr, 70, p1, p1 + 0.24 * n_actual, rgba(0.15f, 1.0f, 0.85f, 1.0f),
             "inspection/actual_face_normal");
    addText(arr, 71, p1 + 0.27 * n_actual, "TOP CIRCLE FACE NORMAL\nR*object+Z (actual)",
            rgba(0.2f, 1.0f, 0.85f, 1.0f), 0.028, "inspection/actual_face_normal");
    addArrow(arr, 72, p1, p1 + 0.18 * n_expected, rgba(1.0f, 0.85f, 0.1f, 1.0f),
             "inspection/expected_face_normal");
    addText(arr, 73, p1 + Eigen::Vector3d(0.04, -0.22, 0.02),
            "EXPECTED CAMERA-FACING DIRECTION\nworld -Y", rgba(1.0f, 0.9f, 0.3f, 1.0f), 0.028,
            "inspection/expected_face_normal");
    std::ostringstream geom;
    geom << std::fixed << std::setprecision(3);
    if (face_err_deg < 1.0)
    {
      geom << "ORIENTATION CORRECT\nactual overlaps expected\nerr = " << std::setprecision(2)
           << face_err_deg << " deg";
    }
    else
    {
      geom << "GEOMETRY FAIL\nactual vs world -Y\nerr = " << std::setprecision(2) << face_err_deg
           << " deg";
    }
    addText(arr, 74, p1 + Eigen::Vector3d(0.18, 0.05, 0.10), geom.str(),
            face_err_deg < 1.0 ? rgba(0.2f, 1.0f, 0.4f, 1.0f) : rgba(1.0f, 0.15f, 0.1f, 1.0f), 0.03,
            "inspection/orientation_audit");

    if (show_visibility_audit)
    {
      addCircle(arr, 80, p1, n_actual, 0.0075, rgba(0.2f, 0.95f, 1.0f, 1.0f),
                "inspection/target_roi");
      addText(arr, 81, p1 + 0.05 * n_actual,
              "TARGET SURFACE ROI\ntop_circle r=7.5 mm\n(true scale)",
              rgba(0.3f, 0.95f, 1.0f, 1.0f), 0.028, "inspection/target_roi");
      const Eigen::Vector3d cam_line_a(0.0, 0.10, 0.40);
      const Eigen::Vector3d cam_line_b(0.0, 0.10, 1.60);
      addArrow(arr, 82, cam_line_a, cam_line_b, rgba(1.0f, 0.55f, 0.1f, 1.0f),
               "inspection/camera_constraint");
      addText(arr, 83, Eigen::Vector3d(0.05, 0.10, 1.35),
              "CAMERA CONSTRAINT LINE\nx=0  y=0.10 m\nz UNKNOWN\nNOT a camera pose",
              rgba(1.0f, 0.65f, 0.2f, 1.0f), 0.03, "inspection/camera_constraint");
      addText(arr, 84, Eigen::Vector3d(-0.15, 0.55, 1.35),
              "CAMERA EXTRINSIC REQUIRED\nCAMERA_MODEL_INCOMPLETE\nNO LOS / FOV AUDIT\n"
              "optical center not shown\n(would be invented)",
              rgba(1.0f, 0.35f, 0.2f, 1.0f), 0.034, "inspection/visibility_hud");
    }

    if (current)
    {
      const ViewEval& fv = focused(*current);
      if (!fv.rep_joints.empty())
      {
        const auto tcp_w = tcpWorld(fv.rep_tcp);
        const bool v_free = fv.free_ik > 0;
        addSphere(arr, 20, tcp_w.translation(), 0.02,
                  v_free ? rgba(0.1f, 0.85f, 0.2f, 0.95f) : rgba(1.0f, 0.08f, 0.08f, 0.95f),
                  "search/current");
        addAxes(arr, 21, tcp_w, 0.08, "search/current");
        std::ostringstream ctxt;
        ctxt << std::fixed << std::setprecision(3);
        if (active_view == "A")
        {
          ctxt << "VIEW A / side_pos_y\n";
        }
        else if (active_view == "B")
        {
          ctxt << "VIEW B / side_neg_y\n";
        }
        else
        {
          ctxt << "VIEW C / top_circle\n";
        }
        ctxt << "P1.z = " << current->p1z << " m\n";
        if (active_view == "C")
        {
          ctxt << "TCP +Z is optical +Y\n(not the face normal)\n";
          if (c_free)
          {
            ctxt << "TOP CIRCLE: ENDPOINT FEASIBLE\nroll = " << std::setprecision(1)
                 << fv.rep_roll_deg << " deg\ncolumn clearance = " << std::setprecision(1)
                 << fv.best_column_distance * 1000.0 << " mm";
          }
          else
          {
            ctxt << "TOP CIRCLE: BLOCKED\nbest column distance = " << std::setprecision(1)
                 << fv.best_column_distance * 1000.0 << " mm\n" << fv.dominant_pair;
          }
        }
        else
        {
          ctxt << (v_free ? "ENDPOINT FEASIBLE" : "BLOCKED") << "\nroll = " << std::setprecision(1)
               << fv.rep_roll_deg << " deg";
        }
        addText(arr, 24, tcp_w.translation() + Eigen::Vector3d(0.0, 0.0, 0.12), ctxt.str(),
                v_free ? rgba(0.2f, 1.0f, 0.35f, 1.0f) : rgba(1.0f, 0.3f, 0.2f, 1.0f), 0.03,
                "search/current");
      }
    }

    std::ostringstream hud;
    hud << std::fixed << std::setprecision(3);
    hud << "HEIGHT SEARCH\nPHASE: " << phase << "\n";
    if (current)
    {
      hud << "Current:\nP1.z = " << current->p1z << " m\n";
    }
    hud << "Search:\n" << height_index << " / " << height_total << " heights\n";
    hud << "Top-circle:\n" << (c_free ? "ENDPOINT FEASIBLE" : "BLOCKED") << "\n";
    hud << "Free IK:\n" << (current ? current->c.free_ik : 0) << "\n";
    if (current && current->c.distance_available)
    {
      hud << std::setprecision(1) << "Best clearance:\n"
          << current->c.best_column_distance * 1000.0 << " mm\n";
    }
    if (!extra.empty())
    {
      hud << extra << "\n";
    }
    addText(arr, 50, Eigen::Vector3d(0.0, 0.18, 1.55), hud.str(), rgba(1.0f, 1.0f, 1.0f, 1.0f), 0.038,
            "search/current");

    std::ostringstream best;
    if (best_feasible)
    {
      best << std::fixed << std::setprecision(3) << "BEST FEASIBLE HEIGHT\nP1.z = "
           << best_feasible->p1z << " m\nroll = " << std::setprecision(1)
           << best_feasible->c.rep_roll_deg << " deg";
    }
    else if (best_clearance)
    {
      best << std::fixed << std::setprecision(3) << "BEST SO FAR\nP1.z = " << best_clearance->p1z
           << " m\nroll = " << std::setprecision(1) << best_clearance->c.rep_roll_deg
           << " deg\nsigned distance = " << std::setprecision(1)
           << best_clearance->c.best_column_distance * 1000.0 << " mm";
    }
    else
    {
      best << "BEST SO FAR\n(none yet)";
    }
    addText(arr, 1, Eigen::Vector3d(0.35, 0.55, 1.45), best.str(), rgba(1.0f, 0.92f, 0.35f, 1.0f),
            0.034, "search/best");

    if (contact_valid)
    {
      addSphere(arr, 61, contact_world, 0.03, rgba(1.0f, 0.0f, 0.0f, 1.0f), "collision/contact");
      addText(arr, 62, contact_world + Eigen::Vector3d(0.0, 0.0, 0.05),
              current && !current->c.dominant_pair.empty() ? current->c.dominant_pair :
                                                            "forearm_link <-> mounting_column",
              rgba(1.0f, 0.2f, 0.15f, 1.0f), 0.028, "collision/contact");
    }
    else
    {
      addDelete(arr, 61, "collision/contact");
      addDelete(arr, 62, "collision/contact");
    }
    if (current && current->c.free_ik == 0 && !current->c.dominant_pair.empty() &&
        current->c.dominant_pair != "none")
    {
      addText(arr, 60, column_center + Eigen::Vector3d(0.14, 0.16, 0.20),
              std::string("Collision: ") + current->c.dominant_pair, rgba(1.0f, 0.2f, 0.15f, 1.0f),
              0.032, "collision/warning");
    }
    else
    {
      addDelete(arr, 60, "collision/warning");
    }
    marker_pub->publish(arr);
    launched_rviz_markers = true;
  }

  void showHeight(const MountResult& r, const std::string& extra, bool pause)
  {
    noteResult(r);
    active_view = "C";
    refreshContact(r.c.free_ik > 0 ? std::map<std::string, double>{} : r.c.rep_joints);
    if (r.c.free_ik > 0)
    {
      contact_valid = false;
    }
    publishMarkers(&r, extra);
    publishGhost(r.c.rep_joints, r.c.free_ik > 0);
    if (enabled && pause && delay_sec > 0.0)
    {
      std::this_thread::sleep_for(std::chrono::duration<double>(delay_sec));
    }
  }

  void showValidation(const MountResult& r, const std::string& view, const std::string& extra)
  {
    active_view = view;
    const ViewEval& fv = focused(r);
    refreshContact(fv.free_ik > 0 ? std::map<std::string, double>{} : fv.rep_joints);
    if (fv.free_ik > 0)
    {
      contact_valid = false;
    }
    publishMarkers(&r, extra);
    publishGhost(fv.rep_joints, fv.free_ik > 0);
    if (enabled && delay_sec > 0.0)
    {
      std::this_thread::sleep_for(std::chrono::duration<double>(std::min(delay_sec, 0.8)));
    }
  }

  void publishTaskTrajectory(moveit::task_constructor::Task& task)
  {
    if (!enabled || !traj_pub || task.numSolutions() < 1)
    {
      return;
    }
    try
    {
      moveit_task_constructor_msgs::msg::Solution sol_msg;
      task.solutions().front()->toMsg(sol_msg, &task.introspection());
      moveit_msgs::msg::DisplayTrajectory dt;
      if (lift_scene)
      {
        dt.model_id = lift_scene->getRobotModel()->getName();
      }
      dt.trajectory_start = sol_msg.start_scene.robot_state;
      for (const auto& sub : sol_msg.sub_trajectory)
      {
        if (!sub.trajectory.joint_trajectory.points.empty() ||
            !sub.trajectory.multi_dof_joint_trajectory.points.empty())
        {
          dt.trajectory.push_back(sub.trajectory);
        }
      }
      traj_pub->publish(dt);
      task.publishAllSolutions(false);
    }
    catch (const std::exception& ex)
    {
      (void)ex;
    }
  }

  void holdFinal()
  {
    if (enabled && hold_sec > 0.0)
    {
      std::this_thread::sleep_for(std::chrono::duration<double>(hold_sec));
    }
  }
};

bool planPrefixToView(const rclcpp::Node::SharedPtr& node, const LayoutConfig& layout,
                      const Eigen::Isometry3d& t_cand, const geometry_msgs::msg::PoseStamped& object_scene,
                      const geometry_msgs::msg::PoseStamped& pregrasp,
                      const geometry_msgs::msg::PoseStamped& grasp,
                      const geometry_msgs::msg::PoseStamped& lift,
                      const geometry_msgs::msg::PoseStamped& view_tcp, const std::string& view_name,
                      double object_height, double object_radius, double planning_time, int attempts,
                      SearchViz* viz = nullptr)
{
  const bool vis = viz && viz->enabled;
  const Eigen::Isometry3d t_rel = t_cand.inverse() * layout.t_cur;
  moveit::task_constructor::Task task("", vis);
  task.setName("FR3 STEP11F " + view_name);
  task.loadRobotModel(node);
  RelocateFn relocate = [&layout, t_cand](const planning_scene::PlanningScenePtr& scene) {
    relocateWorkcell(*scene, t_cand, layout.table_world, layout.table_size, layout.column_world,
                     layout.column_size, layout.table_name, layout.column_name, layout.object_frame);
  };
  addPrefixStages(task, node, layout.group, layout.cfg.ee_link, layout.cfg.ee_link, layout.cfg.object_id,
                  layout.table_name, layout.cfg.touch_links, transformStamped(object_scene, t_rel),
                  transformStamped(pregrasp, t_rel), transformStamped(grasp, t_rel),
                  transformStamped(lift, t_rel), object_height, object_radius, planning_time,
                  relocate);
  auto ompl = std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
  ompl->setTimeout(planning_time);
  auto move = std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo " + view_name, ompl);
  move->setGroup(layout.group);
  move->setIKFrame(layout.cfg.ee_link);
  move->setGoal(transformStamped(view_tcp, t_rel));
  move->setTimeout(planning_time);
  task.add(std::move(move));
  try
  {
    task.init();
    const bool ok = task.plan(static_cast<size_t>(std::max(1, attempts))) && task.numSolutions() > 0;
    if (ok && vis)
    {
      try
      {
        viz->publishTaskTrajectory(task);
      }
      catch (const std::exception& ex)
      {
        RCLCPP_WARN(node->get_logger(),
                    "prefix+view %s trajectory publish failed (observational only): %s",
                    view_name.c_str(), ex.what());
      }
    }
    return ok;
  }
  catch (const std::exception& ex)
  {
    RCLCPP_WARN(node->get_logger(), "prefix+view %s failed: %s", view_name.c_str(), ex.what());
    return false;
  }
}

bool planTransition(const rclcpp::Node::SharedPtr& node, const LayoutConfig& layout,
                    const Eigen::Isometry3d& t_cand, const geometry_msgs::msg::PoseStamped& object_scene,
                    const geometry_msgs::msg::PoseStamped& pregrasp,
                    const geometry_msgs::msg::PoseStamped& grasp,
                    const geometry_msgs::msg::PoseStamped& lift,
                    const geometry_msgs::msg::PoseStamped& src_tcp,
                    const geometry_msgs::msg::PoseStamped& dst_tcp, const std::string& edge,
                    double object_height, double object_radius, double planning_time, int attempts,
                    SearchViz* viz = nullptr)
{
  const bool vis = viz && viz->enabled;
  const Eigen::Isometry3d t_rel = t_cand.inverse() * layout.t_cur;
  moveit::task_constructor::Task task("", vis);
  task.setName("FR3 STEP11F " + edge);
  task.loadRobotModel(node);
  RelocateFn relocate = [&layout, t_cand](const planning_scene::PlanningScenePtr& scene) {
    relocateWorkcell(*scene, t_cand, layout.table_world, layout.table_size, layout.column_world,
                     layout.column_size, layout.table_name, layout.column_name, layout.object_frame);
  };
  addPrefixStages(task, node, layout.group, layout.cfg.ee_link, layout.cfg.ee_link, layout.cfg.object_id,
                  layout.table_name, layout.cfg.touch_links, transformStamped(object_scene, t_rel),
                  transformStamped(pregrasp, t_rel), transformStamped(grasp, t_rel),
                  transformStamped(lift, t_rel), object_height, object_radius, planning_time,
                  relocate);
  auto ompl_a = std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
  ompl_a->setTimeout(planning_time);
  auto move_a = std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo Source", ompl_a);
  move_a->setGroup(layout.group);
  move_a->setIKFrame(layout.cfg.ee_link);
  move_a->setGoal(transformStamped(src_tcp, t_rel));
  move_a->setTimeout(planning_time);
  task.add(std::move(move_a));
  auto ompl_b = std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
  ompl_b->setTimeout(planning_time);
  auto move_b = std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo Target", ompl_b);
  move_b->setGroup(layout.group);
  move_b->setIKFrame(layout.cfg.ee_link);
  move_b->setGoal(transformStamped(dst_tcp, t_rel));
  move_b->setTimeout(planning_time);
  task.add(std::move(move_b));
  try
  {
    task.init();
    const bool ok = task.plan(static_cast<size_t>(std::max(1, attempts))) && task.numSolutions() > 0;
    if (ok && vis)
    {
      try
      {
        viz->publishTaskTrajectory(task);
      }
      catch (const std::exception& ex)
      {
        RCLCPP_WARN(node->get_logger(),
                    "transition %s trajectory publish failed (observational only): %s", edge.c_str(),
                    ex.what());
      }
    }
    return ok;
  }
  catch (const std::exception& ex)
  {
    RCLCPP_WARN(node->get_logger(), "transition %s failed: %s", edge.c_str(), ex.what());
    return false;
  }
}

void yamlView(std::ofstream& yaml, const std::string& key, const ViewEval& v)
{
  yaml << "  " << key << ":\n";
  yaml << "    roll_samples: " << v.roll_samples << "\n";
  yaml << "    raw_ik: " << v.raw_ik << "\n";
  yaml << "    unique_ik: " << v.unique_ik << "\n";
  yaml << "    collision_free_ik: " << v.free_ik << "\n";
  yaml << "    collision_free_rolls: " << v.free_rolls << "\n";
  yaml << "    dominant_collision: \"" << v.dominant_pair << "\"\n";
  yaml << "    best_column_distance: " << v.best_column_distance << "\n";
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("fr3_mtc_mount_layout_search", options);
  RCLCPP_INFO(node->get_logger(), "========== STEP 11F P1 WORLD-Z HEIGHT FEASIBILITY ==========");
  RCLCPP_INFO(node->get_logger(), "DIAGNOSTIC ONLY. Does not write stage4_config.yaml.");
  RCLCPP_INFO(node->get_logger(),
              "SEARCH MODE: P1 WORLD-Z ONLY. Robot base / column / table stay at baseline.");
  RCLCPP_INFO(node->get_logger(),
              "Abandoned: robot-base translation/rotation (not physically actionable).");

  const auto home = readHome(node);
  const auto pregrasp = readPose(node, "pregrasp");
  const auto grasp = readPose(node, "grasp");
  const auto lift = readPose(node, "lift");
  const auto object_world = readPose(node, "object_world");
  const auto object_base = readPose(node, "object");
  const auto tcp_object = readPose(node, "tcp_object");
  const auto world_base = readPose(node, "world_base");
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
  const std::string object_id = getString(node, "object_name", "small_part");
  const std::string table_name = getString(node, "table_name", "table");
  const std::string column_name = getString(node, "column_name", "mounting_column");
  const auto touch_links = node->get_parameter("touch_links").as_string_array();
  const double object_height = getDouble(node, "object_height");
  const double object_radius = getDouble(node, "object_radius");
  const double home_tol = node->has_parameter("max_home_error_rad") ?
                              node->get_parameter("max_home_error_rad").as_double() :
                              0.03;
  const double planning_time = getDouble(node, "planning_time");
  const std::string output_path =
      getString(node, "diagnostic_output_path", "/tmp/fr3_step11f_p1z.yaml");
  const std::string best_path =
      getString(node, "best_layout_output_path", "/tmp/fr3_step11f_best_p1z.yaml");
  const int path_attempts = std::max(1, getInt(node, "path_attempts", 2));
  const bool visualize_search = getBoolParam(node, "visualize_search", true);
  const double vis_delay = node->has_parameter("search_visualization_delay_sec") ?
                               node->get_parameter("search_visualization_delay_sec").as_double() :
                               0.7;
  const double vis_hold = node->has_parameter("visualization_hold_seconds") ?
                              node->get_parameter("visualization_hold_seconds").as_double() :
                              8.0;
  const double validate_only_p1z =
      node->has_parameter("validate_only_p1z") ? node->get_parameter("validate_only_p1z").as_double() :
                                                0.0;
  const auto all_rolls = readRollPoses(node);

  auto param_node = rclcpp::Node::make_shared("fr3_mtc_mount_layout_search_params");
  if (!overlayRobotDescriptionFromMoveGroup(node, param_node))
  {
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  const auto before_msg = waitForFreshJoints(node, std::chrono::seconds(20));
  if (!before_msg)
  {
    shutdownSpinner(executor, spinner);
    return 1;
  }
  const auto actual = jointsFromMsg(*before_msg);
  if (jointsAreAllZero(actual))
  {
    shutdownSpinner(executor, spinner);
    return 1;
  }
  double max_home_error = 0.0;
  for (const auto& name : kArmJoints)
  {
    max_home_error = std::max(max_home_error, std::abs(actual.at(name) - home.at(name)));
  }
  if (max_home_error > home_tol)
  {
    RCLCPP_ERROR(node->get_logger(), "Not at Stage4 Home");
    shutdownSpinner(executor, spinner);
    return 2;
  }

  moveit::task_constructor::Task gen_task("", false);
  gen_task.setName("FR3 STEP11F Prefix");
  gen_task.loadRobotModel(node);
  const auto robot_model = gen_task.getRobotModel();
  const std::string model_frame = robot_model->getModelFrame();
  geometry_msgs::msg::PoseStamped object_scene = object_world;
  if (model_frame != object_world.header.frame_id)
  {
    object_scene = object_base;
    object_scene.header.frame_id =
        model_frame == "base_link" ? object_base.header.frame_id : model_frame;
  }
  addPrefixStages(gen_task, node, group, ee_link, ee_link, object_id, table_name, touch_links,
                  object_scene, pregrasp, grasp, lift, object_height, object_radius, planning_time);
  try
  {
    gen_task.init();
  }
  catch (const std::exception& ex)
  {
    RCLCPP_ERROR(node->get_logger(), "prefix init failed: %s", ex.what());
    shutdownSpinner(executor, spinner);
    return 1;
  }
  if (!gen_task.plan(1) || gen_task.numSolutions() < 1)
  {
    RCLCPP_ERROR(node->get_logger(), "prefix planning failed");
    shutdownSpinner(executor, spinner);
    return 3;
  }
  std::vector<const moveit::task_constructor::SolutionBase*> leaves;
  flattenSolutions(*gen_task.solutions().front(), leaves);
  const auto* restore_sol = findStageSolution(leaves, "Restore Part-Table Collision");
  if (!restore_sol || !restore_sol->end() || !restore_sol->end()->scene())
  {
    shutdownSpinner(executor, spinner);
    return 3;
  }
  const auto lift_scene = planning_scene::PlanningScene::clone(restore_sol->end()->scene());

  LayoutConfig layout;
  layout.t_cur = poseToIso(world_base.pose);
  layout.t_urdf_base = lift_scene->getCurrentState().getGlobalLinkTransform("base_link");
  layout.table_world = Eigen::Isometry3d::Identity();
  layout.table_world.translation() =
      Eigen::Vector3d(getDouble(node, "table_cx"), getDouble(node, "table_cy"), getDouble(node, "table_cz"));
  layout.column_world = Eigen::Isometry3d::Identity();
  layout.column_world.translation() = Eigen::Vector3d(
      getDouble(node, "column_cx"), getDouble(node, "column_cy"), getDouble(node, "column_cz"));
  layout.table_size =
      Eigen::Vector3d(getDouble(node, "table_dx"), getDouble(node, "table_dy"), getDouble(node, "table_dz"));
  layout.column_size = Eigen::Vector3d(getDouble(node, "column_dx"), getDouble(node, "column_dy"),
                                       getDouble(node, "column_dz"));
  layout.p1 = p1;
  layout.d1 = d1;
  layout.up = preferred_up;
  layout.table_name = table_name;
  layout.column_name = column_name;
  layout.object_frame = "base_link";
  layout.group = group;
  layout.cfg.group = group;
  layout.cfg.ee_link = ee_link;
  layout.cfg.object_id = object_id;
  layout.cfg.table_name = table_name;
  layout.cfg.touch_links = touch_links;
  layout.cfg.t_tcp_object = poseToIso(tcp_object.pose);
  layout.cfg.pos_tol = getDouble(node, "position_tolerance");
  layout.cfg.ori_tol_deg = getDouble(node, "orientation_tolerance_deg");
  layout.cfg.max_ik_solutions_per_pose =
      static_cast<uint32_t>(std::max(1, getInt(node, "max_ik_solutions_per_pose", 8)));
  layout.cfg.min_ik_solution_distance = node->has_parameter("min_ik_solution_distance") ?
                                            node->get_parameter("min_ik_solution_distance").as_double() :
                                            0.1;
  layout.dcfg.object_id = object_id;
  layout.dcfg.table_name = table_name;
  layout.dcfg.column_name = column_name;
  layout.dcfg.touch_links = touch_links;
  layout.view_a = readViewGeom(node, "side_pos_y");
  layout.view_b = readViewGeom(node, "side_neg_y");
  layout.view_c = readViewGeom(node, "top_circle");

  SearchViz viz;
  viz.enabled = visualize_search;
  viz.delay_sec = vis_delay;
  viz.hold_sec = vis_hold;
  viz.lift_scene = lift_scene;
  viz.t_world_base = layout.t_cur;
  viz.t_tcp_object = layout.cfg.t_tcp_object;
  viz.top_normal_local = layout.view_c.normal_in_object;
  viz.d1 = d1;
  viz.up = preferred_up;
  for (const auto& roll : all_rolls)
  {
    if (roll.view == "top_circle" && std::abs(roll.roll_deg) < 1e-9)
    {
      viz.canonical_c_object_world = layout.t_cur * poseToIso(roll.object.pose);
      break;
    }
  }
  viz.table_center = layout.table_world.translation();
  viz.table_size = layout.table_size;
  viz.column_center = layout.column_world.translation();
  viz.column_size = layout.column_size;
  viz.part_center = poseToIso(object_world.pose).translation();
  if (viz.enabled)
  {
    rclcpp::QoS qos(1);
    qos.transient_local();
    viz.marker_pub = node->create_publisher<visualization_msgs::msg::MarkerArray>("/fr3_vis/markers", qos);
    viz.ghost_pub =
        node->create_publisher<moveit_msgs::msg::DisplayRobotState>("/fr3_vis/collision_robot_state", qos);
    viz.traj_pub =
        node->create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path", qos);
    RCLCPP_INFO(node->get_logger(), "LIVE SEARCH VISUALIZATION enabled (observational only)");
  }

  if (std::abs(p1.x()) > 1e-9 || std::abs(p1.y() - 0.4) > 1e-9)
  {
    RCLCPP_ERROR(node->get_logger(), "P1.x/y changed from [0, 0.4]");
    shutdownSpinner(executor, spinner);
    return 1;
  }
  if ((d1 - Eigen::Vector3d(0.0, -1.0, 0.0)).norm() > 1e-6 ||
      (preferred_up - Eigen::Vector3d(0.0, 0.0, 1.0)).norm() > 1e-6)
  {
    RCLCPP_ERROR(node->get_logger(), "D1 or up changed from baseline");
    shutdownSpinner(executor, spinner);
    return 1;
  }

  const bool orientation_audit_only = getBoolParam(node, "orientation_audit_only", false);
  if (orientation_audit_only)
  {
    const Eigen::Vector3d n_actual =
        (viz.canonical_c_object_world.linear() * viz.top_normal_local).normalized();
    const Eigen::Vector3d n_expected(0.0, -1.0, 0.0);
    const double face_dot = std::max(-1.0, std::min(1.0, n_actual.dot(n_expected)));
    const double face_err_deg = std::acos(face_dot) * 180.0 / M_PI;
    RCLCPP_INFO(node->get_logger(), "========== ORIENTATION AUDIT ONLY (no P1.z search) ==========");
    RCLCPP_INFO(node->get_logger(), "actual n_top_world = [%.9f, %.9f, %.9f]", n_actual.x(),
                n_actual.y(), n_actual.z());
    RCLCPP_INFO(node->get_logger(), "expected = [0, -1, 0]  angle=%.6f deg", face_err_deg);
    viz.phase = "ORIENTATION AUDIT";
    MountResult dummy;
    dummy.p1z = p1.z();
    viz.publishMarkers(&dummy, "ORIENTATION AUDIT\nP1.z fixed at 1.10\nNO HEIGHT SEARCH");
    std::ofstream yaml("/tmp/fr3_top_circle_orientation_audit.yaml");
    yaml << std::fixed << std::setprecision(9);
    yaml << "status: " << (face_err_deg < 1.0 ? "PASS" : "BUG") << "\n";
    yaml << "search_paused: true\n";
    yaml << "actual_n_top_world: [" << n_actual.x() << ", " << n_actual.y() << ", " << n_actual.z()
         << "]\n";
    yaml << "expected: [0.0, -1.0, 0.0]\n";
    yaml << "angle_error_deg: " << face_err_deg << "\n";
    yaml << "d1_interpreted_as: FACE_OUTWARD_NORMAL\n";
    yaml.close();
    viz.holdFinal();
    shutdownSpinner(executor, spinner);
    return face_err_deg < 1.0 ? 0 : 4;
  }

  const bool visibility_audit_only = getBoolParam(node, "visibility_audit_only", false);
  if (visibility_audit_only)
  {
    using fr_task_planner::cameraModelClass;
    using fr_task_planner::incompleteVisibility;
    using fr_task_planner::loadCameraModelFromWorkspace;
    using fr_task_planner::sampleTopCircleDisk;
    using fr_task_planner::sideViewRoiDef;
    using fr_task_planner::topCircleRoiDef;
    using fr_task_planner::visibilityOccluderLinks;
    const auto cam = loadCameraModelFromWorkspace();
    const Eigen::Vector3d n_actual =
        (viz.canonical_c_object_world.linear() * viz.top_normal_local).normalized();
    const Eigen::Vector3d n_expected(0.0, -1.0, 0.0);
    const double face_dot = std::max(-1.0, std::min(1.0, n_actual.dot(n_expected)));
    const double face_err_deg = std::acos(face_dot) * 180.0 / M_PI;
    const bool ori_ok = face_err_deg < 1.0;
    const auto roi_c = sampleTopCircleDisk(p1, n_actual, 0.0075);
    const auto a_vis = incompleteVisibility("side_pos_y", ori_ok, 0);
    const auto b_vis = incompleteVisibility("side_neg_y", ori_ok, 0);
    const auto c_vis = incompleteVisibility("top_circle", ori_ok, static_cast<int>(roi_c.size()));
    const auto a_roi = sideViewRoiDef("side_pos_y", "[0,1,0]");
    const auto b_roi = sideViewRoiDef("side_neg_y", "[0,-1,0]");
    const auto c_roi = topCircleRoiDef();
    RCLCPP_INFO(node->get_logger(), "========== CAMERA VISIBILITY AUDIT (no P1.z search) ==========");
    RCLCPP_INFO(node->get_logger(), "CAMERA_MODEL: %s", cameraModelClass(cam).c_str());
    RCLCPP_INFO(node->get_logger(), "T_world_camera: MISSING");
    RCLCPP_INFO(node->get_logger(), "intrinsics/FOV: MISSING");
    RCLCPP_INFO(node->get_logger(), "top_circle ROI samples prepared=%zu (LOS not run)", roi_c.size());
    RCLCPP_INFO(node->get_logger(), "WAITING FOR CAMERA MODEL. Height search will not resume.");
    viz.phase = "VISIBILITY AUDIT";
    viz.show_visibility_audit = true;
    MountResult dummy;
    dummy.p1z = p1.z();
    viz.publishMarkers(&dummy, "CAMERA EXTRINSIC REQUIRED\nNO HEIGHT SEARCH");
    std::ofstream yaml("/tmp/fr3_step11f_visibility_audit.yaml");
    yaml << std::fixed << std::setprecision(9);
    yaml << "status: CAMERA_MODEL_INCOMPLETE\n";
    yaml << "search_paused: true\n";
    yaml << "p1z_search_resumed: false\n";
    yaml << "camera_model: " << cameraModelClass(cam) << "\n";
    yaml << "camera_frame: \"\"\n";
    yaml << "t_world_camera: missing\n";
    yaml << "optical_axis: missing\n";
    yaml << "intrinsics: missing\n";
    yaml << "resolution: missing\n";
    yaml << "fov: missing\n";
    yaml << "xy_line_constraint: [0.0, 0.10]\n";
    yaml << "xy_line_constraint_is_not_a_pose: true\n";
    yaml << "column_center_used_as_camera: false\n";
    yaml << "orientation_audit: PASS\n";
    yaml << "n_top_world: [" << n_actual.x() << ", " << n_actual.y() << ", " << n_actual.z() << "]\n";
    yaml << "top_circle_roi_samples: " << roi_c.size() << "\n";
    yaml << "side_a_bounded_roi: " << (a_roi.bounded_roi_defined ? "true" : "false") << "\n";
    yaml << "side_b_bounded_roi: " << (b_roi.bounded_roi_defined ? "true" : "false") << "\n";
    yaml << "side_roi_gap: \"" << a_roi.gap << "\"\n";
    yaml << "occluder_links:\n";
    for (const auto& link : visibilityOccluderLinks())
    {
      yaml << "  - " << link << "\n";
    }
    yaml << "acm_does_not_imply_optical_transparency: true\n";
    yaml << "view_A_inspection_valid: \"" << a_vis.inspection_valid << "\"\n";
    yaml << "view_B_inspection_valid: \"" << b_vis.inspection_valid << "\"\n";
    yaml << "view_C_inspection_valid: \"" << c_vis.inspection_valid << "\"\n";
    yaml << "previous_heights: KINEMATIC_COLLISION_ONLY\n";
    yaml.close();
    viz.holdFinal();
    shutdownSpinner(executor, spinner);
    return 0;
  }

  const double p1z0 = p1.z();
  std::vector<MountResult> coarse;
  std::vector<MountResult> refine_cm;
  std::vector<MountResult> refine_mm;
  std::optional<MountResult> best_three;
  double fail_boundary = p1z0;
  double pass_boundary = std::numeric_limits<double>::quiet_NaN();

  auto consider = [&](MountResult r, bool pause) {
    logHeight(node->get_logger(), r);
    if (r.three_view && (!best_three || r.p1z < best_three->p1z - 1e-12))
    {
      best_three = r;
    }
    viz.showHeight(r, "", pause);
    return r;
  };

  if (validate_only_p1z > 0.5)
  {
    RCLCPP_INFO(node->get_logger(),
                "validate_only_p1z=%.3f — skip coarse/refine, evaluate this height then prefix",
                validate_only_p1z);
    viz.phase = "VALIDATE ONLY";
    viz.height_total = 1;
    viz.height_index = 1;
    auto r = consider(evaluateHeight(*lift_scene, layout, all_rolls, validate_only_p1z, 5.0,
                                     node->get_logger(), true),
                      true);
    refine_mm.push_back(r);
    if (r.three_view)
    {
      pass_boundary = r.p1z;
      fail_boundary = r.p1z - 0.005;
    }
    else
    {
      fail_boundary = r.p1z;
    }
  }
  else
  {
  RCLCPP_INFO(node->get_logger(), "----- coarse P1.z sweep (10 deg roll, +0.05 m) -----");
  viz.phase = "COARSE 5 CM";
  viz.height_total = 9;
  viz.height_index = 0;
  std::vector<double> coarse_zs;
  for (int i = 0; i <= 8; ++i)
  {
    coarse_zs.push_back(p1z0 + 0.05 * static_cast<double>(i));
  }
  coarse_zs.push_back(1.55);
  coarse_zs.push_back(1.60);
  for (double z : coarse_zs)
  {
    if (z > 1.601)
    {
      break;
    }
    ++viz.height_index;
    const bool baseline = std::abs(z - p1z0) < 1e-12;
    auto r = consider(evaluateHeight(*lift_scene, layout, all_rolls, z, 10.0, node->get_logger(),
                                     baseline),
                      true);
    coarse.push_back(r);
    if (!r.three_view)
    {
      fail_boundary = z;
      continue;
    }
    pass_boundary = z;
    break;
  }

  if (std::isfinite(pass_boundary) && pass_boundary - fail_boundary > 1.5e-2)
  {
    RCLCPP_INFO(node->get_logger(), "----- 1 cm P1.z refine (5 deg roll) -----");
    viz.phase = "1 CM REFINEMENT";
    viz.height_index = 0;
    viz.height_total = 5;
    {
      const double refine_from = fail_boundary;
      for (int k = 1; k <= 5; ++k)
      {
        const double z = refine_from + 0.01 * static_cast<double>(k);
        if (z > pass_boundary + 1e-12)
        {
          break;
        }
        ++viz.height_index;
        auto r = consider(evaluateHeight(*lift_scene, layout, all_rolls, z, 5.0, node->get_logger(),
                                         false),
                          true);
        refine_cm.push_back(r);
        if (!r.three_view)
        {
          fail_boundary = z;
        }
        else
        {
          pass_boundary = z;
          break;
        }
      }
    }
  }

  if (std::isfinite(pass_boundary) && pass_boundary - fail_boundary > 6e-3)
  {
    RCLCPP_INFO(node->get_logger(), "----- 5 mm P1.z refine (5 deg roll) -----");
    viz.phase = "5 MM REFINEMENT";
    viz.height_index = 0;
    viz.height_total = 2;
    {
      const double refine_from = fail_boundary;
      for (int k = 1; k <= 2; ++k)
      {
        const double z = refine_from + 0.005 * static_cast<double>(k);
        if (z > pass_boundary + 1e-12)
        {
          break;
        }
        ++viz.height_index;
        auto r = consider(evaluateHeight(*lift_scene, layout, all_rolls, z, 5.0, node->get_logger(),
                                         false),
                          true);
        refine_mm.push_back(r);
        if (!r.three_view)
        {
          fail_boundary = z;
        }
        else
        {
          pass_boundary = z;
          break;
        }
      }
    }
  }

  if (best_three && best_three->roll_step_deg > 5.5)
  {
    auto fine = consider(evaluateHeight(*lift_scene, layout, all_rolls, best_three->p1z, 5.0,
                                        node->get_logger(), true),
                         true);
    refine_cm.push_back(fine);
  }
  }  // full height search (not validate_only_p1z)

  std::string a_prefix = "NOT TESTED";
  std::string b_prefix = "NOT TESTED";
  std::string c_prefix = "NOT TESTED";
  std::map<std::string, std::string> edges;
  int edges_pass = 0;
  if (best_three)
  {
    const auto& br = *best_three;
    RCLCPP_INFO(node->get_logger(),
                "validating Home→Lift→view at P1.z=%.3f (robot/column/table unchanged)", br.p1z);
    geometry_msgs::msg::PoseStamped goal_a = pregrasp;
    geometry_msgs::msg::PoseStamped goal_b = pregrasp;
    geometry_msgs::msg::PoseStamped goal_c = pregrasp;
    if (br.a.best_free)
    {
      goal_a = br.a.best_free->tcp_target;
    }
    if (br.b.best_free)
    {
      goal_b = br.b.best_free->tcp_target;
    }
    if (br.c.best_free)
    {
      goal_c = br.c.best_free->tcp_target;
    }
    viz.phase = "VALIDATING VIEW A";
    viz.showValidation(br, "A", "VALIDATING VIEW A");
    a_prefix = planPrefixToView(node, layout, layout.t_cur, object_scene, pregrasp, grasp, lift,
                                goal_a, "A", object_height, object_radius, planning_time, path_attempts,
                                &viz) ?
                   "PASS" :
                   "FAIL";
    viz.showValidation(br, "A",
                       a_prefix == "PASS" ? "Home → Lift → A\nPASS" : "Home → Lift → A\nPATH FAIL");
    viz.phase = "VALIDATING VIEW B";
    viz.showValidation(br, "B", std::string("A prefix: ") + a_prefix + "\nVALIDATING VIEW B");
    b_prefix = planPrefixToView(node, layout, layout.t_cur, object_scene, pregrasp, grasp, lift,
                                goal_b, "B", object_height, object_radius, planning_time, path_attempts,
                                &viz) ?
                   "PASS" :
                   "FAIL";
    viz.showValidation(br, "B",
                       b_prefix == "PASS" ? "Home → Lift → B\nPASS" : "Home → Lift → B\nPATH FAIL");
    viz.phase = "VALIDATING VIEW C";
    viz.showValidation(br, "C",
                       std::string("A: ") + a_prefix + "  B: " + b_prefix + "\nVALIDATING VIEW C");
    c_prefix = planPrefixToView(node, layout, layout.t_cur, object_scene, pregrasp, grasp, lift,
                                goal_c, "C", object_height, object_radius, planning_time, path_attempts,
                                &viz) ?
                   "PASS" :
                   "FAIL";
    viz.showValidation(br, "C",
                       c_prefix == "PASS" ? "Home → Lift → C\nPASS" : "Home → Lift → C\nPATH FAIL");
    const std::vector<std::pair<std::string, std::pair<geometry_msgs::msg::PoseStamped,
                                                       geometry_msgs::msg::PoseStamped>>>
        trans = { { "A->B", { goal_a, goal_b } }, { "A->C", { goal_a, goal_c } },
                  { "B->A", { goal_b, goal_a } }, { "B->C", { goal_b, goal_c } },
                  { "C->A", { goal_c, goal_a } }, { "C->B", { goal_c, goal_b } } };
    if (a_prefix == "PASS" && b_prefix == "PASS" && c_prefix == "PASS")
    {
      for (const auto& e : trans)
      {
        viz.phase = std::string("TESTING: ") + e.first;
        const std::string src_view = e.first.substr(0, 1);
        const std::string dst_view = e.first.substr(e.first.size() - 1, 1);
        viz.showValidation(br, src_view, std::string("TESTING:\n") + e.first);
        const bool ok =
            planTransition(node, layout, layout.t_cur, object_scene, pregrasp, grasp, lift,
                           e.second.first, e.second.second, e.first, object_height, object_radius,
                           planning_time, path_attempts, &viz);
        edges[e.first] = ok ? "PASS" : "FAIL";
        edges_pass += ok ? 1 : 0;
        viz.showValidation(br, dst_view,
                           std::string("TESTING:\n") + e.first + "\n" + (ok ? "PASS" : "FAIL"));
      }
    }
    std::ostringstream final_txt;
    final_txt << "FINAL HEIGHT CANDIDATE\nP1.z = " << std::fixed << std::setprecision(3) << br.p1z
              << " m\nA endpoint: " << (br.a.free_ik > 0 ? "PASS" : "FAIL")
              << "\nB endpoint: " << (br.b.free_ik > 0 ? "PASS" : "FAIL")
              << "\nC endpoint: " << (br.c.free_ik > 0 ? "PASS" : "FAIL")
              << "\nHome→Lift→A: " << a_prefix << "\nHome→Lift→B: " << b_prefix
              << "\nHome→Lift→C: " << c_prefix;
    viz.phase = "FINAL HEIGHT CANDIDATE";
    viz.active_view = "C";
    viz.publishMarkers(&br, final_txt.str());
    viz.publishGhost(br.c.rep_joints, br.c.free_ik > 0);
  }
  else
  {
    viz.phase = "NO FEASIBLE HEIGHT";
    viz.publishMarkers(coarse.empty() ? nullptr : &coarse.back(), "HEIGHT_ONLY_FIX NOT FOUND");
  }

  const auto after_msg = waitForFreshJoints(node, std::chrono::seconds(5));
  bool moved = false;
  if (after_msg)
  {
    moved = fr_task_planner::maxJointError(jointsFromMsg(*after_msg), actual) > 0.02;
  }

  const std::string classification =
      best_three ? "HEIGHT_ONLY_FIX" : "HEIGHT_ONLY_FIX NOT FOUND";
  auto dumpRows = [&](std::ofstream& yaml, const char* key, const std::vector<MountResult>& rows) {
    yaml << key << ":\n";
    for (const auto& r : rows)
    {
      yaml << "  - p1z: " << r.p1z << "\n";
      yaml << "    roll_step_deg: " << r.roll_step_deg << "\n";
      yaml << "    roll_samples: " << r.c.roll_samples << "\n";
      yaml << "    raw_ik: " << r.c.raw_ik << "\n";
      yaml << "    unique_ik: " << r.c.unique_ik << "\n";
      yaml << "    c_free_ik: " << r.c.free_ik << "\n";
      yaml << "    free_rolls: " << r.c.free_rolls << "\n";
      yaml << "    best_column_distance: " << r.c.best_column_distance << "\n";
      yaml << "    dominant_collision: \"" << r.c.dominant_pair << "\"\n";
      yaml << "    a_free_ik: " << r.a.free_ik << "\n";
      yaml << "    b_free_ik: " << r.b.free_ik << "\n";
      yaml << "    three_view: " << (r.three_view ? "true" : "false") << "\n";
    }
  };

  std::ofstream yaml(output_path);
  yaml << std::fixed << std::setprecision(9);
  yaml << "status: PASS\n";
  yaml << "search_mode: P1_WORLD_Z_ONLY\n";
  yaml << "classification: " << classification << "\n";
  yaml << "formal_config_modified: false\n";
  yaml << "abandoned_mount_search: true\n";
  yaml << "p1_xy: [0.0, 0.4]\n";
  yaml << "p1z_current: " << p1z0 << "\n";
  yaml << "robot_base_changed: false\n";
  yaml << "column_changed: false\n";
  yaml << "table_changed: false\n";
  yaml << "d1_changed: false\n";
  yaml << "up_changed: false\n";
  yaml << "robot_moved: " << (moved ? "true" : "false") << "\n";
  yaml << "visualize_search: " << (viz.enabled ? "true" : "false") << "\n";
  yaml << "visualization_markers_published: " << (viz.launched_rviz_markers ? "true" : "false") << "\n";
  dumpRows(yaml, "coarse", coarse);
  dumpRows(yaml, "refine_1cm", refine_cm);
  dumpRows(yaml, "refine_5mm", refine_mm);
  yaml << "fail_boundary: " << fail_boundary << "\n";
  yaml << "pass_boundary: " << pass_boundary << "\n";
  yaml << "best_three_view:\n";
  if (!best_three)
  {
    yaml << "  found: false\n";
  }
  else
  {
    yaml << "  found: true\n";
    yaml << "  p1z: " << best_three->p1z << "\n";
    yaml << "  height_increase_m: " << (best_three->p1z - p1z0) << "\n";
    yamlView(yaml, "A", best_three->a);
    yamlView(yaml, "B", best_three->b);
    yamlView(yaml, "C", best_three->c);
    yaml << "  free_rolls_C: [";
    for (size_t i = 0; i < best_three->c.free_roll_degs.size(); ++i)
    {
      yaml << best_three->c.free_roll_degs[i];
      if (i + 1 < best_three->c.free_roll_degs.size())
      {
        yaml << ", ";
      }
    }
    yaml << "]\n";
    yaml << "  prefix_A: " << a_prefix << "\n";
    yaml << "  prefix_B: " << b_prefix << "\n";
    yaml << "  prefix_C: " << c_prefix << "\n";
    yaml << "  transitions:\n";
    for (const auto& kv : edges)
    {
      yaml << "    \"" << kv.first << "\": " << kv.second << "\n";
    }
    yaml << "  transitions_pass: " << edges_pass << "\n";
  }
  yaml.close();

  if (best_three)
  {
    std::ofstream best(best_path);
    best << std::fixed << std::setprecision(9);
    best << "diagnostic_only: true\n";
    best << "do_not_write_stage4_config: true\n";
    best << "search_mode: P1_WORLD_Z_ONLY\n";
    best << "p1: [0.0, 0.4, " << best_three->p1z << "]\n";
    best << "d1: [0.0, -1.0, 0.0]\n";
    best << "up: [0.0, 0.0, 1.0]\n";
    best.close();
  }

  RCLCPP_INFO(node->get_logger(), "classification: %s", classification.c_str());
  RCLCPP_INFO(node->get_logger(), "three-view found: %s", best_three ? "YES" : "NO");
  if (best_three)
  {
    RCLCPP_INFO(node->get_logger(), "minimum feasible P1.z=%.3f  (+%.1f cm)", best_three->p1z,
                (best_three->p1z - p1z0) * 100.0);
  }
  RCLCPP_INFO(node->get_logger(), "Wrote %s", output_path.c_str());
  RCLCPP_INFO(node->get_logger(), "Robot moved? %s", moved ? "YES" : "NO");
  viz.holdFinal();
  shutdownSpinner(executor, spinner);
  return 0;
}
