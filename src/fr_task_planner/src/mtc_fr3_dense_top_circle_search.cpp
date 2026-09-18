#include "fr_task_planner/inspection_endpoint_candidates.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <thread>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/task_constructor/solvers/pipeline_planner.h>
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/modify_planning_scene.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/storage.h>
#include <moveit/task_constructor/task.h>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

namespace
{
using fr_task_planner::applyJoints;
using fr_task_planner::collectCollisionContacts;
using fr_task_planner::CollisionCategory;
using fr_task_planner::CollisionDiagConfig;
using fr_task_planner::EndpointCandidate;
using fr_task_planner::EndpointGenerationConfig;
using fr_task_planner::filterRollsByView;
using fr_task_planner::generateValidEndpointCandidates;
using fr_task_planner::isoToPose;
using fr_task_planner::jointL2;
using fr_task_planner::jointsFromState;
using fr_task_planner::kArmJoints;
using fr_task_planner::normalizePairKey;
using fr_task_planner::poseToIso;
using fr_task_planner::RollPose;
using fr_task_planner::ViewGeom;

const char* kPlanningGroup = "fairino3_v6_group";
const char* kPlanningFrame = "base_link";
const char* kEeLink = "gripper_tcp";
const char* kOmplPipeline = "ompl";
const char* kPilzPipeline = "pilz_industrial_motion_planner";
const char* kPilzPlannerId = "LIN";
const char* kColumnName = "mounting_column";

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
    RCLCPP_ERROR(target->get_logger(), "Timed out waiting for /move_group.");
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
                     double object_radius, double planning_time)
{
  task.add(std::make_unique<moveit::task_constructor::stages::CurrentState>("CurrentState"));
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

double canonicalizeRoll(double deg)
{
  double x = std::fmod(deg + 180.0, 360.0);
  if (x < 0.0)
  {
    x += 360.0;
  }
  x -= 180.0;
  if (x <= -180.0 + 1e-9)
  {
    return 180.0;
  }
  return x;
}

RollPose makeRolledPose(double roll_deg, const RollPose& canonical, const Eigen::Isometry3d& t_world_base,
                        const Eigen::Vector3d& p1, const Eigen::Vector3d& d1,
                        const Eigen::Vector3d& center_in_object,
                        const Eigen::Isometry3d& t_tcp_object, int pose_index)
{
  const Eigen::Isometry3d t_world_object0 = t_world_base * poseToIso(canonical.object.pose);
  const Eigen::AngleAxisd rot(canonicalizeRoll(roll_deg) * M_PI / 180.0, d1.normalized());
  Eigen::Isometry3d t_world_object = Eigen::Isometry3d::Identity();
  t_world_object.linear() = rot.toRotationMatrix() * t_world_object0.linear();
  t_world_object.translation() = p1 - t_world_object.linear() * center_in_object;
  const Eigen::Isometry3d t_world_tcp = t_world_object * t_tcp_object.inverse();
  const Eigen::Isometry3d t_base_object = t_world_base.inverse() * t_world_object;
  const Eigen::Isometry3d t_base_tcp = t_world_base.inverse() * t_world_tcp;
  RollPose pose;
  pose.view = "top_circle";
  pose.roll_deg = canonicalizeRoll(roll_deg);
  pose.pose_index = pose_index;
  pose.object.header.frame_id = canonical.object.header.frame_id;
  pose.object.pose = isoToPose(t_base_object);
  pose.tcp.header.frame_id = canonical.tcp.header.frame_id;
  pose.tcp.pose = isoToPose(t_base_tcp);
  return pose;
}

struct IkMetric
{
  EndpointCandidate cand;
  std::string dominant_pair = "none";
  CollisionCategory dominant_category = CollisionCategory::OTHER;
  bool forearm_column = false;
  bool upperarm_column = false;
  bool self_collision = false;
  bool penetration_available = false;
  double forearm_column_penetration = 0.0;
  double max_column_penetration = 0.0;
  bool signed_distance_available = false;
  double signed_distance = std::numeric_limits<double>::quiet_NaN();
  double column_signed_distance = std::numeric_limits<double>::quiet_NaN();
};

struct RollStat
{
  double roll_deg = 0.0;
  int raw_ik = 0;
  int unique_ik = 0;
  int free_ik = 0;
  std::string dominant_pair = "none";
  std::vector<IkMetric> unique;
};

bool sameIk(const std::map<std::string, double>& a, const std::map<std::string, double>& b,
            double thresh)
{
  return jointL2(a, b) < thresh;
}

std::vector<EndpointCandidate> uniqueIks(const std::vector<EndpointCandidate>& raw, double thresh)
{
  std::vector<EndpointCandidate> out;
  for (const auto& cand : raw)
  {
    bool dup = false;
    for (const auto& prev : out)
    {
      if (sameIk(prev.joints, cand.joints, thresh))
      {
        dup = true;
        break;
      }
    }
    if (!dup)
    {
      out.push_back(cand);
    }
  }
  return out;
}

bool trySignedDistance(const planning_scene::PlanningScene& scene, const std::string& group,
                       const std::string& column_name, double& min_signed, double& column_signed)
{
  collision_detection::DistanceRequest req;
  req.enable_signed_distance = true;
  req.enable_nearest_points = true;
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
  min_signed = res.minimum_distance.distance;
  column_signed = std::numeric_limits<double>::infinity();
  bool any_column = false;
  for (const auto& item : res.distances)
  {
    const std::string& a = item.first.first;
    const std::string& b = item.first.second;
    if (a != column_name && b != column_name)
    {
      continue;
    }
    for (const auto& rec : item.second)
    {
      if (!std::isfinite(rec.distance))
      {
        continue;
      }
      any_column = true;
      column_signed = std::min(column_signed, rec.distance);
    }
  }
  if (!any_column)
  {
    column_signed = std::numeric_limits<double>::quiet_NaN();
  }
  return std::isfinite(min_signed);
}

IkMetric measureIk(const planning_scene::PlanningScene& lift_scene, const EndpointCandidate& cand,
                   const CollisionDiagConfig& dcfg, const std::string& group, bool try_distance)
{
  IkMetric metric;
  metric.cand = cand;
  auto scene = planning_scene::PlanningScene::clone(lift_scene.diff());
  applyJoints(scene->getCurrentStateNonConst(), cand.joints);
  const auto snap = collectCollisionContacts(*scene, dcfg);
  std::map<std::string, int> pair_count;
  double max_col_pen = 0.0;
  double forearm_pen = 0.0;
  bool forearm_pen_ok = false;
  for (const auto& contact : snap.contacts)
  {
    pair_count[contact.pair_key] += contact.contact_count;
    if (contact.category == CollisionCategory::ROBOT_SELF)
    {
      metric.self_collision = true;
    }
    const bool col = contact.a == dcfg.column_name || contact.b == dcfg.column_name;
    const bool forearm = contact.a == "forearm_link" || contact.b == "forearm_link";
    const bool upper = contact.a == "upperarm_link" || contact.b == "upperarm_link";
    if (col && forearm)
    {
      metric.forearm_column = true;
      if (contact.depth_available)
      {
        forearm_pen = std::max(forearm_pen, std::abs(contact.depth));
        forearm_pen_ok = true;
      }
    }
    if (col && upper)
    {
      metric.upperarm_column = true;
    }
    if (contact.category == CollisionCategory::ROBOT_COLUMN && contact.depth_available)
    {
      max_col_pen = std::max(max_col_pen, std::abs(contact.depth));
      metric.penetration_available = true;
    }
  }
  metric.forearm_column_penetration = forearm_pen;
  metric.max_column_penetration = max_col_pen;
  if (forearm_pen_ok)
  {
    metric.penetration_available = true;
  }
  if (!pair_count.empty())
  {
    metric.dominant_pair =
        std::max_element(pair_count.begin(), pair_count.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; })
            ->first;
    for (const auto& contact : snap.contacts)
    {
      if (contact.pair_key == metric.dominant_pair)
      {
        metric.dominant_category = contact.category;
        break;
      }
    }
  }
  else if (cand.collision_free)
  {
    metric.dominant_pair = "none";
  }
  if (try_distance)
  {
    double min_signed = 0.0;
    double col_signed = 0.0;
    if (trySignedDistance(*scene, group, dcfg.column_name, min_signed, col_signed))
    {
      metric.signed_distance_available = true;
      metric.signed_distance = min_signed;
      metric.column_signed_distance = col_signed;
    }
  }
  return metric;
}

std::string formatJoints(const std::map<std::string, double>& joints)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < kArmJoints.size(); ++i)
  {
    if (i)
    {
      oss << " ";
    }
    oss << kArmJoints[i] << "=" << joints.at(kArmJoints[i]);
  }
  return oss.str();
}

double nearFeasibleScore(const IkMetric& m)
{
  // Smaller is better. Collision-free always wins.
  if (m.cand.valid)
  {
    return -1.0e6;
  }
  if (m.signed_distance_available && std::isfinite(m.column_signed_distance))
  {
    return -m.column_signed_distance;
  }
  if (m.penetration_available)
  {
    if (m.forearm_column)
    {
      return m.forearm_column_penetration;
    }
    return m.max_column_penetration;
  }
  return 1.0;
}

bool betterNearFeasible(const IkMetric& a, const IkMetric& b)
{
  const double sa = nearFeasibleScore(a);
  const double sb = nearFeasibleScore(b);
  if (std::abs(sa - sb) > 1e-9)
  {
    return sa < sb;
  }
  return std::abs(a.cand.roll_deg) < std::abs(b.cand.roll_deg);
}

std::string yamlEscape(const std::string& s)
{
  return s;
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("fr3_mtc_dense_top_circle_search", options);
  RCLCPP_INFO(node->get_logger(),
              "========== STEP 11E DENSE TOP_CIRCLE SEARCH ==========");
  RCLCPP_INFO(node->get_logger(), "PLAN / IK / COLLISION ONLY. No execute. No geometry change.");

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
  const std::string attach_link = getString(node, "attach_link", kEeLink);
  const std::string object_id = getString(node, "object_name", "small_part");
  const std::string table_name = getString(node, "table_name", "table");
  const std::string column_name = getString(node, "column_name", kColumnName);
  const auto touch_links = node->get_parameter("touch_links").as_string_array();
  const double object_height = getDouble(node, "object_height");
  const double object_radius = getDouble(node, "object_radius");
  const double home_tol = node->has_parameter("max_home_error_rad") ?
                              node->get_parameter("max_home_error_rad").as_double() :
                              0.03;
  const double pos_tol = getDouble(node, "position_tolerance");
  const double ori_tol_deg = getDouble(node, "orientation_tolerance_deg");
  const double planning_time = getDouble(node, "planning_time");
  const std::string output_path =
      getString(node, "diagnostic_output_path", "/tmp/fr3_step11e_dense_top_circle.yaml");
  const int independent_sweeps = std::max(1, getInt(node, "independent_sweeps", 3));
  const double phase_b_step = node->has_parameter("phase_b_step_deg") ?
                                  node->get_parameter("phase_b_step_deg").as_double() :
                                  1.0;
  const double phase_b_window = node->has_parameter("phase_b_window_deg") ?
                                    node->get_parameter("phase_b_window_deg").as_double() :
                                    5.0;
  const int path_attempts = std::max(0, getInt(node, "path_attempts", 3));
  const int max_path_candidates = std::max(0, getInt(node, "max_path_candidates", 5));
  const ViewGeom top_view = readViewGeom(node, "top_circle");
  const auto all_rolls = filterRollsByView(readRollPoses(node), "top_circle");

  auto param_node = rclcpp::Node::make_shared("fr3_mtc_dense_top_circle_search_params");
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
    RCLCPP_ERROR(node->get_logger(), "/joint_states missing");
    shutdownSpinner(executor, spinner);
    return 1;
  }
  const auto actual = jointsFromMsg(*before_msg);
  if (jointsAreAllZero(actual))
  {
    RCLCPP_ERROR(node->get_logger(), "CurrentState is all zeros");
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
  gen_task.setName("FR3 STEP11E Prefix");
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
  addPrefixStages(gen_task, node, group, ee_link, attach_link, object_id, table_name, touch_links,
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
  std::vector<const moveit::task_constructor::SolutionBase*> gen_leaves;
  flattenSolutions(*gen_task.solutions().front(), gen_leaves);
  const auto* restore_sol = findStageSolution(gen_leaves, "Restore Part-Table Collision");
  if (!restore_sol || !restore_sol->end() || !restore_sol->end()->scene())
  {
    RCLCPP_ERROR(node->get_logger(), "Restore scene missing");
    shutdownSpinner(executor, spinner);
    return 3;
  }
  const auto lift_scene = planning_scene::PlanningScene::clone(restore_sol->end()->scene());

  EndpointGenerationConfig cfg;
  cfg.group = group;
  cfg.ee_link = ee_link;
  cfg.object_id = object_id;
  cfg.table_name = table_name;
  cfg.touch_links = touch_links;
  cfg.p1 = p1;
  cfg.d1 = d1;
  cfg.preferred_up = preferred_up;
  cfg.t_tcp_object = poseToIso(tcp_object.pose);
  cfg.t_model_base = lift_scene->getCurrentState().getGlobalLinkTransform("base_link");
  cfg.pos_tol = pos_tol;
  cfg.ori_tol_deg = ori_tol_deg;
  cfg.max_ik_solutions_per_pose =
      static_cast<uint32_t>(std::max(1, getInt(node, "max_ik_solutions_per_pose", 16)));
  cfg.min_ik_solution_distance = node->has_parameter("min_ik_solution_distance") ?
                                     node->get_parameter("min_ik_solution_distance").as_double() :
                                     0.1;

  CollisionDiagConfig dcfg;
  dcfg.object_id = object_id;
  dcfg.table_name = table_name;
  dcfg.column_name = column_name;
  dcfg.touch_links = touch_links;

  const Eigen::Isometry3d t_world_base = poseToIso(world_base.pose);
  const RollPose* canonical = nullptr;
  for (const auto& roll : all_rolls)
  {
    if (std::abs(roll.roll_deg) < 1e-9)
    {
      canonical = &roll;
      break;
    }
  }
  if (!canonical)
  {
    RCLCPP_ERROR(node->get_logger(), "canonical roll=0 top_circle pose missing");
    shutdownSpinner(executor, spinner);
    return 1;
  }

  double max_center_err = 0.0;
  double max_normal_err = 0.0;
  for (const auto& roll : all_rolls)
  {
    const Eigen::Isometry3d obj_w = t_world_base * poseToIso(roll.object.pose);
    const Eigen::Vector3d center = obj_w.translation() + obj_w.linear() * top_view.center_in_object;
    const Eigen::Vector3d normal = (obj_w.linear() * top_view.normal_in_object).normalized();
    max_center_err = std::max(max_center_err, (center - p1).norm());
    max_normal_err = std::max(
        max_normal_err,
        std::acos(std::min(1.0, std::max(-1.0, normal.dot(d1)))) * 180.0 / M_PI);
  }
  RCLCPP_INFO(node->get_logger(),
              "Phase A geometry check: max P1 error=%.6f m  max D1 error=%.6f deg  rolls=%zu",
              max_center_err, max_normal_err, all_rolls.size());
  if (max_center_err > 1e-4 || max_normal_err > 0.05)
  {
    RCLCPP_ERROR(node->get_logger(), "HARD geometry failed: P1/D1 not preserved");
    shutdownSpinner(executor, spinner);
    return 4;
  }

  bool distance_api_ok = false;
  {
    auto probe = planning_scene::PlanningScene::clone(lift_scene->diff());
    double min_s = 0.0;
    double col_s = 0.0;
    distance_api_ok = trySignedDistance(*probe, group, column_name, min_s, col_s);
    RCLCPP_INFO(node->get_logger(), "MoveIt signed distance API: %s",
                distance_api_ok ? "YES" : "NO (using contact penetration)");
  }

  struct SweepSummary
  {
    int run = 0;
    int roll_samples = 0;
    int raw_ik = 0;
    int unique_ik = 0;
    int free_ik = 0;
    std::map<double, RollStat> per_roll;
  };
  std::vector<SweepSummary> sweeps;
  std::vector<IkMetric> all_unique;
  std::vector<IkMetric> free_metrics;

  for (int run = 1; run <= independent_sweeps; ++run)
  {
    RCLCPP_INFO(node->get_logger(), "----- Phase A sweep %d / %d -----", run, independent_sweeps);
    const auto gen =
        generateValidEndpointCandidates(*lift_scene, all_rolls, top_view, cfg, node->get_logger());
    SweepSummary sum;
    sum.run = run;
    sum.roll_samples = static_cast<int>(all_rolls.size());
    sum.raw_ik = static_cast<int>(gen.raw.size());
    std::map<double, std::vector<EndpointCandidate>> by_roll;
    for (const auto& cand : gen.raw)
    {
      by_roll[cand.roll_deg].push_back(cand);
    }
    for (const auto& roll : all_rolls)
    {
      RollStat st;
      st.roll_deg = roll.roll_deg;
      const auto& raw = by_roll[roll.roll_deg];
      st.raw_ik = static_cast<int>(raw.size());
      const auto uniq = uniqueIks(raw, cfg.min_ik_solution_distance);
      st.unique_ik = static_cast<int>(uniq.size());
      std::map<std::string, int> pair_freq;
      for (const auto& cand : uniq)
      {
        const auto metric = measureIk(*lift_scene, cand, dcfg, group, distance_api_ok);
        st.unique.push_back(metric);
        all_unique.push_back(metric);
        if (cand.valid)
        {
          ++st.free_ik;
          ++sum.free_ik;
          free_metrics.push_back(metric);
        }
        if (!cand.valid)
        {
          pair_freq[metric.dominant_pair]++;
        }
      }
      if (!pair_freq.empty())
      {
        st.dominant_pair =
            std::max_element(pair_freq.begin(), pair_freq.end(),
                             [](const auto& a, const auto& b) { return a.second < b.second; })
                ->first;
      }
      else if (st.free_ik > 0)
      {
        st.dominant_pair = "none";
      }
      sum.unique_ik += st.unique_ik;
      sum.per_roll[roll.roll_deg] = st;
      RCLCPP_INFO(node->get_logger(),
                  "run%d  %+7.1f deg  raw %d  unique %d  free %d  dominant %s", run, roll.roll_deg,
                  st.raw_ik, st.unique_ik, st.free_ik, st.dominant_pair.c_str());
    }
    RCLCPP_INFO(node->get_logger(), "sweep %d totals: raw=%d unique=%d free=%d", run, sum.raw_ik,
                sum.unique_ik, sum.free_ik);
    sweeps.push_back(std::move(sum));
  }

  std::map<double, RollStat> combined;
  for (const auto& roll : all_rolls)
  {
    RollStat st;
    st.roll_deg = roll.roll_deg;
    std::vector<EndpointCandidate> pooled;
    std::map<std::string, int> pair_freq;
    for (const auto& sweep : sweeps)
    {
      const auto it = sweep.per_roll.find(roll.roll_deg);
      if (it == sweep.per_roll.end())
      {
        continue;
      }
      st.raw_ik += it->second.raw_ik;
      for (const auto& metric : it->second.unique)
      {
        pooled.push_back(metric.cand);
        if (!metric.cand.valid)
        {
          pair_freq[metric.dominant_pair]++;
        }
      }
    }
    const auto uniq = uniqueIks(pooled, cfg.min_ik_solution_distance);
    st.unique_ik = static_cast<int>(uniq.size());
    for (const auto& cand : uniq)
    {
      if (cand.valid)
      {
        ++st.free_ik;
      }
    }
    if (!pair_freq.empty())
    {
      st.dominant_pair =
          std::max_element(pair_freq.begin(), pair_freq.end(),
                           [](const auto& a, const auto& b) { return a.second < b.second; })
              ->first;
    }
    else if (st.free_ik > 0)
    {
      st.dominant_pair = "none";
    }
    combined[roll.roll_deg] = st;
  }

  std::optional<IkMetric> best;
  for (const auto& metric : all_unique)
  {
    if (!best || betterNearFeasible(metric, *best))
    {
      best = metric;
    }
  }

  std::set<double> free_rolls;
  for (const auto& item : combined)
  {
    if (item.second.free_ik > 0)
    {
      free_rolls.insert(item.first);
    }
  }

  std::vector<double> phase_b_centers;
  if (!free_rolls.empty())
  {
    phase_b_centers.assign(free_rolls.begin(), free_rolls.end());
  }
  else if (!all_unique.empty())
  {
    std::vector<IkMetric> ranked = all_unique;
    std::sort(ranked.begin(), ranked.end(), betterNearFeasible);
    std::set<double> centers;
    for (const auto& metric : ranked)
    {
      centers.insert(metric.cand.roll_deg);
      if (centers.size() >= 3)
      {
        break;
      }
    }
    phase_b_centers.assign(centers.begin(), centers.end());
  }

  std::vector<SweepSummary> phase_b_sweeps;
  std::vector<IkMetric> phase_b_unique;
  bool phase_b_done = false;
  std::vector<double> phase_b_angles;
  if (canonical && !phase_b_centers.empty() && phase_b_step > 0.0)
  {
    phase_b_done = true;
    std::set<double> angle_set;
    for (double center : phase_b_centers)
    {
      const int n = static_cast<int>(std::lround(phase_b_window / phase_b_step));
      for (int k = -n; k <= n; ++k)
      {
        angle_set.insert(canonicalizeRoll(center + k * phase_b_step));
      }
    }
    phase_b_angles.assign(angle_set.begin(), angle_set.end());
    std::vector<RollPose> fine_rolls;
    int idx = 1000;
    double fine_center_err = 0.0;
    double fine_normal_err = 0.0;
    for (double ang : phase_b_angles)
    {
      auto pose = makeRolledPose(ang, *canonical, t_world_base, p1, d1, top_view.center_in_object,
                                 cfg.t_tcp_object, idx++);
      const Eigen::Isometry3d obj_w = t_world_base * poseToIso(pose.object.pose);
      const Eigen::Vector3d center = obj_w.translation() + obj_w.linear() * top_view.center_in_object;
      const Eigen::Vector3d normal = (obj_w.linear() * top_view.normal_in_object).normalized();
      fine_center_err = std::max(fine_center_err, (center - p1).norm());
      fine_normal_err = std::max(
          fine_normal_err,
          std::acos(std::min(1.0, std::max(-1.0, normal.dot(d1)))) * 180.0 / M_PI);
      fine_rolls.push_back(pose);
    }
    RCLCPP_INFO(node->get_logger(),
                "Phase B 1 deg local: centers=%zu angles=%zu P1_err=%.6f D1_err=%.6f",
                phase_b_centers.size(), fine_rolls.size(), fine_center_err, fine_normal_err);
    const auto gen =
        generateValidEndpointCandidates(*lift_scene, fine_rolls, top_view, cfg, node->get_logger());
    SweepSummary sum;
    sum.run = 0;
    sum.roll_samples = static_cast<int>(fine_rolls.size());
    sum.raw_ik = static_cast<int>(gen.raw.size());
    std::map<double, std::vector<EndpointCandidate>> by_roll;
    for (const auto& cand : gen.raw)
    {
      by_roll[cand.roll_deg].push_back(cand);
    }
    for (const auto& roll : fine_rolls)
    {
      RollStat st;
      st.roll_deg = roll.roll_deg;
      const auto& raw = by_roll[roll.roll_deg];
      st.raw_ik = static_cast<int>(raw.size());
      const auto uniq = uniqueIks(raw, cfg.min_ik_solution_distance);
      st.unique_ik = static_cast<int>(uniq.size());
      std::map<std::string, int> pair_freq;
      for (const auto& cand : uniq)
      {
        const auto metric = measureIk(*lift_scene, cand, dcfg, group, distance_api_ok);
        st.unique.push_back(metric);
        phase_b_unique.push_back(metric);
        all_unique.push_back(metric);
        if (cand.valid)
        {
          ++st.free_ik;
          ++sum.free_ik;
          free_metrics.push_back(metric);
          free_rolls.insert(cand.roll_deg);
        }
        else
        {
          pair_freq[metric.dominant_pair]++;
        }
      }
      if (!pair_freq.empty())
      {
        st.dominant_pair =
            std::max_element(pair_freq.begin(), pair_freq.end(),
                             [](const auto& a, const auto& b) { return a.second < b.second; })
                ->first;
      }
      else if (st.free_ik > 0)
      {
        st.dominant_pair = "none";
      }
      sum.unique_ik += st.unique_ik;
      sum.per_roll[roll.roll_deg] = st;
      RCLCPP_INFO(node->get_logger(), "phaseB %+7.1f deg  raw %d unique %d free %d  %s",
                  roll.roll_deg, st.raw_ik, st.unique_ik, st.free_ik, st.dominant_pair.c_str());
    }
    phase_b_sweeps.push_back(std::move(sum));
    for (const auto& metric : phase_b_unique)
    {
      if (!best || betterNearFeasible(metric, *best))
      {
        best = metric;
      }
    }
  }

  std::vector<IkMetric> path_candidates;
  if (!free_metrics.empty())
  {
    std::sort(free_metrics.begin(), free_metrics.end(), [](const IkMetric& a, const IkMetric& b) {
      if (std::abs(a.cand.roll_deg) != std::abs(b.cand.roll_deg))
      {
        return std::abs(a.cand.roll_deg) < std::abs(b.cand.roll_deg);
      }
      return a.cand.joint_distance_from_lift < b.cand.joint_distance_from_lift;
    });
    std::vector<IkMetric> picked;
    auto consider = [&](const IkMetric& m) {
      for (const auto& prev : picked)
      {
        if (sameIk(prev.cand.joints, m.cand.joints, cfg.min_ik_solution_distance))
        {
          return;
        }
      }
      picked.push_back(m);
    };
    consider(free_metrics.front());
    auto max_clear = std::min_element(free_metrics.begin(), free_metrics.end(), betterNearFeasible);
    consider(*max_clear);
    auto mild = std::min_element(free_metrics.begin(), free_metrics.end(),
                                 [](const IkMetric& a, const IkMetric& b) {
                                   return a.cand.joint_distance_from_lift <
                                          b.cand.joint_distance_from_lift;
                                 });
    consider(*mild);
    for (const auto& m : free_metrics)
    {
      if (static_cast<int>(picked.size()) >= max_path_candidates)
      {
        break;
      }
      consider(m);
    }
    path_candidates = picked;
  }

  struct PathResult
  {
    double roll_deg = 0.0;
    int ik_index = 0;
    bool planned = false;
    std::string status = "NOT TESTED";
  };
  std::vector<PathResult> path_results;
  bool any_path_ok = false;
  if (!path_candidates.empty())
  {
    RCLCPP_INFO(node->get_logger(), "Path validation for %zu collision-free endpoints",
                path_candidates.size());
    for (const auto& metric : path_candidates)
    {
      PathResult pr;
      pr.roll_deg = metric.cand.roll_deg;
      pr.ik_index = metric.cand.ik_index;
      moveit::task_constructor::Task path_task("", false);
      path_task.setName("FR3 STEP11E LiftToTopCircle");
      path_task.loadRobotModel(node);
      addPrefixStages(path_task, node, group, ee_link, attach_link, object_id, table_name,
                      touch_links, object_scene, pregrasp, grasp, lift, object_height,
                      object_radius, planning_time);
      auto ompl =
          std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
      ompl->setTimeout(planning_time);
      auto move_c = std::make_unique<moveit::task_constructor::stages::MoveTo>(
          "MoveTo Dense TopCircle", ompl);
      move_c->setGroup(group);
      move_c->setIKFrame(ee_link);
      move_c->setGoal(metric.cand.tcp_target);
      move_c->setTimeout(planning_time);
      path_task.add(std::move(move_c));
      try
      {
        path_task.init();
        pr.planned = path_task.plan(static_cast<size_t>(std::max(1, path_attempts))) &&
                     path_task.numSolutions() > 0;
      }
      catch (const std::exception& ex)
      {
        pr.status = std::string("init/plan exception: ") + ex.what();
        path_results.push_back(pr);
        continue;
      }
      pr.status = pr.planned ? "PASS" : "FAIL";
      any_path_ok = any_path_ok || pr.planned;
      RCLCPP_INFO(node->get_logger(), "Lift→top_circle roll=%+.1f ik=%d : %s", pr.roll_deg,
                  pr.ik_index, pr.status.c_str());
      path_results.push_back(pr);
    }
  }

  const auto after_msg = waitForFreshJoints(node, std::chrono::seconds(5));
  bool robot_moved = false;
  if (after_msg)
  {
    const double drift = fr_task_planner::maxJointError(jointsFromMsg(*after_msg), actual);
    robot_moved = drift > 0.02;
    RCLCPP_INFO(node->get_logger(), "Robot moved because of this node? %s",
                robot_moved ? "YES" : "NO");
  }

  int total_raw = 0;
  int total_unique = 0;
  int total_free = 0;
  int forearm_col_n = 0;
  int upper_col_n = 0;
  int self_n = 0;
  int other_n = 0;
  for (const auto& metric : all_unique)
  {
    ++total_unique;
    if (metric.cand.valid)
    {
      ++total_free;
    }
    if (metric.forearm_column)
    {
      ++forearm_col_n;
    }
    if (metric.upperarm_column)
    {
      ++upper_col_n;
    }
    if (metric.self_collision)
    {
      ++self_n;
    }
    if (!metric.cand.valid && !metric.forearm_column && !metric.upperarm_column &&
        !metric.self_collision)
    {
      ++other_n;
    }
  }
  for (const auto& sweep : sweeps)
  {
    total_raw += sweep.raw_ik;
  }

  std::string classification = "INCONCLUSIVE";
  if (all_rolls.empty() || sweeps.empty())
  {
    classification = "INCONCLUSIVE";
  }
  else if (free_rolls.empty())
  {
    classification = "COLLISION_BLOCKED";
  }
  else if (any_path_ok)
  {
    classification = "READY";
  }
  else
  {
    classification = "ENDPOINT_ONLY";
  }

  std::ofstream yaml(output_path);
  yaml << std::fixed << std::setprecision(9);
  yaml << "status: PASS\n";
  yaml << "classification: " << classification << "\n";
  yaml << "fairino_config_commit: 1bc8133\n";
  yaml << "planner_base_commit: 0eb2c84\n";
  yaml << "p1: [" << p1.x() << ", " << p1.y() << ", " << p1.z() << "]\n";
  yaml << "d1: [" << d1.x() << ", " << d1.y() << ", " << d1.z() << "]\n";
  yaml << "preferred_up: [" << preferred_up.x() << ", " << preferred_up.y() << ", "
       << preferred_up.z() << "]\n";
  yaml << "geometry_changed: false\n";
  yaml << "collision_policy_changed: false\n";
  yaml << "roll_step_deg: 5.0\n";
  yaml << "roll_samples: " << all_rolls.size() << "\n";
  yaml << "independent_sweeps: " << independent_sweeps << "\n";
  yaml << "max_ik_solutions_per_pose: " << cfg.max_ik_solutions_per_pose << "\n";
  yaml << "min_ik_solution_distance: " << cfg.min_ik_solution_distance << "\n";
  yaml << "ik_enumeration: setFromIK with random nearby seeds, per-pose L2 dedup\n";
  yaml << "signed_distance_api: " << (distance_api_ok ? "true" : "false") << "\n";
  yaml << "penetration_metric: "
       << (distance_api_ok ? "signed_distance_and_contact_depth" : "contact_penetration_depth")
       << "\n";
  yaml << "penetration_is_not_strict_clearance: " << (distance_api_ok ? "false" : "true") << "\n";
  yaml << "max_p1_error_m: " << max_center_err << "\n";
  yaml << "max_d1_error_deg: " << max_normal_err << "\n";
  yaml << "robot_moved: " << (robot_moved ? "true" : "false") << "\n";
  yaml << "task_execute: false\n";
  yaml << "sweeps:\n";
  for (const auto& sweep : sweeps)
  {
    yaml << "  - run: " << sweep.run << "\n";
    yaml << "    roll_samples: " << sweep.roll_samples << "\n";
    yaml << "    raw_ik: " << sweep.raw_ik << "\n";
    yaml << "    unique_ik: " << sweep.unique_ik << "\n";
    yaml << "    collision_free_ik: " << sweep.free_ik << "\n";
  }
  yaml << "combined_raw_ik: " << total_raw << "\n";
  yaml << "combined_unique_ik: " << total_unique << "\n";
  yaml << "combined_free_ik: " << total_free << "\n";
  yaml << "per_roll:\n";
  for (const auto& roll : all_rolls)
  {
    const auto& st = combined[roll.roll_deg];
    yaml << "  - roll_deg: " << st.roll_deg << "\n";
    yaml << "    raw_ik: " << st.raw_ik << "\n";
    yaml << "    unique_ik: " << st.unique_ik << "\n";
    yaml << "    collision_free_ik: " << st.free_ik << "\n";
    yaml << "    dominant_collision: \"" << yamlEscape(st.dominant_pair) << "\"\n";
  }
  yaml << "collision_free_roll_angles:\n";
  if (free_rolls.empty())
  {
    yaml << "  []\n";
  }
  else
  {
    for (double ang : free_rolls)
    {
      yaml << "  - " << ang << "\n";
    }
  }
  yaml << "collision_summary:\n";
  yaml << "  forearm_column_unique_iks: " << forearm_col_n << "\n";
  yaml << "  upperarm_column_unique_iks: " << upper_col_n << "\n";
  yaml << "  self_collision_unique_iks: " << self_n << "\n";
  yaml << "  other_colliding_unique_iks: " << other_n << "\n";
  yaml << "best_near_feasible:\n";
  if (!best)
  {
    yaml << "  found: false\n";
  }
  else
  {
    yaml << "  found: true\n";
    yaml << "  roll_deg: " << best->cand.roll_deg << "\n";
    yaml << "  ik_index: " << best->cand.ik_index << "\n";
    yaml << "  valid: " << (best->cand.valid ? "true" : "false") << "\n";
    yaml << "  collision_pair: \"" << yamlEscape(best->dominant_pair) << "\"\n";
    yaml << "  forearm_column: " << (best->forearm_column ? "true" : "false") << "\n";
    yaml << "  signed_distance_available: " << (best->signed_distance_available ? "true" : "false")
         << "\n";
    if (best->signed_distance_available)
    {
      yaml << "  signed_distance: " << best->signed_distance << "\n";
      yaml << "  column_signed_distance: " << best->column_signed_distance << "\n";
    }
    yaml << "  penetration_available: " << (best->penetration_available ? "true" : "false") << "\n";
    yaml << "  forearm_column_penetration: " << best->forearm_column_penetration << "\n";
    yaml << "  max_column_penetration: " << best->max_column_penetration << "\n";
    yaml << "  joints:\n";
    for (const auto& name : kArmJoints)
    {
      yaml << "    " << name << ": " << best->cand.joints.at(name) << "\n";
    }
    yaml << "  why: ";
    if (best->cand.valid)
    {
      yaml << "collision-free endpoint\n";
    }
    else if (best->signed_distance_available)
    {
      yaml << "least-negative/most-positive robot-column signed distance\n";
    }
    else
    {
      yaml << "smallest forearm-column contact penetration (diagnostic, not clearance)\n";
    }
  }
  yaml << "phase_b:\n";
  yaml << "  performed: " << (phase_b_done ? "true" : "false") << "\n";
  yaml << "  step_deg: " << phase_b_step << "\n";
  yaml << "  window_deg: " << phase_b_window << "\n";
  yaml << "  centers: [";
  for (size_t i = 0; i < phase_b_centers.size(); ++i)
  {
    if (i)
    {
      yaml << ", ";
    }
    yaml << phase_b_centers[i];
  }
  yaml << "]\n";
  yaml << "  angle_count: " << phase_b_angles.size() << "\n";
  if (!phase_b_sweeps.empty())
  {
    yaml << "  raw_ik: " << phase_b_sweeps.front().raw_ik << "\n";
    yaml << "  unique_ik: " << phase_b_sweeps.front().unique_ik << "\n";
    yaml << "  collision_free_ik: " << phase_b_sweeps.front().free_ik << "\n";
  }
  yaml << "path_validation:\n";
  if (path_results.empty())
  {
    yaml << "  tested: false\n";
    yaml << "  reason: no collision-free endpoint\n";
  }
  else
  {
    yaml << "  tested: true\n";
    yaml << "  any_pass: " << (any_path_ok ? "true" : "false") << "\n";
    yaml << "  candidates:\n";
    for (const auto& pr : path_results)
    {
      yaml << "    - roll_deg: " << pr.roll_deg << "\n";
      yaml << "      ik_index: " << pr.ik_index << "\n";
      yaml << "      lift_to_target: " << pr.status << "\n";
    }
  }
  yaml.close();

  RCLCPP_INFO(node->get_logger(), "classification: %s", classification.c_str());
  RCLCPP_INFO(node->get_logger(), "collision-free rolls: %zu", free_rolls.size());
  if (best)
  {
    RCLCPP_INFO(node->get_logger(), "best roll=%+.1f pair=%s forearm_col=%s pen=%.6f dist_ok=%s",
                best->cand.roll_deg, best->dominant_pair.c_str(),
                best->forearm_column ? "YES" : "NO", best->forearm_column_penetration,
                best->signed_distance_available ? "YES" : "NO");
    RCLCPP_INFO(node->get_logger(), "best joints: %s", formatJoints(best->cand.joints).c_str());
  }
  RCLCPP_INFO(node->get_logger(), "Wrote %s", output_path.c_str());
  RCLCPP_INFO(node->get_logger(), "REAL ROBOT COMMANDS SENT: NO");

  shutdownSpinner(executor, spinner);
  return 0;
}
