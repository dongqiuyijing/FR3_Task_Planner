#include "fr_task_planner/inspection_endpoint_candidates.hpp"
#include "fr_task_planner/inspection_visibility.hpp"

#include <algorithm>
#include <array>
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
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <moveit/collision_detection/collision_common.h>
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
using fr_task_planner::checkGeometricVisibility;
using fr_task_planner::collectCollisionContacts;
using fr_task_planner::CollisionDiagConfig;
using fr_task_planner::DesignCamera;
using fr_task_planner::EndpointCandidate;
using fr_task_planner::EndpointGenerationConfig;
using fr_task_planner::filterRollsByView;
using fr_task_planner::fixedInspectionDesignCamera;
using fr_task_planner::generateValidEndpointCandidates;
using fr_task_planner::GeometricVisibilityResult;
using fr_task_planner::jointL2;
using fr_task_planner::jointsFromState;
using fr_task_planner::kArmJoints;
using fr_task_planner::maxJointError;
using fr_task_planner::poseToIso;
using fr_task_planner::RollPose;
using fr_task_planner::tcpInBase;
using fr_task_planner::ViewGeom;

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
  std::string hud = "STEP 12";
  std::string a_status = "PENDING";
  std::string b_status = "PENDING";
  std::string c_status = "PENDING";
  GeometricVisibilityResult vis;
  std::optional<EndpointCandidate> current;
  std::optional<EndpointCandidate> best;
  std::chrono::steady_clock::time_point last_pub{};
  double min_period = 0.4;

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
    addSphere(arr, 10, p1, 0.012, rgba(1.0f, 0.2f, 0.9f, 1.0f), "inspection/p1");
    addText(arr, 11, p1 + Eigen::Vector3d(0.0, 0.0, 0.06), "P1 SURFACE CENTER",
            rgba(1.0f, 0.4f, 1.0f, 1.0f), 0.03, "inspection/p1");
    addArrow(arr, 12, p1, p1 + 0.18 * d1, rgba(0.2f, 0.9f, 1.0f, 1.0f), "inspection/d1");
    addText(arr, 13, p1 + 0.22 * d1, "FACE OUTWARD NORMAL -Y", rgba(0.3f, 0.95f, 1.0f, 1.0f), 0.028,
            "inspection/d1");
    addSphere(arr, 20, camera.optical_center_world, 0.02, rgba(1.0f, 0.55f, 0.1f, 1.0f),
              "inspection/camera");
    addArrow(arr, 21, camera.optical_center_world,
             camera.optical_center_world + 0.22 * camera.optical_forward_world,
             rgba(1.0f, 0.7f, 0.15f, 1.0f), "inspection/camera");
    addText(arr, 22, camera.optical_center_world + Eigen::Vector3d(0.05, 0.0, 0.08),
            "CAMERA DESIGN POSITION\noptical +Y  (NOT calibrated)",
            rgba(1.0f, 0.7f, 0.2f, 1.0f), 0.03, "inspection/camera");
    addCircle(arr, 30, p1, d1, 0.0075, rgba(0.2f, 1.0f, 0.85f, 1.0f), "inspection/target_face");
    addText(arr, 31, p1 + Eigen::Vector3d(0.0, -0.05, 0.08), "TARGET CIRCULAR / SIDE ROI",
            rgba(0.3f, 1.0f, 0.85f, 1.0f), 0.026, "inspection/target_face");
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
    addText(arr, 61, Eigen::Vector3d(-0.25, 0.55, 1.40),
            std::string("A ") + a_status + "\nB " + b_status + "\nC " + c_status,
            rgba(0.9f, 0.95f, 1.0f, 1.0f), 0.032, "hud/views");
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
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("fr3_mtc_complete_abc_search", options);

  RCLCPP_INFO(node->get_logger(), "========== STEP 12 FIXED ABC COMPLETE-TASK SEARCH ==========");
  RCLCPP_INFO(node->get_logger(), "PLAN / SIM VISUALIZATION ONLY. No Gazebo execute. No real robot.");

  const auto home = readHome(node);
  const auto pregrasp = readPose(node, "pregrasp");
  const auto grasp = readPose(node, "grasp");
  const auto lift = readPose(node, "lift");
  const auto object_world = readPose(node, "object_world");
  const auto object_base = readPose(node, "object");
  const auto world_base = readPose(node, "world_base");
  const auto tcp_object = readPose(node, "tcp_object");
  const ViewGeom view_a = readViewGeom(node, "side_pos_y");
  const ViewGeom view_b = readViewGeom(node, "side_neg_y");
  const ViewGeom view_c = readViewGeom(node, "top_circle");
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
                                            "/tmp/fr3_step12_best_sampled_complete_task.yaml");
  const std::string diag_path =
      getString(node, "diagnostic_output_path", "/tmp/fr3_step12_search.yaml");
  const bool visualize = getBool(node, "visualize_search", true);
  const double vis_hold = node->has_parameter("visualization_hold_seconds") ?
                              node->get_parameter("visualization_hold_seconds").as_double() :
                              12.0;
  const auto all_rolls = readRollPoses(node);
  const Eigen::Isometry3d t_world_base = poseToIso(world_base.pose);
  const Eigen::Isometry3d t_tcp_object = poseToIso(tcp_object.pose);
  DesignCamera camera = fixedInspectionDesignCamera();
  if (node->has_parameter("camera_design_x"))
  {
    camera.optical_center_world = Eigen::Vector3d(getDouble(node, "camera_design_x"),
                                                  getDouble(node, "camera_design_y"),
                                                  getDouble(node, "camera_design_z"));
  }

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
  viz.hud = "STEP 12  P1.z=1.20  STATIC GEOMETRY";

  const bool p1_ok = std::abs(p1.x()) < 1e-9 && std::abs(p1.y() - 0.4) < 1e-9 &&
                     std::abs(p1.z() - 1.20) < 1e-9;
  if (!p1_ok)
  {
    RCLCPP_ERROR(node->get_logger(), "GEOMETRY_INVALID: P1 is not [0,0.4,1.2], got [%.6f, %.6f, %.6f]",
                 p1.x(), p1.y(), p1.z());
    failure_class = "GEOMETRY_INVALID";
  }

  const std::array<const ViewGeom*, 3> views = { &view_a, &view_b, &view_c };
  bool geom_pass = p1_ok;
  for (const auto* view : views)
  {
    const auto rolls = filterRollsByView(all_rolls, view->name);
    int ori_ok = 0;
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
  if (maxJointError(actual, home) > home_tol)
  {
    RCLCPP_ERROR(node->get_logger(), "Not at Stage4 Home");
    shutdownSpinner(executor, spinner);
    return 2;
  }

  moveit::task_constructor::Task gen_task("", false);
  gen_task.setName("FR3 Step12 Prefix");
  gen_task.loadRobotModel(node);
  const auto robot_model = gen_task.getRobotModel();
  geometry_msgs::msg::PoseStamped object_scene = object_world;
  if (robot_model->getModelFrame() != object_world.header.frame_id)
  {
    object_scene = object_base;
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
                    object_scene, pregrasp, grasp, lift, object_height, object_radius, planning_time);
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
  viz.hud = "GENERATING A/B/C ENDPOINTS @ P1.z=1.20";
  evaluate_view(view_a, a_counts, a_kept, "A side_pos_y");
  viz.a_status = a_kept.empty() ? "NO ENDPOINT" : (std::to_string(a_kept.size()) + " vis+free");
  evaluate_view(view_b, b_counts, b_kept, "B side_neg_y");
  viz.b_status = b_kept.empty() ? "NO ENDPOINT" : (std::to_string(b_kept.size()) + " vis+free");
  evaluate_view(view_c, c_counts, c_kept, "C top_circle");
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
    yaml << "camera_forward: [0, 1, 0]\n";
    yaml << "order: A,B,C\n";
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

  if (c_counts.collision_free == 0)
  {
    failure_class = "C_COLLISION_BLOCKED_AT_FIXED_H_1_20";
    RCLCPP_ERROR(node->get_logger(),
                 "C_COLLISION_BLOCKED_AT_FIXED_H_1_20 rolls=%d rawIK=%d vis=%d free=0 pair=%s "
                 "best_signed=%.4f",
                 c_counts.roll_samples, c_counts.raw_ik, c_counts.visibility_valid,
                 c_counts.dominant_pair.c_str(),
                 c_counts.distance_available ? c_counts.best_column_distance : NAN);
    write_diag("FAIL", failure_class, nullptr);
    viz.hud = "C_COLLISION_BLOCKED_AT_FIXED_H_1_20";
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
    if (a_counts.raw_ik == 0 || b_counts.raw_ik == 0 || c_counts.raw_ik == 0)
    {
      failure_class = "IK_UNREACHABLE";
    }
    else if (a_counts.collision_free == 0 || b_counts.collision_free == 0)
    {
      failure_class = "ROBOT_COLUMN_COLLISION";
    }
    else
    {
      failure_class = "VISIBILITY_BLOCKED";
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
    failure_class = "A_TRANSITION_FAILED";
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
                    object_scene, pregrasp, grasp, lift, object_height, object_radius, planning_time);
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
    yaml << "p1: [0.0, 0.4, 1.2]\n";
    yaml << "camera_design_position: [" << camera.optical_center_world.x() << ", "
         << camera.optical_center_world.y() << ", " << camera.optical_center_world.z() << "]\n";
    yaml << "camera_forward: [0.0, 1.0, 0.0]\n";
    yaml << "camera_calibrated: false\n";
    yaml << "order: [side_pos_y, side_neg_y, top_circle]\n";
    yaml << "pregrasp_m: 0.08\n";
    yaml << "lift_m: 0.08\n";
    yaml << "t_tcp_object_xyzw: [" << tcp_object.pose.orientation.x << ", "
         << tcp_object.pose.orientation.y << ", " << tcp_object.pose.orientation.z << ", "
         << tcp_object.pose.orientation.w << "]\n";
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
    yaml << "  - name: Home_to_PreGrasp\n";
    yaml << "  - name: PreGrasp_to_Grasp\n";
    yaml << "  - name: Grasp_Attach_Lift\n";
    yaml << "  - name: Lift_to_A\n";
    yaml << "  - name: A_to_B\n";
    yaml << "  - name: B_to_C\n";
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
  viz.hud = "BEST SAMPLED FEASIBLE COMPLETE TASK\nHome→A→B→C PLAYBACK (RViz only)";
  viz.a_status = "VISIBLE";
  viz.b_status = "VISIBLE";
  viz.c_status = "VISIBLE";
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
