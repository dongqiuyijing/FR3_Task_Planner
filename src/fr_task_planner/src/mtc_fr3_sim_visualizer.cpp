#include "fr_task_planner/inspection_endpoint_candidates.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <thread>

#include <geometry_msgs/msg/point.hpp>
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
using fr_task_planner::CollisionDiagConfig;
using fr_task_planner::diagnoseIkCollisions;
using fr_task_planner::DifferentialCollision;
using fr_task_planner::EndpointCandidate;
using fr_task_planner::EndpointGenerationConfig;
using fr_task_planner::filterRollsByView;
using fr_task_planner::generateValidEndpointCandidates;
using fr_task_planner::isoToPose;
using fr_task_planner::kArmJoints;
using fr_task_planner::poseToIso;
using fr_task_planner::RollPose;
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

bool getBool(const rclcpp::Node::SharedPtr& node, const std::string& name, bool fallback)
{
  if (!node->has_parameter(name))
  {
    return fallback;
  }
  return node->get_parameter(name).as_bool();
}

std::vector<std::string> readTouchLinks(const rclcpp::Node::SharedPtr& node)
{
  return node->get_parameter("touch_links").as_string_array();
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
                                          const rclcpp::Node::SharedPtr& helper, int timeout_s)
{
  auto client = std::make_shared<rclcpp::SyncParametersClient>(helper, "move_group");
  if (!client->wait_for_service(std::chrono::seconds(timeout_s)))
  {
    RCLCPP_ERROR(target->get_logger(), "Timed out waiting for /move_group.");
    return false;
  }
  for (const auto& parameter :
       client->get_parameters({ "robot_description", "robot_description_semantic" }))
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

struct LiveSceneReport
{
  bool object_world = false;
  bool object_attached = false;
  bool table = false;
  bool column = false;
  std::vector<std::string> world_ids;
};

LiveSceneReport inspectLiveScene(const rclcpp::Node::SharedPtr& node, const std::string& object_id,
                                 const std::string& table_name, const std::string& column_name)
{
  LiveSceneReport report;
  auto client = node->create_client<moveit_msgs::srv::GetPlanningScene>("get_planning_scene");
  if (!client->wait_for_service(std::chrono::seconds(8)))
  {
    return report;
  }
  auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
  request->components.components =
      moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_NAMES |
      moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_GEOMETRY |
      moveit_msgs::msg::PlanningSceneComponents::ROBOT_STATE_ATTACHED_OBJECTS;
  auto future = client->async_send_request(request);
  if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready)
  {
    return report;
  }
  const auto response = future.get();
  for (const auto& obj : response->scene.world.collision_objects)
  {
    report.world_ids.push_back(obj.id);
    report.object_world = report.object_world || obj.id == object_id;
    report.table = report.table || obj.id == table_name;
    report.column = report.column || obj.id == column_name;
  }
  for (const auto& obj : response->scene.robot_state.attached_collision_objects)
  {
    report.object_attached = report.object_attached || obj.object.id == object_id;
  }
  return report;
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

struct ViewSpec
{
  std::string name;
  geometry_msgs::msg::PoseStamped tcp;
  geometry_msgs::msg::PoseStamped object;
};

ViewSpec readViewSpec(const rclcpp::Node::SharedPtr& node, const std::string& prefix)
{
  ViewSpec spec;
  spec.name = getString(node, prefix + "_view", "");
  spec.tcp = readPose(node, prefix);
  spec.object = readPose(node, prefix + "_object");
  return spec;
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

void addPrefixAndViews(moveit::task_constructor::Task& task, const rclcpp::Node::SharedPtr& node,
                       const std::string& group, const std::string& ee_link,
                       const std::string& attach_link, const std::string& object_id,
                       const std::string& table_name, const std::vector<std::string>& touch_links,
                       const geometry_msgs::msg::PoseStamped& object_scene,
                       const geometry_msgs::msg::PoseStamped& pregrasp,
                       const geometry_msgs::msg::PoseStamped& grasp,
                       const geometry_msgs::msg::PoseStamped& lift, const ViewSpec& source,
                       const ViewSpec& target, double object_height, double object_radius,
                       double planning_time)
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

  auto ompl_a =
      std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
  ompl_a->setTimeout(planning_time);
  auto move_a = std::make_unique<moveit::task_constructor::stages::MoveTo>(
      "MoveTo Source Inspection View " + source.name, ompl_a);
  move_a->setGroup(group);
  move_a->setIKFrame(ee_link);
  move_a->setGoal(source.tcp);
  move_a->setTimeout(planning_time);
  task.add(std::move(move_a));

  auto ompl_b =
      std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
  ompl_b->setTimeout(planning_time);
  auto move_b = std::make_unique<moveit::task_constructor::stages::MoveTo>(
      "MoveTo Target Inspection View " + target.name, ompl_b);
  move_b->setGroup(group);
  move_b->setIKFrame(ee_link);
  move_b->setGoal(target.tcp);
  move_b->setTimeout(planning_time);
  task.add(std::move(move_b));
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
              const Eigen::Vector3d& end, const std_msgs::msg::ColorRGBA& color,
              const std::string& ns)
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

void addCube(visualization_msgs::msg::MarkerArray& arr, int id, const Eigen::Vector3d& center,
             const Eigen::Vector3d& size, const std_msgs::msg::ColorRGBA& color,
             const std::string& ns)
{
  auto m = baseMarker(id, ns, visualization_msgs::msg::Marker::CUBE);
  m.pose.position = toPoint(center);
  m.scale.x = size.x();
  m.scale.y = size.y();
  m.scale.z = size.z();
  m.color = color;
  arr.markers.push_back(m);
}

void addCylinder(visualization_msgs::msg::MarkerArray& arr, int id, const Eigen::Vector3d& center,
                 double radius, double height, const std_msgs::msg::ColorRGBA& color,
                 const std::string& ns)
{
  auto m = baseMarker(id, ns, visualization_msgs::msg::Marker::CYLINDER);
  m.pose.position = toPoint(center);
  m.scale.x = 2.0 * radius;
  m.scale.y = 2.0 * radius;
  m.scale.z = height;
  m.color = color;
  arr.markers.push_back(m);
}

struct WorkcellGeom
{
  Eigen::Vector3d table_center = Eigen::Vector3d::Zero();
  Eigen::Vector3d table_size = Eigen::Vector3d::Zero();
  Eigen::Vector3d column_center = Eigen::Vector3d::Zero();
  Eigen::Vector3d column_size = Eigen::Vector3d::Zero();
  Eigen::Vector3d part_center = Eigen::Vector3d::Zero();
  Eigen::Vector3d robot_base = Eigen::Vector3d::Zero();
  double part_radius = 0.0075;
  double part_height = 0.035;
};

Eigen::Isometry3d poseInWorld(const geometry_msgs::msg::PoseStamped& pose,
                              const Eigen::Isometry3d& t_world_base)
{
  const Eigen::Isometry3d local = poseToIso(pose.pose);
  if (pose.header.frame_id == "world")
  {
    return local;
  }
  return t_world_base * local;
}

bool pairIs(const std::string& a, const std::string& b, const std::string& x, const std::string& y)
{
  return (a == x && b == y) || (a == y && b == x);
}

struct SafetyReport
{
  std::string domain = "unset";
  bool simulation = false;
  bool fr3_real = false;
  bool gazebo_sim = false;
  bool real_nodes = false;
  bool allow_execute = false;
  std::string reason;
};

SafetyReport checkSafety(const rclcpp::Node::SharedPtr& node, const std::string& robot_description)
{
  SafetyReport report;
  const char* domain = std::getenv("ROS_DOMAIN_ID");
  report.domain = domain ? std::string(domain) : "unset";
  report.fr3_real = robot_description.find("FR3RealSystem") != std::string::npos;
  report.gazebo_sim = robot_description.find("GazeboSimSystem") != std::string::npos ||
                      robot_description.find("gz_ros2_control") != std::string::npos;
  const auto names = node->get_node_graph_interface()->get_node_names();
  for (const auto& name : names)
  {
    if (name.find("real_bringup") != std::string::npos ||
        name.find("ros2_cmd_server") != std::string::npos ||
        name.find("FR3Real") != std::string::npos)
    {
      report.real_nodes = true;
    }
  }
  report.simulation = report.gazebo_sim && !report.fr3_real && !report.real_nodes;
  if (report.fr3_real || report.real_nodes)
  {
    report.allow_execute = false;
    report.reason = "real FR3 hardware or real bringup detected; execution aborted";
  }
  else if (!report.gazebo_sim)
  {
    report.allow_execute = false;
    report.reason = "GazeboSimSystem not confirmed";
  }
  else
  {
    report.allow_execute = true;
    report.reason = "simulation hardware only";
  }
  return report;
}

struct GhostPick
{
  const EndpointCandidate* cand = nullptr;
  DifferentialCollision diff;
  Eigen::Vector3d contact_world = Eigen::Vector3d::Zero();
  bool contact_valid = false;
  bool forearm_column = false;
};

GhostPick pickTopCircleGhost(const planning_scene::PlanningScene& lift_scene,
                             const std::vector<EndpointCandidate>& raw,
                             const CollisionDiagConfig& dcfg, const Eigen::Isometry3d& t_world_base,
                             const std::string& model_frame)
{
  GhostPick best;
  int best_score = -1000000;
  for (const auto& cand : raw)
  {
    if (std::abs(cand.roll_deg) > 1e-6)
    {
      continue;
    }
    auto scene = planning_scene::PlanningScene::clone(lift_scene.diff());
    applyJoints(scene->getCurrentStateNonConst(), cand.joints);
    const auto diff = diagnoseIkCollisions(lift_scene, cand.joints, dcfg);
    bool forearm_column = false;
    int extra_pairs = 0;
    for (const auto& contact : diff.full.contacts)
    {
      if (pairIs(contact.a, contact.b, "forearm_link", dcfg.column_name))
      {
        forearm_column = true;
      }
      else
      {
        ++extra_pairs;
      }
    }
    if (!forearm_column)
    {
      continue;
    }
    int score = 1000 - extra_pairs * 20 - (diff.self_only.collision ? 80 : 0);
    if (score > best_score)
    {
      best_score = score;
      best.cand = &cand;
      best.diff = diff;
      best.forearm_column = true;
      collision_detection::CollisionRequest req;
      req.contacts = true;
      req.max_contacts = 200;
      req.max_contacts_per_pair = 8;
      collision_detection::CollisionResult res;
      scene->checkCollision(req, res);
      for (const auto& item : res.contacts)
      {
        if (pairIs(item.first.first, item.first.second, "forearm_link", dcfg.column_name) &&
            !item.second.empty())
        {
          Eigen::Vector3d pos = item.second.front().pos;
          if (model_frame == "world")
          {
            best.contact_world = pos;
          }
          else
          {
            best.contact_world = t_world_base * pos;
          }
          best.contact_valid = pos.norm() > 1e-9 || std::isfinite(pos.x());
          if (!std::isfinite(best.contact_world.x()))
          {
            best.contact_valid = false;
          }
          break;
        }
      }
    }
  }
  if (best.cand)
  {
    return best;
  }
  for (const auto& cand : raw)
  {
    const auto diff = diagnoseIkCollisions(lift_scene, cand.joints, dcfg);
    bool forearm_column = false;
    for (const auto& contact : diff.full.contacts)
    {
      if (pairIs(contact.a, contact.b, "forearm_link", dcfg.column_name))
      {
        forearm_column = true;
      }
    }
    if (forearm_column)
    {
      best.cand = &cand;
      best.diff = diff;
      best.forearm_column = true;
      break;
    }
  }
  return best;
}

moveit_msgs::msg::DisplayRobotState makeGhost(
    const planning_scene::PlanningScene& lift_scene, const std::map<std::string, double>& joints)
{
  auto scene = planning_scene::PlanningScene::clone(lift_scene.diff());
  applyJoints(scene->getCurrentStateNonConst(), joints);
  moveit_msgs::msg::DisplayRobotState display;
  moveit::core::robotStateToRobotStateMsg(scene->getCurrentState(), display.state, true);
  moveit_msgs::msg::ObjectColor forearm;
  forearm.id = "forearm_link";
  forearm.color = rgba(1.0f, 0.05f, 0.05f, 0.95f);
  display.highlight_links.push_back(forearm);
  moveit_msgs::msg::ObjectColor upper;
  upper.id = "upperarm_link";
  upper.color = rgba(1.0f, 0.45f, 0.05f, 0.7f);
  display.highlight_links.push_back(upper);
  display.hide = false;
  return display;
}

moveit_msgs::msg::DisplayTrajectory makeDisplayTrajectory(
    const moveit_task_constructor_msgs::msg::Solution& sol, const std::string& model_id)
{
  moveit_msgs::msg::DisplayTrajectory dt;
  dt.model_id = model_id;
  dt.trajectory_start = sol.start_scene.robot_state;
  for (const auto& sub : sol.sub_trajectory)
  {
    if (!sub.trajectory.joint_trajectory.points.empty() ||
        !sub.trajectory.multi_dof_joint_trajectory.points.empty())
    {
      dt.trajectory.push_back(sub.trajectory);
    }
  }
  return dt;
}

visualization_msgs::msg::MarkerArray buildStaticMarkers(
    const Eigen::Vector3d& p1, const Eigen::Vector3d& d1, const Eigen::Vector3d& up,
    const Eigen::Isometry3d& a_tcp, const Eigen::Isometry3d& b_tcp,
    const Eigen::Isometry3d& c_tcp, const WorkcellGeom& workcell, const GhostPick& ghost,
    const std::string& stage_text)
{
  visualization_msgs::msg::MarkerArray arr;

  addCube(arr, 1, workcell.table_center, workcell.table_size, rgba(0.55f, 0.38f, 0.18f, 0.28f),
          "workcell/table");
  addText(arr, 2,
          workcell.table_center + Eigen::Vector3d(0.0, 0.0, 0.5 * workcell.table_size.z() + 0.06),
          "TABLE", rgba(1.0f, 0.85f, 0.45f, 1.0f), 0.05, "workcell/table_label");

  addCube(arr, 3, workcell.column_center, workcell.column_size, rgba(0.35f, 0.45f, 0.55f, 0.30f),
          "workcell/column");
  addText(arr, 4,
          workcell.column_center + Eigen::Vector3d(0.0, 0.14, 0.5 * workcell.column_size.z() + 0.04),
          "MOUNTING COLUMN", rgba(0.75f, 0.85f, 1.0f, 1.0f), 0.045, "workcell/column_label");

  addCylinder(arr, 5, workcell.part_center, workcell.part_radius, workcell.part_height,
              rgba(0.95f, 0.75f, 0.15f, 0.95f), "workcell/part");
  addSphere(arr, 6, workcell.part_center, 0.04, rgba(1.0f, 0.9f, 0.1f, 0.22f),
            "workcell/part_hint");
  addArrow(arr, 7, workcell.part_center + Eigen::Vector3d(0.0, -0.12, 0.10), workcell.part_center,
           rgba(1.0f, 0.85f, 0.1f, 1.0f), "workcell/part_label");
  addText(arr, 8, workcell.part_center + Eigen::Vector3d(0.0, -0.14, 0.12),
          "PART\nsmall_part 15mm x 35mm\nVISUAL MARKER ONLY", rgba(1.0f, 0.92f, 0.4f, 1.0f), 0.03,
          "workcell/part_label");

  addText(arr, 9, workcell.robot_base + Eigen::Vector3d(0.0, -0.08, 0.08), "FR3 BASE",
          rgba(0.85f, 0.95f, 1.0f, 1.0f), 0.04, "workcell/base_label");

  addSphere(arr, 10, p1, 0.03, rgba(0.1f, 0.75f, 1.0f, 0.95f), "inspection/p1");
  addText(arr, 11, p1 + Eigen::Vector3d(0.0, 0.0, 0.07), "P1 / INSPECTION CENTER",
          rgba(0.8f, 0.95f, 1.0f, 1.0f), 0.038, "inspection/p1");
  addArrow(arr, 12, p1, p1 + 0.20 * d1, rgba(1.0f, 0.85f, 0.1f, 1.0f), "inspection/d1");
  addText(arr, 13, p1 + 0.23 * d1, "D1 / FACE NORMAL / CAMERA DIRECTION",
          rgba(1.0f, 0.9f, 0.3f, 1.0f), 0.032, "inspection/d1");
  addArrow(arr, 14, p1, p1 + 0.20 * up, rgba(0.85f, 0.25f, 0.95f, 1.0f), "inspection/up");
  addText(arr, 15, p1 + 0.23 * up, "UP / WORLD +Z", rgba(0.95f, 0.7f, 1.0f, 1.0f), 0.032,
          "inspection/up");

  addSphere(arr, 20, a_tcp.translation(), 0.018, rgba(0.1f, 0.85f, 0.2f, 0.95f),
            "inspection/view_a");
  addAxes(arr, 21, a_tcp, 0.09, "inspection/view_a");
  addText(arr, 24, a_tcp.translation() + Eigen::Vector3d(0.08, 0.04, 0.08),
          "VIEW A\nside_pos_y\nREACHABLE", rgba(0.2f, 1.0f, 0.3f, 1.0f), 0.032, "inspection/view_a");

  addSphere(arr, 30, b_tcp.translation(), 0.018, rgba(0.1f, 0.85f, 0.2f, 0.95f),
            "inspection/view_b");
  addAxes(arr, 31, b_tcp, 0.09, "inspection/view_b");
  addText(arr, 34, b_tcp.translation() + Eigen::Vector3d(0.08, -0.06, 0.08),
          "VIEW B\nside_neg_y\nREACHABLE", rgba(0.2f, 1.0f, 0.3f, 1.0f), 0.032, "inspection/view_b");

  addSphere(arr, 40, c_tcp.translation(), 0.022, rgba(1.0f, 0.08f, 0.08f, 0.95f),
            "inspection/view_c");
  addAxes(arr, 41, c_tcp, 0.09, "inspection/view_c");
  addText(arr, 44, c_tcp.translation() + Eigen::Vector3d(0.0, 0.0, 0.12),
          "VIEW C\ntop_circle\nUNREACHABLE\nFOREARM <-> COLUMN COLLISION",
          rgba(1.0f, 0.25f, 0.2f, 1.0f), 0.032, "inspection/view_c");

  addText(arr, 50, Eigen::Vector3d(0.0, 0.15, 1.48), std::string("Current stage:\n") + stage_text,
          rgba(1.0f, 1.0f, 1.0f, 1.0f), 0.05, "stage/current");

  addText(arr, 60,
          workcell.column_center + Eigen::Vector3d(0.12, 0.16, 0.35),
          "Collision: forearm_link <-> mounting_column", rgba(1.0f, 0.15f, 0.15f, 1.0f), 0.035,
          "collision/warning");
  if (ghost.contact_valid)
  {
    addSphere(arr, 61, ghost.contact_world, 0.03, rgba(1.0f, 0.0f, 0.0f, 1.0f),
              "collision/contact");
    addText(arr, 62, ghost.contact_world + Eigen::Vector3d(0.0, 0.0, 0.05),
            "forearm_link <-> mounting_column", rgba(1.0f, 0.4f, 0.4f, 1.0f), 0.03,
            "collision/contact");
  }
  return arr;
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("fr3_mtc_sim_visualizer", options);

  const bool execute_gazebo = getBool(node, "execute_gazebo", false);
  const bool visualize_only = getBool(node, "visualize_only", true) || !execute_gazebo;
  const double hold_seconds = node->has_parameter("hold_seconds") ?
                                  node->get_parameter("hold_seconds").as_double() :
                                  2.0;
  const double vis_hold = node->has_parameter("visualization_hold_seconds") ?
                              node->get_parameter("visualization_hold_seconds").as_double() :
                              0.0;
  const bool hold_for_introspection = getBool(node, "hold_for_introspection", true);
  const int startup_timeout =
      node->has_parameter("startup_timeout_sec") ?
          static_cast<int>(node->get_parameter("startup_timeout_sec").as_int()) :
          90;

  RCLCPP_INFO(node->get_logger(), "========== STEP 11D WORKCELL VISUALIZATION ==========");
  RCLCPP_INFO(node->get_logger(), "[VIS] visualize_only=%s execute_gazebo=%s",
              visualize_only ? "true" : "false", execute_gazebo ? "true" : "false");
  RCLCPP_INFO(node->get_logger(), "[VIS] top_circle will NOT be executed");
  RCLCPP_INFO(node->get_logger(), "REAL ROBOT COMMANDS SENT: NO");

  const auto home = readHome(node);
  const auto pregrasp = readPose(node, "pregrasp");
  const auto grasp = readPose(node, "grasp");
  const auto lift = readPose(node, "lift");
  const auto object_world = readPose(node, "object_world");
  const auto object_base = readPose(node, "object");
  const auto world_base = readPose(node, "world_base");
  const auto tcp_object = readPose(node, "tcp_object");
  const ViewSpec source = readViewSpec(node, "source");
  const ViewSpec target = readViewSpec(node, "target");
  const ViewGeom top_view = readViewGeom(node, "top_circle");
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
  const auto touch_links = readTouchLinks(node);
  const double object_height = getDouble(node, "object_height");
  const double object_radius = getDouble(node, "object_radius");
  const double home_tol = node->has_parameter("max_home_error_rad") ?
                              node->get_parameter("max_home_error_rad").as_double() :
                              0.03;
  const double pos_tol = getDouble(node, "position_tolerance");
  const double ori_tol_deg = getDouble(node, "orientation_tolerance_deg");
  const double planning_time = getDouble(node, "planning_time");
  WorkcellGeom workcell;
  workcell.table_center = Eigen::Vector3d(getDouble(node, "table_cx"), getDouble(node, "table_cy"),
                                          getDouble(node, "table_cz"));
  workcell.table_size = Eigen::Vector3d(getDouble(node, "table_dx"), getDouble(node, "table_dy"),
                                        getDouble(node, "table_dz"));
  workcell.column_center = Eigen::Vector3d(getDouble(node, "column_cx"), getDouble(node, "column_cy"),
                                           getDouble(node, "column_cz"));
  workcell.column_size = Eigen::Vector3d(getDouble(node, "column_dx"), getDouble(node, "column_dy"),
                                         getDouble(node, "column_dz"));
  workcell.part_center =
      Eigen::Vector3d(object_world.pose.position.x, object_world.pose.position.y,
                      object_world.pose.position.z);
  workcell.robot_base =
      Eigen::Vector3d(world_base.pose.position.x, world_base.pose.position.y, world_base.pose.position.z);
  workcell.part_radius = object_radius;
  workcell.part_height = object_height;
  const auto all_rolls = readRollPoses(node);
  const Eigen::Isometry3d t_world_base = poseToIso(world_base.pose);

  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
  auto marker_pub = node->create_publisher<visualization_msgs::msg::MarkerArray>(kMarkerTopic, qos);
  auto ghost_pub = node->create_publisher<moveit_msgs::msg::DisplayRobotState>(kGhostTopic, qos);
  auto traj_pub = node->create_publisher<moveit_msgs::msg::DisplayTrajectory>(kTrajTopic, qos);

  auto stage_text = std::make_shared<std::string>("HOME");
  auto ghost_msg = std::make_shared<moveit_msgs::msg::DisplayRobotState>();
  auto ghost_pick = std::make_shared<GhostPick>();
  auto a_tcp = std::make_shared<Eigen::Isometry3d>(poseInWorld(source.tcp, t_world_base));
  auto b_tcp = std::make_shared<Eigen::Isometry3d>(poseInWorld(target.tcp, t_world_base));
  auto c_tcp = std::make_shared<Eigen::Isometry3d>(Eigen::Isometry3d::Identity());
  for (const auto& roll : all_rolls)
  {
    if (roll.view == "top_circle" && std::abs(roll.roll_deg) < 1e-9)
    {
      *c_tcp = poseInWorld(roll.tcp, t_world_base);
      break;
    }
  }

  auto publish_vis = [&]() {
    auto markers = buildStaticMarkers(p1, d1, preferred_up, *a_tcp, *b_tcp, *c_tcp, workcell,
                                      *ghost_pick, *stage_text);
    marker_pub->publish(markers);
    if (!ghost_msg->state.joint_state.name.empty() ||
        !ghost_msg->state.multi_dof_joint_state.joint_names.empty())
    {
      ghost_pub->publish(*ghost_msg);
    }
  };

  auto vis_timer = node->create_wall_timer(std::chrono::milliseconds(400), publish_vis);
  publish_vis();

  auto param_node = rclcpp::Node::make_shared("fr3_mtc_sim_visualizer_params");
  if (!overlayRobotDescriptionFromMoveGroup(node, param_node, startup_timeout))
  {
    rclcpp::shutdown();
    return 1;
  }

  std::string robot_description;
  node->get_parameter("robot_description", robot_description);
  const bool urdf_has_world = robot_description.find("name=\"world\"") != std::string::npos ||
                              robot_description.find("name='world'") != std::string::npos;
  RCLCPP_INFO(node->get_logger(), "robot_description contains world link: %s",
              urdf_has_world ? "YES" : "NO");
  const SafetyReport safety = checkSafety(node, robot_description);
  RCLCPP_INFO(node->get_logger(), "========== SAFETY ISOLATION ==========");
  RCLCPP_INFO(node->get_logger(), "ROS_DOMAIN_ID: %s", safety.domain.c_str());
  RCLCPP_INFO(node->get_logger(), "simulation: %s", safety.simulation ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "FR3RealSystem active: %s", safety.fr3_real ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "GazeboSimSystem present: %s", safety.gazebo_sim ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "real bringup / ros2_cmd_server: %s",
              safety.real_nodes ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "real gripper command used: NO");
  RCLCPP_INFO(node->get_logger(), "safety reason: %s", safety.reason.c_str());

  if (execute_gazebo)
  {
    if (!safety.allow_execute || safety.fr3_real || safety.real_nodes)
    {
      RCLCPP_ERROR(node->get_logger(),
                   "ABORT EXECUTION: real hardware detected. visualize_only only.");
    }
    else
    {
      RCLCPP_WARN(node->get_logger(),
                  "execute_gazebo requested but Gazebo trajectory execution is NOT IMPLEMENTED "
                  "in STEP 11C. Falling back to RViz visualize_only.");
    }
  }

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  LiveSceneReport live_before;
  for (int attempt = 0; attempt < 40 && rclcpp::ok(); ++attempt)
  {
    live_before = inspectLiveScene(node, object_id, table_name, column_name);
    if (live_before.table && live_before.column)
    {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  {
    std::ostringstream ids;
    for (size_t i = 0; i < live_before.world_ids.size(); ++i)
    {
      if (i)
      {
        ids << ", ";
      }
      ids << live_before.world_ids[i];
    }
    RCLCPP_INFO(node->get_logger(), "PlanningScene world objects: [%s]",
                ids.str().empty() ? "none" : ids.str().c_str());
    RCLCPP_INFO(node->get_logger(), "PlanningScene table=%s column=%s part_world=%s part_attached=%s",
                live_before.table ? "YES" : "NO", live_before.column ? "YES" : "NO",
                live_before.object_world ? "YES" : "NO",
                live_before.object_attached ? "YES" : "NO");
  }

  const auto before_msg = waitForFreshJoints(node, std::chrono::seconds(startup_timeout));
  if (!before_msg)
  {
    RCLCPP_ERROR(node->get_logger(), "/joint_states missing. STEP 11C FAIL.");
    shutdownSpinner(executor, spinner);
    return 1;
  }
  const auto actual = jointsFromMsg(*before_msg);
  if (jointsAreAllZero(actual))
  {
    RCLCPP_ERROR(node->get_logger(), "CurrentState is all zeros. STEP 11C FAIL.");
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
    RCLCPP_ERROR(node->get_logger(), "Current is not Stage4 Home. Planning aborted.");
    shutdownSpinner(executor, spinner);
    return 2;
  }

  *stage_text = "HOME";
  RCLCPP_INFO(node->get_logger(), "[VIS] STEP 1/7: Home");
  publish_vis();

  moveit::task_constructor::Task task("", true);
  task.setName("FR3 STEP11C A->B Visualization");
  task.loadRobotModel(node);
  const auto robot_model = task.getRobotModel();
  if (!robot_model || !robot_model->hasJointModelGroup(group) || !robot_model->hasLinkModel(ee_link))
  {
    RCLCPP_ERROR(node->get_logger(), "Robot model / group / TCP missing");
    shutdownSpinner(executor, spinner);
    return 1;
  }
  const std::string model_frame = robot_model->getModelFrame();
  geometry_msgs::msg::PoseStamped object_scene = object_world;
  if (model_frame != object_world.header.frame_id)
  {
    object_scene = object_base;
    object_scene.header.frame_id =
        model_frame == "base_link" ? object_base.header.frame_id : model_frame;
  }

  addPrefixAndViews(task, node, group, ee_link, attach_link, object_id, table_name, touch_links,
                    object_scene, pregrasp, grasp, lift, source, target, object_height,
                    object_radius, planning_time);

  RCLCPP_INFO(node->get_logger(),
              "MTC sequence: Home → PreGrasp(OMPL) → Grasp(Pilz LIN) → Attach(predicted) → "
              "Lift(Pilz LIN) → %s(OMPL) → %s(OMPL)",
              source.name.c_str(), target.name.c_str());
  RCLCPP_INFO(node->get_logger(), "top_circle is diagnostic only and is not a MoveTo goal");
  RCLCPP_INFO(node->get_logger(), "Collision checking: ENABLED");

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

  RCLCPP_INFO(node->get_logger(), "[VIS] STEP 2/7: Moving to PreGrasp (planning)");
  *stage_text = "PREGRASP";
  publish_vis();
  const auto plan_t0 = std::chrono::steady_clock::now();
  const auto plan_result = task.plan(1);
  const double planning_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - plan_t0).count();
  const size_t num_solutions = task.numSolutions();
  RCLCPP_INFO(node->get_logger(), "planning result: %s  solutions=%zu  time=%.3f s",
              plan_result ? "SUCCESS" : "FAIL", num_solutions, planning_s);
  if (!plan_result || num_solutions < 1)
  {
    std::ostringstream failure;
    task.explainFailure(failure);
    RCLCPP_ERROR(node->get_logger(), "A→B planning failed:\n%s", failure.str().c_str());
    shutdownSpinner(executor, spinner);
    return 3;
  }

  const auto& solution = *task.solutions().front();
  moveit_task_constructor_msgs::msg::Solution sol_msg;
  solution.toMsg(sol_msg, &task.introspection());
  traj_pub->publish(makeDisplayTrajectory(sol_msg, robot_model->getName()));
  task.publishAllSolutions(false);

  std::vector<const moveit::task_constructor::SolutionBase*> leaves;
  flattenSolutions(solution, leaves);
  const auto* restore_sol = findStageSolution(leaves, "Restore Part-Table Collision");
  if (!restore_sol || !restore_sol->end() || !restore_sol->end()->scene())
  {
    RCLCPP_ERROR(node->get_logger(), "Restore scene missing after A→B plan");
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
      static_cast<uint32_t>(node->has_parameter("max_ik_solutions_per_pose") ?
                                node->get_parameter("max_ik_solutions_per_pose").as_int() :
                                8);
  cfg.min_ik_solution_distance = node->has_parameter("min_ik_solution_distance") ?
                                     node->get_parameter("min_ik_solution_distance").as_double() :
                                     0.1;

  CollisionDiagConfig dcfg;
  dcfg.object_id = object_id;
  dcfg.table_name = table_name;
  dcfg.column_name = column_name;
  dcfg.touch_links = touch_links;

  const auto top_rolls = filterRollsByView(all_rolls, "top_circle");
  const auto top_gen =
      generateValidEndpointCandidates(*lift_scene, top_rolls, top_view, cfg, node->get_logger());
  *ghost_pick = pickTopCircleGhost(*lift_scene, top_gen.raw, dcfg, t_world_base, model_frame);
  if (ghost_pick->cand)
  {
    *ghost_msg = makeGhost(*lift_scene, ghost_pick->cand->joints);
    RCLCPP_INFO(node->get_logger(),
                "[VIS] TOP_CIRCLE ghost IK roll=%.1f idx=%d forearm_column=%s contact_marker=%s",
                ghost_pick->cand->roll_deg, ghost_pick->cand->ik_index,
                ghost_pick->forearm_column ? "YES" : "NO",
                ghost_pick->contact_valid ? "YES" : "NO");
  }
  else
  {
    RCLCPP_WARN(node->get_logger(), "[VIS] no top_circle colliding IK selected for ghost");
  }

  RCLCPP_INFO(node->get_logger(), "[VIS] STEP 3/7: Grasp approach");
  RCLCPP_INFO(node->get_logger(), "[VIS] STEP 4/7: Lift");
  RCLCPP_INFO(node->get_logger(), "[VIS] STEP 5/7: %s inspection", source.name.c_str());
  RCLCPP_INFO(node->get_logger(), "[VIS] STEP 6/7: %s inspection", target.name.c_str());
  RCLCPP_INFO(node->get_logger(), "[VIS] STEP 7/7: top_circle diagnostic only");
  RCLCPP_INFO(node->get_logger(), "[VIS] TOP_CIRCLE NOT EXECUTED");
  RCLCPP_INFO(node->get_logger(), "[VIS] Reason: forearm_link <-> mounting_column collision");
  RCLCPP_INFO(node->get_logger(),
              "[VIS] raw IK=%zu collision-free=%zu (counts vary with IK seeds)", top_gen.raw.size(),
              top_gen.valid.size());

  struct StageHold
  {
    std::string label;
    double seconds;
  };
  const std::vector<StageHold> holds = {
    { "HOME", 1.5 },
    { "PREGRASP", 2.5 },
    { "GRASP", 2.0 },
    { "LIFT", 2.0 },
    { std::string("INSPECTION A (") + source.name + ")", std::max(2.0, hold_seconds) },
    { std::string("INSPECTION B (") + target.name + ")", std::max(2.0, hold_seconds) },
    { "TOP CIRCLE COLLISION DIAGNOSTIC", 6.0 },
  };
  for (const auto& hold : holds)
  {
    if (!rclcpp::ok())
    {
      break;
    }
    *stage_text = hold.label;
    publish_vis();
    RCLCPP_INFO(node->get_logger(), "[VIS] Current stage: %s", hold.label.c_str());
    std::this_thread::sleep_for(std::chrono::duration<double>(hold.seconds));
  }
  *stage_text = "TOP CIRCLE COLLISION DIAGNOSTIC";
  publish_vis();

  std::ofstream yaml("/tmp/fr3_step11d_workcell.yaml");
  yaml << "status: PASS\n";
  yaml << "sequence: Home -> PreGrasp -> Grasp -> Lift -> " << source.name << " -> " << target.name
       << "\n";
  yaml << "planner: MTC + OMPL + Pilz LIN\n";
  yaml << "handcrafted_trajectory: false\n";
  yaml << "planning_successful: true\n";
  yaml << "planning_time_s: " << planning_s << "\n";
  yaml << "top_circle_executed: false\n";
  yaml << "top_circle_raw_ik: " << top_gen.raw.size() << "\n";
  yaml << "top_circle_collision_free: " << top_gen.valid.size() << "\n";
  yaml << "ghost_selected: " << (ghost_pick->cand ? "true" : "false") << "\n";
  yaml << "collision_pair: forearm_link <-> mounting_column\n";
  yaml << "contact_marker: " << (ghost_pick->contact_valid ? "true" : "false") << "\n";
  yaml << "execute_gazebo_implemented: false\n";
  yaml << "real_robot_commands_sent: false\n";
  yaml << "ros_domain_id: \"" << safety.domain << "\"\n";
  yaml << "urdf_has_world: " << (urdf_has_world ? "true" : "false") << "\n";
  yaml << "planning_scene_table: " << (live_before.table ? "true" : "false") << "\n";
  yaml << "planning_scene_column: " << (live_before.column ? "true" : "false") << "\n";
  yaml << "p1: [" << p1.x() << ", " << p1.y() << ", " << p1.z() << "]\n";
  yaml << "d1: [" << d1.x() << ", " << d1.y() << ", " << d1.z() << "]\n";
  yaml << "preferred_up: [" << preferred_up.x() << ", " << preferred_up.y() << ", "
       << preferred_up.z() << "]\n";
  yaml << "table_center: [" << workcell.table_center.x() << ", " << workcell.table_center.y()
       << ", " << workcell.table_center.z() << "]\n";
  yaml << "column_center: [" << workcell.column_center.x() << ", " << workcell.column_center.y()
       << ", " << workcell.column_center.z() << "]\n";
  yaml.close();

  const auto after_msg = waitForFreshJoints(node, std::chrono::seconds(5));
  if (after_msg)
  {
    const auto after = jointsFromMsg(*after_msg);
    double drift = 0.0;
    for (const auto& name : kArmJoints)
    {
      drift = std::max(drift, std::abs(after.at(name) - actual.at(name)));
    }
    RCLCPP_INFO(node->get_logger(), "Robot moved because of this node? %s",
                drift > 0.02 ? "YES" : "NO");
  }
  const LiveSceneReport live_after =
      inspectLiveScene(node, object_id, table_name, column_name);
  RCLCPP_INFO(node->get_logger(),
              "LIVE SCENE after: table=%s column=%s part_world=%s attached=%s "
              "(before table=%s column=%s part_world=%s attached=%s)",
              live_after.table ? "YES" : "NO", live_after.column ? "YES" : "NO",
              live_after.object_world ? "YES" : "NO", live_after.object_attached ? "YES" : "NO",
              live_before.table ? "YES" : "NO", live_before.column ? "YES" : "NO",
              live_before.object_world ? "YES" : "NO",
              live_before.object_attached ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(),
              "RViz: select the MTC solution in Motion Planning Tasks to replay "
              "Home→A→B. Red ghost is top_circle colliding IK, never executed.");

  if (hold_for_introspection)
  {
    RCLCPP_INFO(node->get_logger(),
                "Visualization holding for RViz. Ctrl+C to exit. REAL ROBOT COMMANDS SENT: NO");
    spinner.join();
  }
  else
  {
    if (vis_hold > 0.0)
    {
      std::this_thread::sleep_for(std::chrono::duration<double>(vis_hold));
    }
    shutdownSpinner(executor, spinner);
  }
  return 0;
}
