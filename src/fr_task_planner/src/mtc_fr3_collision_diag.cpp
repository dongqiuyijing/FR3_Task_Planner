#include "fr_task_planner/inspection_endpoint_candidates.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <thread>

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
using fr_task_planner::acmEntryAllowed;
using fr_task_planner::applyJoints;
using fr_task_planner::AttachedGeometryReport;
using fr_task_planner::CollisionCategory;
using fr_task_planner::CollisionDiagConfig;
using fr_task_planner::collisionCategoryName;
using fr_task_planner::ContactRecord;
using fr_task_planner::diagnoseIkCollisions;
using fr_task_planner::DifferentialCollision;
using fr_task_planner::EndpointCandidate;
using fr_task_planner::EndpointGenerationConfig;
using fr_task_planner::filterRollsByView;
using fr_task_planner::generateValidEndpointCandidates;
using fr_task_planner::inspectAttachedGeometry;
using fr_task_planner::isoToPose;
using fr_task_planner::jointsFromState;
using fr_task_planner::kArmJoints;
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
                                          const rclcpp::Node::SharedPtr& helper)
{
  auto client = std::make_shared<rclcpp::SyncParametersClient>(helper, "move_group");
  if (!client->wait_for_service(std::chrono::seconds(60)))
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

void inspectLiveScene(const rclcpp::Node::SharedPtr& node, const std::string& object_id,
                      bool& world_present, bool& attached_present, bool& table_present,
                      bool& column_present, const std::string& table_name,
                      const std::string& column_name)
{
  world_present = attached_present = table_present = column_present = false;
  auto client = node->create_client<moveit_msgs::srv::GetPlanningScene>("get_planning_scene");
  if (!client->wait_for_service(std::chrono::seconds(5)))
  {
    return;
  }
  auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
  request->components.components = moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_NAMES |
                                   moveit_msgs::msg::PlanningSceneComponents::ROBOT_STATE_ATTACHED_OBJECTS;
  auto future = client->async_send_request(request);
  if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready)
  {
    return;
  }
  const auto response = future.get();
  for (const auto& obj : response->scene.world.collision_objects)
  {
    world_present = world_present || obj.id == object_id;
    table_present = table_present || obj.id == table_name;
    column_present = column_present || obj.id == column_name;
  }
  for (const auto& obj : response->scene.robot_state.attached_collision_objects)
  {
    attached_present = attached_present || obj.object.id == object_id;
  }
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

std::string formatPose(const Eigen::Isometry3d& transform)
{
  const auto pose = isoToPose(transform);
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << "xyz=(" << pose.position.x << ", " << pose.position.y
      << ", " << pose.position.z << ") xyzw=(" << pose.orientation.x << ", " << pose.orientation.y
      << ", " << pose.orientation.z << ", " << pose.orientation.w << ")";
  return oss.str();
}

std::string formatVec(const Eigen::Vector3d& v)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
  return oss.str();
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

std::string formatContacts(const std::vector<ContactRecord>& contacts)
{
  if (contacts.empty())
  {
    return "none";
  }
  std::ostringstream oss;
  for (size_t i = 0; i < contacts.size(); ++i)
  {
    if (i)
    {
      oss << "; ";
    }
    oss << contacts[i].pair_key << " [" << collisionCategoryName(contacts[i].category) << "]";
    if (contacts[i].depth_available)
    {
      oss << " depth=" << contacts[i].depth;
    }
  }
  return oss.str();
}

struct IkDiag
{
  EndpointCandidate cand;
  DifferentialCollision diff;
};

std::string rootCauseLetter(size_t raw_n, size_t full_free, size_t no_part_free, size_t self_hit,
                            size_t no_table_free, size_t no_column_free,
                            const std::map<CollisionCategory, int>& cat_ik,
                            bool acm_bug, bool attach_bug)
{
  if (attach_bug)
  {
    return "G";
  }
  if (acm_bug)
  {
    return "F";
  }
  const int part_robot = cat_ik.count(CollisionCategory::PART_NON_TOUCH_ROBOT) ?
                             cat_ik.at(CollisionCategory::PART_NON_TOUCH_ROBOT) :
                             0;
  const int part_env = (cat_ik.count(CollisionCategory::PART_TABLE) ?
                            cat_ik.at(CollisionCategory::PART_TABLE) :
                            0) +
                       (cat_ik.count(CollisionCategory::PART_COLUMN) ?
                            cat_ik.at(CollisionCategory::PART_COLUMN) :
                            0);
  const int robot_col = cat_ik.count(CollisionCategory::ROBOT_COLUMN) ?
                            cat_ik.at(CollisionCategory::ROBOT_COLUMN) :
                            0;
  const int robot_tab =
      cat_ik.count(CollisionCategory::ROBOT_TABLE) ? cat_ik.at(CollisionCategory::ROBOT_TABLE) : 0;
  const int self =
      cat_ik.count(CollisionCategory::ROBOT_SELF) ? cat_ik.at(CollisionCategory::ROBOT_SELF) : 0;
  const bool part_clears = raw_n > 0 && no_part_free > raw_n / 2 && full_free == 0;
  const bool column_clears = raw_n > 0 && no_column_free > raw_n / 2;
  const bool table_clears = raw_n > 0 && no_table_free > raw_n / 2;
  int winners = 0;
  std::string letter = "I";
  if (self_hit > raw_n / 2 && self >= static_cast<int>(raw_n) / 2)
  {
    letter = "A";
    ++winners;
  }
  if (column_clears && robot_col >= static_cast<int>(raw_n) / 2)
  {
    letter = "B";
    ++winners;
  }
  if (table_clears && robot_tab >= static_cast<int>(raw_n) / 2)
  {
    letter = "C";
    ++winners;
  }
  if (part_clears && part_robot >= static_cast<int>(raw_n) / 2)
  {
    letter = "D";
    ++winners;
  }
  if (part_clears && part_env >= static_cast<int>(raw_n) / 2)
  {
    letter = "E";
    ++winners;
  }
  if (winners > 1)
  {
    return "H";
  }
  if (winners == 1)
  {
    return letter;
  }
  if (full_free == 0 && raw_n > 0)
  {
    return "H";
  }
  return "I";
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("fr3_mtc_collision_diag", options);
  RCLCPP_INFO(node->get_logger(), "STEP 11B DIAGNOSIS ONLY. NO EXECUTION. NO POLICY CHANGE.");

  const auto home = readHome(node);
  const auto pregrasp = readPose(node, "pregrasp");
  const auto grasp = readPose(node, "grasp");
  const auto lift = readPose(node, "lift");
  const auto object_world = readPose(node, "object_world");
  const auto object_base = readPose(node, "object");
  const auto tcp_object = readPose(node, "tcp_object");
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
  const std::string output_path =
      getString(node, "diagnostic_output_path", "/tmp/fr3_step11b_top_circle.yaml");
  const ViewGeom top_view = readViewGeom(node, "top_circle");
  const ViewGeom side_pos_view = readViewGeom(node, "side_pos_y");
  const ViewGeom side_neg_view = readViewGeom(node, "side_neg_y");
  const auto all_rolls = readRollPoses(node);

  auto param_node = rclcpp::Node::make_shared("fr3_mtc_collision_diag_params");
  if (!overlayRobotDescriptionFromMoveGroup(node, param_node))
  {
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  bool live_world_before = false;
  bool live_attached_before = false;
  bool live_table_before = false;
  bool live_column_before = false;
  inspectLiveScene(node, object_id, live_world_before, live_attached_before, live_table_before,
                   live_column_before, table_name, column_name);
  const auto before_msg = waitForFreshJoints(node, std::chrono::seconds(10));
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
  gen_task.setName("FR3 STEP11B Prefix");
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
  const auto side_pos_gen = generateValidEndpointCandidates(
      *lift_scene, filterRollsByView(all_rolls, "side_pos_y"), side_pos_view, cfg, node->get_logger());
  const auto side_neg_gen = generateValidEndpointCandidates(
      *lift_scene, filterRollsByView(all_rolls, "side_neg_y"), side_neg_view, cfg, node->get_logger());

  const AttachedGeometryReport attached = inspectAttachedGeometry(*lift_scene, object_id);
  const RollPose* canonical = nullptr;
  for (const auto& roll : top_rolls)
  {
    if (std::abs(roll.roll_deg) < 1e-9)
    {
      canonical = &roll;
      break;
    }
  }
  Eigen::Isometry3d t_world_tcp_theory = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d t_world_object_theory = Eigen::Isometry3d::Identity();
  if (canonical)
  {
    t_world_tcp_theory = cfg.t_model_base * poseToIso(canonical->tcp.pose);
    t_world_object_theory = t_world_tcp_theory * cfg.t_tcp_object;
  }

  std::vector<IkDiag> diags;
  diags.reserve(top_gen.raw.size());
  for (const auto& cand : top_gen.raw)
  {
    IkDiag item;
    item.cand = cand;
    item.diff = diagnoseIkCollisions(*lift_scene, cand.joints, dcfg);
    diags.push_back(item);
  }

  std::map<std::string, int> pair_freq;
  std::map<CollisionCategory, int> cat_pair;
  std::map<CollisionCategory, int> cat_ik;
  std::map<int, std::vector<size_t>> by_roll;
  size_t full_hit = 0;
  size_t full_free = 0;
  size_t no_part_free = 0;
  size_t self_hit = 0;
  size_t no_table_free = 0;
  size_t no_column_free = 0;
  bool any_depth = false;
  bool part_touch_illegal = false;
  bool part_wrist_forearm = false;
  std::set<std::string> part_non_touch_links;
  for (size_t i = 0; i < diags.size(); ++i)
  {
    const auto& item = diags[i];
    by_roll[static_cast<int>(std::lround(item.cand.roll_deg))].push_back(i);
    if (item.diff.full.collision)
    {
      ++full_hit;
    }
    else
    {
      ++full_free;
    }
    if (!item.diff.no_part.collision)
    {
      ++no_part_free;
    }
    if (item.diff.self_only.collision)
    {
      ++self_hit;
    }
    if (!item.diff.no_table.collision)
    {
      ++no_table_free;
    }
    if (!item.diff.no_column.collision)
    {
      ++no_column_free;
    }
    any_depth = any_depth || item.diff.full.depth_available;
    std::set<CollisionCategory> seen_cat;
    std::set<std::string> seen_pair;
    for (const auto& contact : item.diff.full.contacts)
    {
      if (seen_pair.insert(contact.pair_key).second)
      {
        pair_freq[contact.pair_key] += 1;
        cat_pair[contact.category] += 1;
      }
      seen_cat.insert(contact.category);
      if (contact.category == CollisionCategory::PART_TOUCH_ROBOT)
      {
        part_touch_illegal = true;
      }
      if (contact.category == CollisionCategory::PART_NON_TOUCH_ROBOT)
      {
        const std::string other = (contact.a == object_id) ? contact.b : contact.a;
        part_non_touch_links.insert(other);
        if (other.find("wrist") != std::string::npos || other.find("forearm") != std::string::npos)
        {
          part_wrist_forearm = true;
        }
      }
    }
    for (const auto cat : seen_cat)
    {
      cat_ik[cat] += 1;
    }
  }

  std::vector<std::pair<std::string, int>> pair_sorted(pair_freq.begin(), pair_freq.end());
  std::sort(pair_sorted.begin(), pair_sorted.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  auto pickControl = [&](const auto& gen) -> const EndpointCandidate* {
    if (!gen.valid.empty())
    {
      return &gen.valid.front();
    }
    if (!gen.raw.empty())
    {
      return &gen.raw.front();
    }
    return nullptr;
  };
  const EndpointCandidate* side_pos = pickControl(side_pos_gen);
  const EndpointCandidate* side_neg = pickControl(side_neg_gen);
  DifferentialCollision side_pos_diff;
  DifferentialCollision side_neg_diff;
  bool side_pos_ok = false;
  bool side_neg_ok = false;
  if (side_pos)
  {
    side_pos_diff = diagnoseIkCollisions(*lift_scene, side_pos->joints, dcfg);
    side_pos_ok = !side_pos_diff.full.collision && side_pos->valid;
  }
  if (side_neg)
  {
    side_neg_diff = diagnoseIkCollisions(*lift_scene, side_neg->joints, dcfg);
    side_neg_ok = !side_neg_diff.full.collision && side_neg->valid;
  }

  std::vector<std::string> acm_touch_rows;
  bool acm_bug = false;
  for (const auto& link : touch_links)
  {
    const bool allowed = acmEntryAllowed(*lift_scene, object_id, link);
    acm_touch_rows.push_back(object_id + " <-> " + link + " allowed=" + (allowed ? "YES" : "NO"));
    if (!allowed)
    {
      acm_bug = true;
    }
  }
  const std::vector<std::string> wrist_forearm = { "wrist1_link", "wrist2_link", "wrist3_link",
                                                   "forearm_link" };
  for (const auto& link : wrist_forearm)
  {
    const bool allowed = acmEntryAllowed(*lift_scene, object_id, link);
    acm_touch_rows.push_back(object_id + " <-> " + link + " allowed=" + (allowed ? "YES" : "NO"));
  }

  Eigen::Vector3d cyl_z_at_target = attached.local_z_in_world.normalized();
  AttachedGeometryReport attached_at_target = attached;
  for (const auto& item : diags)
  {
    if (std::abs(item.cand.roll_deg) > 1e-9)
    {
      continue;
    }
    auto target_scene = fr_task_planner::cloneDiagnosticScene(*lift_scene);
    fr_task_planner::applyJointsToScene(*target_scene, item.cand.joints);
    attached_at_target = inspectAttachedGeometry(*target_scene, object_id);
    if (attached_at_target.present)
    {
      cyl_z_at_target = attached_at_target.local_z_in_world.normalized();
    }
    break;
  }
  const Eigen::Vector3d cyl_z = attached.local_z_in_world.normalized();
  const Eigen::Vector3d expected_top_z(0.0, -1.0, 0.0);
  const double cyl_z_err_deg =
      std::acos(std::min(1.0, std::max(-1.0, cyl_z_at_target.dot(expected_top_z)))) * 180.0 / M_PI;
  bool attach_bug = false;
  if (canonical)
  {
    const Eigen::Vector3d theory_xyz = t_world_object_theory.translation();
    const Eigen::Vector3d expected_xyz(0.0, 0.4175, 1.1);
    if ((theory_xyz - expected_xyz).norm() > 0.002)
    {
      attach_bug = true;
    }
    if (attached.shape != "cylinder" || std::abs(attached.radius - 0.0075) > 1e-6 ||
        std::abs(attached.height - 0.035) > 1e-6)
    {
      attach_bug = true;
    }
    if (cyl_z_err_deg > ori_tol_deg)
    {
      attach_bug = true;
    }
  }

  const std::string letter =
      rootCauseLetter(diags.size(), full_free, no_part_free, self_hit, no_table_free,
                      no_column_free, cat_ik, part_touch_illegal || acm_bug, attach_bug);

  RCLCPP_INFO(node->get_logger(), "========== STEP 11B TOUCH LINKS ==========");
  for (const auto& link : touch_links)
  {
    RCLCPP_INFO(node->get_logger(), "configured touch link: %s", link.c_str());
  }
  for (const auto& row : acm_touch_rows)
  {
    RCLCPP_INFO(node->get_logger(), "ACM %s", row.c_str());
  }
  RCLCPP_INFO(node->get_logger(), "T_tcp_object %s", formatPose(cfg.t_tcp_object).c_str());
  RCLCPP_INFO(node->get_logger(), "attached shape=%s r=%.6f h=%.6f link=%s", attached.shape.c_str(),
              attached.radius, attached.height, attached.attached_link.c_str());
  if (canonical)
  {
    RCLCPP_INFO(node->get_logger(), "TOP_CIRCLE theoretical T_world_tcp %s",
                formatPose(t_world_tcp_theory).c_str());
    RCLCPP_INFO(node->get_logger(), "TOP_CIRCLE theoretical T_world_object %s",
                formatPose(t_world_object_theory).c_str());
  }
  RCLCPP_INFO(node->get_logger(), "attached cylinder +Z in world at lift %s",
              formatVec(cyl_z).c_str());
  RCLCPP_INFO(node->get_logger(),
              "attached cylinder +Z in world at canonical IK %s (top_circle expect -Y, err=%.3f deg)",
              formatVec(cyl_z_at_target).c_str(), cyl_z_err_deg);
  RCLCPP_INFO(node->get_logger(), "RAW IK top_circle=%zu valid=%zu", top_gen.raw.size(),
              top_gen.valid.size());

  for (const auto& item : diags)
  {
    RCLCPP_INFO(node->get_logger(),
                "IK roll=%.1f idx=%d %s bounds=%s full=%s no_part=%s self=%s no_table=%s "
                "no_column=%s contacts=%s",
                item.cand.roll_deg, item.cand.ik_index, formatJoints(item.cand.joints).c_str(),
                item.cand.bounds_ok ? "YES" : "NO", item.diff.full.collision ? "YES" : "NO",
                item.diff.no_part.collision ? "YES" : "NO",
                item.diff.self_only.collision ? "YES" : "NO",
                item.diff.no_table.collision ? "YES" : "NO",
                item.diff.no_column.collision ? "YES" : "NO",
                formatContacts(item.diff.full.contacts).c_str());
  }

  RCLCPP_INFO(node->get_logger(), "CONTROL side_pos_y valid_ik=%s collision=%s",
              side_pos && side_pos->valid ? "YES" : "NO",
              side_pos_diff.full.collision ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "CONTROL side_neg_y valid_ik=%s collision=%s",
              side_neg && side_neg->valid ? "YES" : "NO",
              side_neg_diff.full.collision ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(),
              "DIFF full_hit=%zu full_free=%zu no_part_free=%zu self_hit=%zu no_table_free=%zu "
              "no_column_free=%zu",
              full_hit, full_free, no_part_free, self_hit, no_table_free, no_column_free);
  RCLCPP_INFO(node->get_logger(), "ROOT_CAUSE_LETTER %s", letter.c_str());
  RCLCPP_INFO(node->get_logger(), "penetration depth %s",
              any_depth ? "available from FCL Contact.depth" : "unavailable");

  {
    std::ofstream yaml(output_path);
    yaml.setf(std::ios::fixed);
    yaml.precision(6);
    yaml << "step: 11b\n";
    yaml << "diagnosis_only: true\n";
    yaml << "official_candidate_validity: full_scene_only\n";
    yaml << "raw_ik: " << diags.size() << "\n";
    yaml << "full_scene_collision: " << full_hit << "\n";
    yaml << "full_scene_free: " << full_free << "\n";
    yaml << "without_attached_part_free: " << no_part_free << "\n";
    yaml << "robot_self_collision: " << self_hit << "\n";
    yaml << "without_table_free: " << no_table_free << "\n";
    yaml << "without_column_free: " << no_column_free << "\n";
    yaml << "penetration_depth: " << (any_depth ? "available" : "unavailable") << "\n";
    yaml << "root_cause_letter: " << letter << "\n";
    yaml << "acm_bug_suspected: " << ((part_touch_illegal || acm_bug) ? "true" : "false") << "\n";
    yaml << "attachment_bug_suspected: " << (attach_bug ? "true" : "false") << "\n";
    yaml << "part_touch_illegal_contact: " << (part_touch_illegal ? "true" : "false") << "\n";
    yaml << "part_wrist_forearm_contact: " << (part_wrist_forearm ? "true" : "false") << "\n";
    yaml << "p1: [" << p1.x() << ", " << p1.y() << ", " << p1.z() << "]\n";
    yaml << "d1: [" << d1.x() << ", " << d1.y() << ", " << d1.z() << "]\n";
    yaml << "preferred_up: [" << preferred_up.x() << ", " << preferred_up.y() << ", "
         << preferred_up.z() << "]\n";
    yaml << "t_tcp_object: \"" << formatPose(cfg.t_tcp_object) << "\"\n";
    yaml << "t_world_tcp_theory: \"" << formatPose(t_world_tcp_theory) << "\"\n";
    yaml << "t_world_object_theory: \"" << formatPose(t_world_object_theory) << "\"\n";
    yaml << "attached_shape: " << attached.shape << "\n";
    yaml << "attached_radius: " << attached.radius << "\n";
    yaml << "attached_height: " << attached.height << "\n";
    yaml << "attached_link: " << attached.attached_link << "\n";
    yaml << "attached_local_z_world_lift: \"" << formatVec(cyl_z) << "\"\n";
    yaml << "attached_local_z_world_canonical: \"" << formatVec(cyl_z_at_target) << "\"\n";
    yaml << "attached_local_z_vs_neg_y_deg: " << cyl_z_err_deg << "\n";
    yaml << "table_center: [" << getDouble(node, "table_cx") << ", " << getDouble(node, "table_cy")
         << ", " << getDouble(node, "table_cz") << "]\n";
    yaml << "table_dims: [" << getDouble(node, "table_dx") << ", " << getDouble(node, "table_dy")
         << ", " << getDouble(node, "table_dz") << "]\n";
    yaml << "column_center: [" << getDouble(node, "column_cx") << ", " << getDouble(node, "column_cy")
         << ", " << getDouble(node, "column_cz") << "]\n";
    yaml << "column_dims: [" << getDouble(node, "column_dx") << ", " << getDouble(node, "column_dy")
         << ", " << getDouble(node, "column_dz") << "]\n";
    yaml << "configured_touch_links:\n";
    for (const auto& link : touch_links)
    {
      yaml << "  - " << link << "\n";
    }
    yaml << "acm_entries:\n";
    for (const auto& row : acm_touch_rows)
    {
      yaml << "  - \"" << row << "\"\n";
    }
    yaml << "control_side_pos_y:\n";
    yaml << "  found: " << (side_pos ? "true" : "false") << "\n";
    yaml << "  valid: " << (side_pos && side_pos->valid ? "true" : "false") << "\n";
    yaml << "  collision: " << (side_pos_diff.full.collision ? "true" : "false") << "\n";
    if (side_pos)
    {
      yaml << "  candidate_id: " << side_pos->candidate_id << "\n";
    }
    yaml << "control_side_neg_y:\n";
    yaml << "  found: " << (side_neg ? "true" : "false") << "\n";
    yaml << "  valid: " << (side_neg && side_neg->valid ? "true" : "false") << "\n";
    yaml << "  collision: " << (side_neg_diff.full.collision ? "true" : "false") << "\n";
    if (side_neg)
    {
      yaml << "  candidate_id: " << side_neg->candidate_id << "\n";
    }
    yaml << "pair_frequency:\n";
    for (const auto& item : pair_sorted)
    {
      yaml << "  - pair: \"" << item.first << "\"\n";
      yaml << "    count: " << item.second << "\n";
      yaml << "    percent: " << (diags.empty() ? 0.0 : 100.0 * item.second / diags.size()) << "\n";
    }
    yaml << "category_ik_counts:\n";
    for (const auto& item : cat_ik)
    {
      yaml << "  " << collisionCategoryName(item.first) << ": " << item.second << "\n";
    }
    yaml << "rolls:\n";
    for (const auto& roll_item : by_roll)
    {
      std::map<std::string, int> local;
      size_t roll_hit = 0;
      size_t roll_free = 0;
      for (const auto idx : roll_item.second)
      {
        if (diags[idx].diff.full.collision)
        {
          ++roll_hit;
        }
        else
        {
          ++roll_free;
        }
        for (const auto& contact : diags[idx].diff.full.contacts)
        {
          local[contact.pair_key] += 1;
        }
      }
      std::string dom = "none";
      int dom_n = 0;
      for (const auto& item : local)
      {
        if (item.second > dom_n)
        {
          dom_n = item.second;
          dom = item.first;
        }
      }
      yaml << "  - roll_deg: " << roll_item.first << "\n";
      yaml << "    raw_ik: " << roll_item.second.size() << "\n";
      yaml << "    collision: " << roll_hit << "\n";
      yaml << "    free: " << roll_free << "\n";
      yaml << "    dominant_pair: \"" << dom << "\"\n";
    }
    yaml << "canonical_roll_0:\n";
    for (const auto& item : diags)
    {
      if (std::abs(item.cand.roll_deg) > 1e-9)
      {
        continue;
      }
      yaml << "  - ik_index: " << item.cand.ik_index << "\n";
      yaml << "    bounds_ok: " << (item.cand.bounds_ok ? "true" : "false") << "\n";
      yaml << "    joints:\n";
      for (const auto& name : kArmJoints)
      {
        yaml << "      " << name << ": " << item.cand.joints.at(name) << "\n";
      }
      yaml << "    full_collision: " << (item.diff.full.collision ? "true" : "false") << "\n";
      yaml << "    no_part_collision: " << (item.diff.no_part.collision ? "true" : "false") << "\n";
      yaml << "    self_collision: " << (item.diff.self_only.collision ? "true" : "false") << "\n";
      yaml << "    no_table_collision: " << (item.diff.no_table.collision ? "true" : "false")
           << "\n";
      yaml << "    no_column_collision: " << (item.diff.no_column.collision ? "true" : "false")
           << "\n";
      yaml << "    contacts:\n";
      for (const auto& contact : item.diff.full.contacts)
      {
        yaml << "      - pair: \"" << contact.pair_key << "\"\n";
        yaml << "        category: " << collisionCategoryName(contact.category) << "\n";
        yaml << "        count: " << contact.contact_count << "\n";
        if (contact.depth_available)
        {
          yaml << "        depth: " << contact.depth << "\n";
        }
        else
        {
          yaml << "        depth: unavailable\n";
        }
      }
    }
    yaml << "raw_iks:\n";
    for (const auto& item : diags)
    {
      yaml << "  - roll_deg: " << item.cand.roll_deg << "\n";
      yaml << "    raw_ik_index: " << item.cand.ik_index << "\n";
      yaml << "    bounds_ok: " << (item.cand.bounds_ok ? "true" : "false") << "\n";
      yaml << "    collision: " << (item.diff.full.collision ? "true" : "false") << "\n";
      yaml << "    contact_count: " << item.diff.full.contact_count << "\n";
      yaml << "    joints:\n";
      for (const auto& name : kArmJoints)
      {
        yaml << "      " << name << ": " << item.cand.joints.at(name) << "\n";
      }
      yaml << "    contacts:\n";
      for (const auto& contact : item.diff.full.contacts)
      {
        yaml << "      - pair: \"" << contact.pair_key << "\"\n";
        yaml << "        category: " << collisionCategoryName(contact.category) << "\n";
        yaml << "        count: " << contact.contact_count << "\n";
        yaml << "        depth: " << (contact.depth_available ? std::to_string(contact.depth) :
                                                               "unavailable")
             << "\n";
      }
      yaml << "    differential:\n";
      yaml << "      full: " << (item.diff.full.collision ? "true" : "false") << "\n";
      yaml << "      no_part: " << (item.diff.no_part.collision ? "true" : "false") << "\n";
      yaml << "      self_only: " << (item.diff.self_only.collision ? "true" : "false") << "\n";
      yaml << "      no_table: " << (item.diff.no_table.collision ? "true" : "false") << "\n";
      yaml << "      no_column: " << (item.diff.no_column.collision ? "true" : "false") << "\n";
    }
  }

  const auto after_msg = waitForFreshJoints(node, std::chrono::seconds(5));
  if (after_msg)
  {
    const double drift = fr_task_planner::maxJointError(jointsFromMsg(*after_msg), actual);
    RCLCPP_INFO(node->get_logger(), "Max joint drift: %.6f rad", drift);
    RCLCPP_INFO(node->get_logger(), "Robot moved because of this node? %s",
                drift > 0.02 ? "YES" : "NO");
  }
  bool live_world_after = false;
  bool live_attached_after = false;
  bool live_table_after = false;
  bool live_column_after = false;
  inspectLiveScene(node, object_id, live_world_after, live_attached_after, live_table_after,
                   live_column_after, table_name, column_name);
  RCLCPP_INFO(node->get_logger(), "LIVE SCENE world=%s attached=%s table=%s column=%s",
              live_world_after ? "YES" : "NO", live_attached_after ? "YES" : "NO",
              live_table_after ? "YES" : "NO", live_column_after ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "Wrote %s", output_path.c_str());

  const bool pass = !diags.empty() && side_pos_ok && side_neg_ok && !live_world_after &&
                    !live_attached_after && live_table_after && live_column_after &&
                    !attach_bug;
  shutdownSpinner(executor, spinner);
  return pass ? 0 : 4;
}
