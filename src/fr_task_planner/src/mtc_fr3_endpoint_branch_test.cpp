#include "fr_task_planner/inspection_endpoint_candidates.hpp"

#include <chrono>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <thread>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/collision_detection/collision_matrix.h>
#include <moveit/task_constructor/container.h>
#include <moveit/task_constructor/solvers/pipeline_planner.h>
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/modify_planning_scene.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/storage.h>
#include <moveit/task_constructor/task.h>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <moveit_task_constructor_msgs/msg/solution.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

namespace
{
using fr_task_planner::EndpointCandidate;
using fr_task_planner::EndpointGenerationConfig;
using fr_task_planner::RollPose;
using fr_task_planner::ViewGeom;
using fr_task_planner::applyJoints;
using fr_task_planner::capCandidates;
using fr_task_planner::filterRollsByView;
using fr_task_planner::generateValidEndpointCandidates;
using fr_task_planner::jointsFromState;
using fr_task_planner::kArmJoints;
using fr_task_planner::makeStageName;
using fr_task_planner::maxJointError;
using fr_task_planner::poseError;
using fr_task_planner::poseToIso;
using fr_task_planner::tcpInBase;

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

struct BranchRecord
{
  EndpointCandidate cand;
  bool attempted = true;
  bool complete = false;
  std::string failure = "OMPL transit failure";
  double duration = 0.0;
  double path = 0.0;
  double q_err = 0.0;
  double p1_err = 0.0;
  double d1_err = 0.0;
  double up_err = 0.0;
  size_t collisions = 0;
  size_t bounds = 0;
  bool attached = false;
  bool mapping_ok = true;
};

std::vector<std::map<std::string, double>> extractLastArmSegment(
    const moveit_task_constructor_msgs::msg::Solution& msg)
{
  std::vector<std::vector<std::map<std::string, double>>> segs;
  for (const auto& sub : msg.sub_trajectory)
  {
    std::vector<std::map<std::string, double>> points;
    const auto& jt = sub.trajectory.joint_trajectory;
    for (const auto& point : jt.points)
    {
      std::map<std::string, double> joints;
      const size_t n = std::min(jt.joint_names.size(), point.positions.size());
      for (size_t i = 0; i < n; ++i)
      {
        joints[jt.joint_names[i]] = point.positions[i];
      }
      bool has_arm = false;
      for (const auto& name : kArmJoints)
      {
        has_arm = has_arm || joints.count(name) != 0;
      }
      if (has_arm)
      {
        points.push_back(std::move(joints));
      }
    }
    if (!points.empty())
    {
      segs.push_back(std::move(points));
    }
  }
  if (segs.empty())
  {
    return {};
  }
  return segs.back();
}

double lastSegmentDuration(const moveit_task_constructor_msgs::msg::Solution& msg)
{
  double last = 0.0;
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
      last = rclcpp::Duration(jt.points.back().time_from_start).seconds();
    }
  }
  return last;
}

double pathLength(const std::vector<std::map<std::string, double>>& points)
{
  double acc = 0.0;
  for (size_t i = 1; i < points.size(); ++i)
  {
    acc += fr_task_planner::jointL2(points[i], points[i - 1]);
  }
  return acc;
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("fr3_mtc_endpoint_branch_test", options);
  RCLCPP_INFO(node->get_logger(), "THIS STEP IS PLAN-ONLY. NO TRAJECTORY EXECUTION IS PERFORMED.");
  RCLCPP_INFO(node->get_logger(), "STEP 11: one View, one Alternatives container. No order search.");

  const std::string view_name = getString(node, "view_name", "");
  if (view_name != "side_pos_y" && view_name != "side_neg_y" && view_name != "top_circle")
  {
    RCLCPP_ERROR(node->get_logger(), "view_name must be one required view, got %s", view_name.c_str());
    rclcpp::shutdown();
    return 1;
  }

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
  const int64_t max_endpoint_candidates =
      node->has_parameter("max_endpoint_candidates") ?
          node->get_parameter("max_endpoint_candidates").as_int() :
          0;
  const std::string output_path = getString(node, "diagnostic_output_path",
                                            "/tmp/fr3_step11_" + view_name + ".yaml");
  const bool hold = node->has_parameter("hold_for_introspection") &&
                    node->get_parameter("hold_for_introspection").as_bool();
  const ViewGeom view = readViewGeom(node, view_name);
  const auto all_rolls = filterRollsByView(readRollPoses(node), view_name);

  auto param_node = rclcpp::Node::make_shared("fr3_mtc_endpoint_branch_params");
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
  gen_task.setName("FR3 Endpoint Candidate Prefix");
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
    RCLCPP_ERROR(node->get_logger(), "candidate prefix init failed: %s", ex.what());
    shutdownSpinner(executor, spinner);
    return 1;
  }
  if (!gen_task.plan(1) || gen_task.numSolutions() < 1)
  {
    RCLCPP_ERROR(node->get_logger(), "candidate prefix planning failed");
    shutdownSpinner(executor, spinner);
    return 3;
  }
  std::vector<const moveit::task_constructor::SolutionBase*> gen_leaves;
  flattenSolutions(*gen_task.solutions().front(), gen_leaves);
  const auto* restore_sol = findStageSolution(gen_leaves, "Restore Part-Table Collision");
  if (!restore_sol || !restore_sol->end() || !restore_sol->end()->scene())
  {
    RCLCPP_ERROR(node->get_logger(), "candidate Restore scene missing");
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

  const auto generated =
      generateValidEndpointCandidates(*lift_scene, all_rolls, view, cfg, node->get_logger());
  auto candidates = capCandidates(generated.valid, static_cast<int>(max_endpoint_candidates));
  RCLCPP_INFO(node->get_logger(),
              "STEP 10 helper reused. view=%s rolls=%zu raw_ik=%zu valid=%zu used=%zu",
              view_name.c_str(), all_rolls.size(), generated.raw.size(), generated.valid.size(),
              candidates.size());
  if (candidates.empty())
  {
    RCLCPP_ERROR(node->get_logger(), "No valid endpoint candidates");
    shutdownSpinner(executor, spinner);
    return 4;
  }
  bool has_canonical = false;
  for (const auto& cand : candidates)
  {
    has_canonical = has_canonical || std::abs(cand.roll_deg) < 1e-9;
    RCLCPP_INFO(node->get_logger(), "branch child %s roll=%.1f ik=%d", cand.candidate_id.c_str(),
                cand.roll_deg, cand.ik_index);
  }

  moveit::task_constructor::Task task("", true);
  task.setName("FR3 Endpoint Branch Task " + view_name);
  task.loadRobotModel(node);
  addPrefixStages(task, node, group, ee_link, attach_link, object_id, table_name, touch_links,
                  object_scene, pregrasp, grasp, lift, object_height, object_radius, planning_time);
  auto ompl = std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
  ompl->setTimeout(planning_time);
  auto alts = std::make_unique<moveit::task_constructor::Alternatives>("Inspection Endpoints");
  alts->setPruning(false);
  for (const auto& cand : candidates)
  {
    auto move = std::make_unique<moveit::task_constructor::stages::MoveTo>(makeStageName(cand), ompl);
    move->setGroup(group);
    move->setGoal(cand.joints);
    move->setTimeout(planning_time);
    alts->add(std::move(move));
  }
  task.add(std::move(alts));
  RCLCPP_INFO(node->get_logger(),
              "========== OFFICIAL TASK ==========\nCurrentState → ... → Restore → "
              "Alternatives(%zu joint-goal OMPL children)\nONE shared prefix? YES",
              candidates.size());

  try
  {
    task.init();
  }
  catch (const std::exception& ex)
  {
    RCLCPP_ERROR(node->get_logger(), "official task.init() failed: %s", ex.what());
    shutdownSpinner(executor, spinner);
    return 1;
  }
  const size_t requested = candidates.size();
  const auto plan_ok = task.plan(requested);
  std::ostringstream state;
  task.printState(state);
  RCLCPP_INFO(node->get_logger(), "plan result=%s solutions=%zu requested=%zu",
              plan_ok ? "SUCCESS" : "FAIL", task.numSolutions(), requested);
  RCLCPP_INFO(node->get_logger(), "MTC task state:\n%s", state.str().c_str());

  std::map<std::string, BranchRecord> records;
  for (const auto& cand : candidates)
  {
    BranchRecord rec;
    rec.cand = cand;
    rec.attempted = true;
    rec.complete = false;
    rec.failure = "OMPL transit failure";
    records[cand.candidate_id] = rec;
  }

  size_t mapping_errors = 0;
  size_t illegal_all = 0;
  size_t bounds_all = 0;
  double max_q_err = 0.0;
  double max_p1 = 0.0;
  double max_d1 = 0.0;
  for (const auto& solution_ptr : task.solutions())
  {
    std::vector<const moveit::task_constructor::SolutionBase*> leaves;
    flattenSolutions(*solution_ptr, leaves);
    const EndpointCandidate* matched = nullptr;
    const moveit::task_constructor::SolutionBase* end_sol = nullptr;
    for (const auto& cand : candidates)
    {
      const auto* found = findStageSolution(leaves, makeStageName(cand));
      if (found)
      {
        matched = &cand;
        end_sol = found;
        break;
      }
    }
    if (!matched || !end_sol || !end_sol->end() || !end_sol->end()->scene())
    {
      ++mapping_errors;
      continue;
    }
    auto& rec = records[matched->candidate_id];
    rec.complete = true;
    rec.failure = "ok";
    rec.attached = end_sol->end()->scene()->getCurrentState().hasAttachedBody(object_id);
    const auto q_final = jointsFromState(end_sol->end()->scene()->getCurrentState());
    rec.q_err = maxJointError(q_final, matched->joints);
    if (rec.q_err > 1e-4)
    {
      rec.mapping_ok = false;
      ++mapping_errors;
    }
    moveit::core::RobotState fk(robot_model);
    fk.setToDefaultValues();
    applyJoints(fk, q_final);
    const Eigen::Isometry3d actual_tcp_base = tcpInBase(fk, ee_link);
    const Eigen::Isometry3d actual_object_world =
        cfg.t_model_base * actual_tcp_base * cfg.t_tcp_object;
    const Eigen::Vector3d actual_center =
        actual_object_world.translation() + actual_object_world.linear() * view.center_in_object;
    const Eigen::Vector3d actual_normal =
        (actual_object_world.linear() * view.normal_in_object).normalized();
    const Eigen::Vector3d actual_up = (actual_object_world.linear() * view.up_in_object).normalized();
    rec.p1_err = (actual_center - p1).norm();
    rec.d1_err =
        std::acos(std::min(1.0, std::max(-1.0, actual_normal.dot(d1)))) * 180.0 / M_PI;
    rec.up_err =
        std::acos(std::min(1.0, std::max(-1.0, actual_up.dot(preferred_up)))) * 180.0 / M_PI;
    max_q_err = std::max(max_q_err, rec.q_err);
    max_p1 = std::max(max_p1, rec.p1_err);
    max_d1 = std::max(max_d1, rec.d1_err);

    moveit_task_constructor_msgs::msg::Solution msg;
    solution_ptr->toMsg(msg);
    const auto pts = extractLastArmSegment(msg);
    rec.duration = lastSegmentDuration(msg);
    rec.path = pathLength(pts);
    auto* jmg = robot_model->getJointModelGroup(group);
    const auto* start_scene =
        end_sol->start() && end_sol->start()->scene() ? end_sol->start()->scene().get() : nullptr;
    for (const auto& q : pts)
    {
      applyJoints(fk, q);
      if (jmg && !fk.satisfiesBounds(jmg))
      {
        ++rec.bounds;
      }
      if (start_scene)
      {
        const auto diag = start_scene->diff();
        applyJoints(diag->getCurrentStateNonConst(), q);
        collision_detection::CollisionRequest req;
        collision_detection::CollisionResult res;
        diag->checkCollision(req, res);
        if (res.collision)
        {
          ++rec.collisions;
        }
      }
    }
    illegal_all += rec.collisions;
    bounds_all += rec.bounds;
    RCLCPP_INFO(node->get_logger(),
                "complete %s roll=%.1f ik=%d q_err=%.3e p1=%.3e d1=%.3e T=%.3f L=%.3f",
                matched->candidate_id.c_str(), matched->roll_deg, matched->ik_index, rec.q_err,
                rec.p1_err, rec.d1_err, rec.duration, rec.path);
  }

  std::set<std::string> reached_ids;
  std::set<int> reached_rolls;
  std::set<std::string> reached_ik;
  size_t complete_count = 0;
  bool canonical_reached = false;
  for (const auto& item : records)
  {
    if (!item.second.complete)
    {
      continue;
    }
    ++complete_count;
    reached_ids.insert(item.first);
    reached_rolls.insert(static_cast<int>(std::lround(item.second.cand.roll_deg)));
    reached_ik.insert(item.first);
    if (std::abs(item.second.cand.roll_deg) < 1e-9)
    {
      canonical_reached = true;
    }
  }
  std::string diversity = "NONE";
  if (complete_count == 0)
  {
    diversity = "NONE";
  }
  else if (complete_count == 1)
  {
    diversity = "LIMITED";
  }
  else if (reached_rolls.size() >= 2)
  {
    diversity = "READY";
  }
  else
  {
    diversity = "IK_ONLY";
  }

  {
    std::ofstream yaml(output_path);
    yaml.setf(std::ios::fixed);
    yaml.precision(6);
    yaml << "view_name: " << view_name << "\n";
    yaml << "endpoint_candidates: " << candidates.size() << "\n";
    yaml << "alternatives_children: " << candidates.size() << "\n";
    yaml << "requested_max_solutions: " << requested << "\n";
    yaml << "complete_solutions: " << complete_count << "\n";
    yaml << "unique_endpoint_ids: " << reached_ids.size() << "\n";
    yaml << "distinct_roll_angles: " << reached_rolls.size() << "\n";
    yaml << "canonical_roll_reachable: " << (canonical_reached ? "true" : "false") << "\n";
    yaml << "mapping_errors: " << mapping_errors << "\n";
    yaml << "max_final_joint_error: " << max_q_err << "\n";
    yaml << "max_p1_error: " << max_p1 << "\n";
    yaml << "max_d1_error: " << max_d1 << "\n";
    yaml << "illegal_collision_states: " << illegal_all << "\n";
    yaml << "bounds_violations: " << bounds_all << "\n";
    yaml << "diversity: " << diversity << "\n";
    yaml << "ranking: false\n";
    yaml << "execution: false\n";
    yaml << "branches:\n";
    for (const auto& cand : candidates)
    {
      const auto& rec = records.at(cand.candidate_id);
      yaml << "  - candidate_id: " << cand.candidate_id << "\n";
      yaml << "    digest: " << cand.digest << "\n";
      yaml << "    roll_deg: " << cand.roll_deg << "\n";
      yaml << "    ik_index: " << cand.ik_index << "\n";
      yaml << "    endpoint_valid: true\n";
      yaml << "    attempted: true\n";
      yaml << "    complete: " << (rec.complete ? "true" : "false") << "\n";
      yaml << "    duration: " << rec.duration << "\n";
      yaml << "    path: " << rec.path << "\n";
      yaml << "    q_err: " << rec.q_err << "\n";
      yaml << "    p1_err: " << rec.p1_err << "\n";
      yaml << "    d1_err: " << rec.d1_err << "\n";
      yaml << "    up_err: " << rec.up_err << "\n";
      yaml << "    attached: " << (rec.attached ? "true" : "false") << "\n";
      yaml << "    failure: " << rec.failure << "\n";
    }
  }

  const bool pass = complete_count >= 1 && canonical_reached && mapping_errors == 0 &&
                    illegal_all == 0 && bounds_all == 0 && max_p1 <= pos_tol &&
                    max_d1 <= ori_tol_deg;
  RCLCPP_INFO(node->get_logger(),
              "VIEW %s complete=%zu unique_ids=%zu rolls=%zu canonical=%s diversity=%s %s",
              view_name.c_str(), complete_count, reached_ids.size(), reached_rolls.size(),
              canonical_reached ? "YES" : "NO", diversity.c_str(), pass ? "PASS" : "FAIL");

  task.publishAllSolutions(false);
  const auto after_msg = waitForFreshJoints(node, std::chrono::seconds(5));
  if (after_msg)
  {
    const double drift = maxJointError(jointsFromMsg(*after_msg), actual);
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
  if (hold)
  {
    spinner.join();
  }
  else
  {
    shutdownSpinner(executor, spinner);
  }
  return pass ? 0 : 4;
}
