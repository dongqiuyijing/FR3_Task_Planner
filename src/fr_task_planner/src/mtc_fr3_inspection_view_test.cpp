#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/collision_detection/collision_common.h>
#include <moveit/collision_detection/collision_matrix.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/attached_body.h>
#include <moveit/robot_state/robot_state.h>
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
const char* kPlanningGroup = "fairino3_v6_group";
const char* kPlanningFrame = "base_link";
const char* kEeLink = "gripper_tcp";
const char* kOmplPipeline = "ompl";
const char* kPilzPipeline = "pilz_industrial_motion_planner";
const char* kPilzPlannerId = "LIN";
const std::vector<std::string> kArmJoints = { "j1", "j2", "j3", "j4", "j5", "j6" };
const std::vector<std::string> kArmBodyLinks = { "base_link",    "shoulder_link", "upperarm_link",
                                                 "forearm_link", "wrist1_link",   "wrist2_link",
                                                 "wrist3_link" };
constexpr double kLateralTolM = 0.001;
constexpr double kLinOriTolDeg = 0.5;
constexpr double kAttachPosTolM = 0.001;
constexpr double kAttachOriTolDeg = 0.5;

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
  if (!node->has_parameter("touch_links"))
  {
    throw std::runtime_error("Missing parameter: touch_links");
  }
  const auto parameter = node->get_parameter("touch_links");
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_STRING_ARRAY)
  {
    return parameter.as_string_array();
  }
  throw std::runtime_error("touch_links must be a string array, got type " +
                           std::to_string(static_cast<int>(parameter.get_type())));
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
    oss << name << "=" << it->second << " rad\n";
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

double maxJointError(const std::map<std::string, double>& a, const std::map<std::string, double>& b)
{
  double max_error = 0.0;
  for (const auto& name : kArmJoints)
  {
    max_error = std::max(max_error, std::abs(a.at(name) - b.at(name)));
  }
  return max_error;
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

std::map<std::string, double> jointsFromState(const moveit::core::RobotState& state)
{
  std::map<std::string, double> joints;
  for (const auto& name : kArmJoints)
  {
    joints[name] = state.getVariablePosition(name);
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

Eigen::Isometry3d tcpInBase(const moveit::core::RobotState& state, const std::string& ee_link)
{
  return state.getGlobalLinkTransform("base_link").inverse() * state.getGlobalLinkTransform(ee_link);
}

Eigen::Isometry3d tcpInWorld(const moveit::core::RobotState& state, const std::string& ee_link,
                             const std::string& model_frame, const Eigen::Isometry3d& t_world_base)
{
  const Eigen::Isometry3d t_model_tcp = state.getGlobalLinkTransform(ee_link);
  if (model_frame == "world")
  {
    return t_model_tcp;
  }
  if (model_frame == "base_link")
  {
    return t_world_base * t_model_tcp;
  }
  return t_world_base * state.getGlobalLinkTransform("base_link").inverse() * t_model_tcp;
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

struct TrajectorySegment
{
  std::string planner_id;
  std::string comment;
  std::vector<std::map<std::string, double>> points;
};

std::vector<TrajectorySegment> extractArmSegments(
    const moveit_task_constructor_msgs::msg::Solution& solution_msg)
{
  std::vector<TrajectorySegment> segments;
  for (const auto& sub : solution_msg.sub_trajectory)
  {
    TrajectorySegment segment;
    segment.planner_id = sub.info.planner_id;
    segment.comment = sub.info.comment;
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
        segment.points.push_back(std::move(joints));
      }
    }
    if (!segment.points.empty())
    {
      segments.push_back(std::move(segment));
    }
  }
  return segments;
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

bool acmAllowed(const planning_scene::PlanningScene& scene, const std::string& a,
                const std::string& b)
{
  collision_detection::AllowedCollision::Type type;
  return scene.getAllowedCollisionMatrix().getEntry(a, b, type) &&
         type == collision_detection::AllowedCollision::ALWAYS;
}

bool isTouchLink(const std::string& link, const std::vector<std::string>& touch_links)
{
  return std::find(touch_links.begin(), touch_links.end(), link) != touch_links.end();
}

bool isArmBodyLink(const std::string& link)
{
  return std::find(kArmBodyLinks.begin(), kArmBodyLinks.end(), link) != kArmBodyLinks.end();
}

struct ObjectPairContact
{
  std::string other;
  double depth = 0.0;
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
};

std::vector<ObjectPairContact> rawObjectContacts(const planning_scene::PlanningScene& scene,
                                                 const std::string& object_id,
                                                 const std::string& table_name,
                                                 const std::vector<std::string>& touch_links)
{
  const auto diag = scene.diff();
  auto& acm = diag->getAllowedCollisionMatrixNonConst();
  if (!touch_links.empty())
  {
    acm.setEntry(object_id, touch_links, false);
  }
  acm.setEntry(object_id, table_name, true);
  collision_detection::CollisionRequest req;
  req.contacts = true;
  req.max_contacts = 50;
  req.max_contacts_per_pair = 3;
  collision_detection::CollisionResult res;
  diag->checkCollision(req, res);

  std::vector<ObjectPairContact> out;
  for (const auto& item : res.contacts)
  {
    const std::string& a = item.first.first;
    const std::string& b = item.first.second;
    if (a != object_id && b != object_id)
    {
      continue;
    }
    const std::string other = (a == object_id) ? b : a;
    if (other == table_name)
    {
      continue;
    }
    ObjectPairContact contact;
    contact.other = other;
    if (!item.second.empty())
    {
      contact.depth = item.second.front().depth;
      contact.pos = item.second.front().pos;
    }
    out.push_back(contact);
  }
  return out;
}

void classifyObjectContacts(const std::vector<ObjectPairContact>& contacts,
                            const std::vector<std::string>& touch_links,
                            std::vector<ObjectPairContact>& allowed,
                            std::vector<ObjectPairContact>& illegal)
{
  allowed.clear();
  illegal.clear();
  for (const auto& contact : contacts)
  {
    if (isTouchLink(contact.other, touch_links))
    {
      allowed.push_back(contact);
    }
    else
    {
      illegal.push_back(contact);
    }
  }
}

void logContacts(const rclcpp::Logger& logger, const std::string& title,
                 const std::vector<ObjectPairContact>& contacts)
{
  if (contacts.empty())
  {
    RCLCPP_INFO(logger, "%s\nNONE", title.c_str());
    return;
  }
  RCLCPP_INFO(logger, "%s", title.c_str());
  for (const auto& contact : contacts)
  {
    RCLCPP_INFO(logger, "  small_part <-> %s  depth=%.6f m  pos=(%.6f, %.6f, %.6f)",
                contact.other.c_str(), contact.depth, contact.pos.x(), contact.pos.y(),
                contact.pos.z());
  }
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
    RCLCPP_ERROR(target->get_logger(), "Timed out waiting for /move_group. Start stage4_full first.");
    return false;
  }
  const std::vector<std::string> names = { "robot_description", "robot_description_semantic" };
  const auto values = client->get_parameters(names);
  for (const auto& parameter : values)
  {
    if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET)
    {
      RCLCPP_ERROR(target->get_logger(), "/move_group is missing parameter '%s'",
                   parameter.get_name().c_str());
      return false;
    }
    const auto value = parameter.as_string();
    setStringParam(target, parameter.get_name(), value);
    if (parameter.get_name() == "robot_description")
    {
      const bool has_world = value.find("name=\"world\"") != std::string::npos ||
                             value.find("name='world'") != std::string::npos;
      RCLCPP_INFO(target->get_logger(),
                  "Overlaid robot_description size=%zu world_link=%s", value.size(),
                  has_world ? "YES" : "NO");
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

void inspectLiveScene(const rclcpp::Node::SharedPtr& node, const std::string& object_id,
                      bool& world_present, bool& attached_present)
{
  world_present = false;
  attached_present = false;
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
    if (obj.id == object_id)
    {
      world_present = true;
    }
  }
  for (const auto& obj : response->scene.robot_state.attached_collision_objects)
  {
    if (obj.object.id == object_id)
    {
      attached_present = true;
    }
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

std::string diagnoseInspectFailure(const moveit::core::RobotModelConstPtr& robot_model,
                                   const std::string& group, const std::string& ee_link,
                                   const std::string& attach_link, const std::string& object_id,
                                   const std::vector<std::string>& touch_links,
                                   const geometry_msgs::msg::PoseStamped& inspect,
                                   const std::map<std::string, double>& seed)
{
  auto* jmg = robot_model->getJointModelGroup(group);
  if (!jmg)
  {
    return "OTHER";
  }
  moveit::core::RobotState ik_state(robot_model);
  ik_state.setToDefaultValues();
  applyJoints(ik_state, seed);
  const bool ik_ok = ik_state.setFromIK(jmg, inspect.pose, ee_link, 0.2);
  if (!ik_ok)
  {
    return "TARGET IK FAILURE";
  }
  planning_scene::PlanningScene scene(robot_model);
  scene.getCurrentStateNonConst() = ik_state;
  moveit_msgs::msg::AttachedCollisionObject attached;
  attached.link_name = attach_link;
  attached.object.id = object_id;
  attached.object.operation = moveit_msgs::msg::CollisionObject::ADD;
  attached.touch_links = touch_links;
  scene.processAttachedCollisionObjectMsg(attached);
  collision_detection::CollisionRequest req;
  collision_detection::CollisionResult res;
  req.contacts = true;
  req.max_contacts = 20;
  scene.checkCollision(req, res);
  if (res.collision)
  {
    return "TARGET COLLISION";
  }
  return "OMPL TRANSIT FAILURE";
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("fr3_mtc_inspection_view_test", options);
  RCLCPP_INFO(node->get_logger(), "THIS STEP IS PLAN-ONLY. NO TRAJECTORY EXECUTION IS PERFORMED.");
  RCLCPP_INFO(node->get_logger(), "Physical gripper close: NOT EXECUTED");
  RCLCPP_INFO(node->get_logger(), "Predicted grasp scene transition: YES");
  RCLCPP_INFO(node->get_logger(), "STEP 7: one view reachability. No order search.");

  const auto home = readHome(node);
  const auto pregrasp = readPose(node, "pregrasp");
  const auto grasp = readPose(node, "grasp");
  const auto lift = readPose(node, "lift");
  const auto object_world = readPose(node, "object_world");
  const auto object_base = readPose(node, "object");
  const auto world_base = readPose(node, "world_base");
  const auto tcp_object = readPose(node, "tcp_object");
  const auto inspect = readPose(node, "inspect");
  const auto inspect_object = readPose(node, "inspect_object");
  const std::string view_name = getString(node, "view_name", "");
  const Eigen::Vector3d p1(getDouble(node, "p1_x"), getDouble(node, "p1_y"), getDouble(node, "p1_z"));
  const Eigen::Vector3d d1 =
      Eigen::Vector3d(getDouble(node, "d1_x"), getDouble(node, "d1_y"), getDouble(node, "d1_z"))
          .normalized();
  const Eigen::Vector3d preferred_up =
      Eigen::Vector3d(getDouble(node, "preferred_up_x"), getDouble(node, "preferred_up_y"),
                      getDouble(node, "preferred_up_z"))
          .normalized();
  const Eigen::Vector3d center_in_object(getDouble(node, "center_in_object_x"),
                                         getDouble(node, "center_in_object_y"),
                                         getDouble(node, "center_in_object_z"));
  const Eigen::Vector3d normal_in_object(getDouble(node, "normal_in_object_x"),
                                         getDouble(node, "normal_in_object_y"),
                                         getDouble(node, "normal_in_object_z"));
  const Eigen::Vector3d up_in_object(getDouble(node, "up_in_object_x"),
                                     getDouble(node, "up_in_object_y"),
                                     getDouble(node, "up_in_object_z"));
  const std::string group = getString(node, "planning_group", kPlanningGroup);
  const std::string planning_frame = getString(node, "planning_frame", kPlanningFrame);
  const std::string ee_link = getString(node, "ee_link", kEeLink);
  const std::string attach_link = getString(node, "attach_link", kEeLink);
  const std::string object_id = getString(node, "object_name", "small_part");
  const std::string table_name = getString(node, "table_name", "table");
  const std::string object_shape = getString(node, "object_shape", "cylinder");
  const auto touch_links = readTouchLinks(node);
  const double object_height = getDouble(node, "object_height");
  const double object_radius = getDouble(node, "object_radius");
  const double lift_distance = getDouble(node, "lift_distance");
  const double home_tol = node->has_parameter("max_home_error_rad") ?
                              node->get_parameter("max_home_error_rad").as_double() :
                              0.03;
  const double pos_tol = getDouble(node, "position_tolerance");
  const double ori_tol_deg = getDouble(node, "orientation_tolerance_deg");
  const double planning_time = getDouble(node, "planning_time");
  const double up_tol_deg = node->has_parameter("max_up_angle_error_deg") ?
                                node->get_parameter("max_up_angle_error_deg").as_double() :
                                5.0;
  const double view_center_tol = node->has_parameter("view_center_tolerance") ?
                                     node->get_parameter("view_center_tolerance").as_double() :
                                     0.005;
  const int64_t max_solutions =
      node->has_parameter("max_solutions") ? node->get_parameter("max_solutions").as_int() : 5;
  if (view_name != "side_pos_y" && view_name != "side_neg_y" && view_name != "top_circle")
  {
    RCLCPP_ERROR(node->get_logger(), "view_name must be side_pos_y|side_neg_y|top_circle, got '%s'",
                 view_name.c_str());
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "Parameters loaded. Overlaying robot_description from /move_group.");
  auto param_node = rclcpp::Node::make_shared("fr3_mtc_inspection_view_params");
  if (!overlayRobotDescriptionFromMoveGroup(node, param_node))
  {
    rclcpp::shutdown();
    return 1;
  }
  RCLCPP_INFO(node->get_logger(), "robot_description overlay complete.");

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  bool live_world_before = false;
  bool live_attached_before = false;
  inspectLiveScene(node, object_id, live_world_before, live_attached_before);

  const auto before_msg = waitForFreshJoints(node, std::chrono::seconds(10));
  if (!before_msg)
  {
    RCLCPP_ERROR(node->get_logger(), "/joint_states missing. STEP 7 FAIL.");
    shutdownSpinner(executor, spinner);
    return 1;
  }
  const auto actual = jointsFromMsg(*before_msg);
  if (jointsAreAllZero(actual))
  {
    RCLCPP_ERROR(node->get_logger(), "CurrentState is all zeros. STEP 7 FAIL.");
    shutdownSpinner(executor, spinner);
    return 1;
  }

  double max_home_error = 0.0;
  for (const auto& name : kArmJoints)
  {
    max_home_error = std::max(max_home_error, std::abs(actual.at(name) - home.at(name)));
  }

  RCLCPP_INFO(node->get_logger(), "========== STEP 7 PREFLIGHT ==========");
  RCLCPP_INFO(node->get_logger(), "View name:\n%s", view_name.c_str());
  RCLCPP_INFO(node->get_logger(), "Canonical inspect TCP:\n%s", formatPose(inspect).c_str());
  RCLCPP_INFO(node->get_logger(), "Canonical inspect object:\n%s", formatPose(inspect_object).c_str());
  RCLCPP_INFO(node->get_logger(), "Current joints:\n%s", formatJoints(actual).c_str());
  RCLCPP_INFO(node->get_logger(), "Expected Home:\n%s", formatJoints(home).c_str());
  RCLCPP_INFO(node->get_logger(), "Max Home error: %.6f rad (tol=%.3f) %s", max_home_error, home_tol,
              max_home_error <= home_tol ? "PASS" : "FAIL");
  RCLCPP_INFO(node->get_logger(), "PreGrasp:\n%s", formatPose(pregrasp).c_str());
  RCLCPP_INFO(node->get_logger(), "Grasp:\n%s", formatPose(grasp).c_str());
  RCLCPP_INFO(node->get_logger(), "Lift:\n%s", formatPose(lift).c_str());
  RCLCPP_INFO(node->get_logger(), "Lift distance from YAML:\n%.6f", lift_distance);
  {
    std::ostringstream yaml_touch;
    std::vector<std::string> arm_allowed;
    for (const auto& link : touch_links)
    {
      if (!yaml_touch.str().empty())
      {
        yaml_touch << ", ";
      }
      yaml_touch << link;
      if (isArmBodyLink(link))
      {
        arm_allowed.push_back(link);
      }
    }
    RCLCPP_INFO(node->get_logger(), "========== STEP 5A TOUCH LINKS ==========");
    RCLCPP_INFO(node->get_logger(), "YAML gripper_touch_links:\n%s", yaml_touch.str().c_str());
    RCLCPP_INFO(node->get_logger(),
                "Arm links incorrectly allowed:\n%s",
                arm_allowed.empty() ? "NONE" : yaml_touch.str().c_str());
  }
  if (max_home_error > home_tol)
  {
    RCLCPP_ERROR(node->get_logger(), "Current is not Stage4 Home. Planning aborted.");
    shutdownSpinner(executor, spinner);
    return 2;
  }

  moveit::task_constructor::Task task("", true);
  task.setName("FR3 Inspection Reachability Task");
  task.loadRobotModel(node);
  const auto robot_model = task.getRobotModel();
  if (!robot_model || !robot_model->hasJointModelGroup(group) || !robot_model->hasLinkModel(ee_link))
  {
    RCLCPP_ERROR(node->get_logger(), "Robot model / group / TCP missing");
    shutdownSpinner(executor, spinner);
    return 1;
  }
  const std::string model_frame = robot_model->getModelFrame();
  const Eigen::Isometry3d t_world_base = poseToIso(world_base.pose);
  geometry_msgs::msg::PoseStamped object_scene = object_world;
  if (model_frame != object_world.header.frame_id)
  {
    object_scene = object_base;
    object_scene.header.frame_id = model_frame == "base_link" ? object_base.header.frame_id : model_frame;
  }
  RCLCPP_INFO(node->get_logger(), "RobotModel model frame: %s", model_frame.c_str());
  RCLCPP_INFO(node->get_logger(), "Object CollisionObject frame: %s", object_scene.header.frame_id.c_str());

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

  auto allow_touch = std::make_unique<moveit::task_constructor::stages::ModifyPlanningScene>(
      "Allow Gripper-Part Contact");
  allow_touch->allowCollisions(object_id, touch_links, true);
  task.add(std::move(allow_touch));

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
  // GRASP_COMMIT: predicted attach only. Physical gripper close is not executed.
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

  auto ompl_inspect =
      std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
  ompl_inspect->setTimeout(planning_time);
  const std::string inspect_stage_name = "MoveTo Inspection View " + view_name;
  auto move_inspect =
      std::make_unique<moveit::task_constructor::stages::MoveTo>(inspect_stage_name, ompl_inspect);
  move_inspect->setGroup(group);
  move_inspect->setIKFrame(ee_link);
  move_inspect->setGoal(inspect);
  move_inspect->setTimeout(planning_time);
  task.add(std::move(move_inspect));

  RCLCPP_INFO(node->get_logger(), "========== MTC TASK ==========");
  RCLCPP_INFO(node->get_logger(),
              "CurrentState\n↓ Prepare Object On Table\n↓ MoveTo PreGrasp / OMPL\n↓ Allow "
              "Gripper-Part Contact\n↓ MoveTo Grasp / Pilz LIN\n↓ Attach Part To TCP\n↓ MoveTo "
              "Lift / Pilz LIN\n↓ Restore Part-Table Collision\n↓ %s / OMPL",
              inspect_stage_name.c_str());
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

  const auto plan_result = task.plan(static_cast<size_t>(std::max<int64_t>(1, max_solutions)));
  const size_t num_solutions = task.numSolutions();
  std::ostringstream state;
  task.printState(state);
  RCLCPP_INFO(node->get_logger(), "planning result: %s", plan_result ? "SUCCESS" : "FAIL");
  RCLCPP_INFO(node->get_logger(), "Complete solution count: %zu", num_solutions);
  RCLCPP_INFO(node->get_logger(), "MTC task state:\n%s", state.str().c_str());
  if (!plan_result || num_solutions < 1)
  {
    std::ostringstream failure;
    task.explainFailure(failure);
    RCLCPP_ERROR(node->get_logger(), "planning failure:\n%s", failure.str().c_str());
    RCLCPP_ERROR(node->get_logger(), "Complete Solution: NO");
    const std::string cls =
        diagnoseInspectFailure(robot_model, group, ee_link, attach_link, object_id, touch_links,
                               inspect, actual);
    RCLCPP_ERROR(node->get_logger(), "FAILURE CLASSIFICATION:\n%s", cls.c_str());
    RCLCPP_ERROR(node->get_logger(), "Target IK exists: %s",
                 cls == "TARGET IK FAILURE" ? "NO" : "YES");
    RCLCPP_ERROR(node->get_logger(), "Target collision-free: %s",
                 cls == "TARGET COLLISION" ? "NO" : (cls == "TARGET IK FAILURE" ? "UNKNOWN" : "YES"));
    task.publishAllSolutions(false);
    shutdownSpinner(executor, spinner);
    return 3;
  }

  const auto& solution = *task.solutions().front();
  std::vector<const moveit::task_constructor::SolutionBase*> leaves;
  flattenSolutions(solution, leaves);
  const auto* prepare_sol = findStageSolution(leaves, "Prepare Object On Table");
  const auto* pregrasp_sol = findStageSolution(leaves, "MoveTo PreGrasp");
  const auto* allow_sol = findStageSolution(leaves, "Allow Gripper-Part Contact");
  const auto* grasp_sol = findStageSolution(leaves, "MoveTo Grasp");
  const auto* attach_sol = findStageSolution(leaves, "Attach Part To TCP");
  const auto* lift_sol = findStageSolution(leaves, "MoveTo Lift");
  const auto* restore_sol = findStageSolution(leaves, "Restore Part-Table Collision");
  const auto* inspect_sol = findStageSolution(leaves, inspect_stage_name);

  RCLCPP_INFO(node->get_logger(), "========== OBJECT SCENE VALIDATION ==========");
  RCLCPP_INFO(node->get_logger(), "Object ID:\n%s", object_id.c_str());
  RCLCPP_INFO(node->get_logger(), "Shape:\n%s", object_shape.c_str());
  RCLCPP_INFO(node->get_logger(), "Height:\n%.6f", object_height);
  RCLCPP_INFO(node->get_logger(), "Radius:\n%.6f", object_radius);
  RCLCPP_INFO(node->get_logger(), "Initial frame:\n%s", object_scene.header.frame_id.c_str());
  RCLCPP_INFO(node->get_logger(), "Initial pose:\n%s", formatPose(object_scene).c_str());
  RCLCPP_INFO(node->get_logger(), "YAML world pose:\n%s", formatPose(object_world).c_str());
  bool part_table_allowed = false;
  bool part_touch_at_home = false;
  if (prepare_sol && prepare_sol->end() && prepare_sol->end()->scene())
  {
    const auto& scene = *prepare_sol->end()->scene();
    part_table_allowed = acmAllowed(scene, object_id, table_name);
    for (const auto& link : touch_links)
    {
      part_touch_at_home = part_touch_at_home || acmAllowed(scene, object_id, link);
    }
    const bool world_obj = static_cast<bool>(scene.getWorld()->getObject(object_id));
    RCLCPP_INFO(node->get_logger(), "World object after prepare: %s", world_obj ? "YES" : "NO");
    collision_detection::CollisionRequest req;
    collision_detection::CollisionResult res;
    req.contacts = true;
    req.max_contacts = 20;
    scene.checkCollision(req, res);
    RCLCPP_INFO(node->get_logger(), "Home+object collision: %s", res.collision ? "YES" : "NO");
    for (const auto& contact : res.contacts)
    {
      RCLCPP_INFO(node->get_logger(), "  contact %s <-> %s", contact.first.first.c_str(),
                  contact.first.second.c_str());
    }
  }
  RCLCPP_INFO(node->get_logger(), "Part-table collision allowed:\n%s",
              part_table_allowed ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "Part-gripper contact at Home:\n%s",
              part_touch_at_home ? "YES" : "NO");

  bool part_touch_before_grasp = false;
  if (allow_sol && allow_sol->end() && allow_sol->end()->scene())
  {
    const auto& scene = *allow_sol->end()->scene();
    for (const auto& link : touch_links)
    {
      part_touch_before_grasp = part_touch_before_grasp || acmAllowed(scene, object_id, link);
    }
  }
  RCLCPP_INFO(node->get_logger(), "Part-gripper contact before LIN grasp:\n%s",
              part_touch_before_grasp ? "YES" : "NO");

  moveit_task_constructor_msgs::msg::Solution solution_msg;
  solution.toMsg(solution_msg);
  const auto segments = extractArmSegments(solution_msg);
  RCLCPP_INFO(node->get_logger(), "arm trajectory segments: %zu", segments.size());
  if (segments.size() < 4)
  {
    RCLCPP_ERROR(node->get_logger(), "Need OMPL + Grasp LIN + Lift LIN + Inspection OMPL segments");
    shutdownSpinner(executor, spinner);
    return 3;
  }
  const auto& ompl_seg = segments[0];
  const auto& grasp_seg = segments[1];
  const auto& lift_seg = segments[2];
  const auto& inspect_seg = segments[3];

  moveit::core::RobotState fk_state(robot_model);
  fk_state.setToDefaultValues();
  applyJoints(fk_state, actual);
  const Eigen::Isometry3d t_pre_target = poseToIso(pregrasp.pose);
  const Eigen::Isometry3d t_grasp_target = poseToIso(grasp.pose);
  const Eigen::Isometry3d t_lift_target = poseToIso(lift.pose);

  applyJoints(fk_state, ompl_seg.points.back());
  double pre_pos = 0.0;
  double pre_ori = 0.0;
  poseError(t_pre_target, tcpInBase(fk_state, ee_link), pre_pos, pre_ori);
  const bool pregrasp_world =
      pregrasp_sol && pregrasp_sol->end() && pregrasp_sol->end()->scene() &&
      static_cast<bool>(pregrasp_sol->end()->scene()->getWorld()->getObject(object_id)) &&
      !pregrasp_sol->end()->scene()->getCurrentState().hasAttachedBody(object_id);
  RCLCPP_INFO(node->get_logger(), "========== PREGRASP ==========");
  RCLCPP_INFO(node->get_logger(), "OMPL points: %zu", ompl_seg.points.size());
  RCLCPP_INFO(node->get_logger(), "PreGrasp position error: %.6f m", pre_pos);
  RCLCPP_INFO(node->get_logger(), "PreGrasp orientation error: %.6f deg", pre_ori);
  RCLCPP_INFO(node->get_logger(), "object state: %s", pregrasp_world ? "WORLD" : "UNEXPECTED");

  applyJoints(fk_state, grasp_seg.points.back());
  double grasp_pos = 0.0;
  double grasp_ori = 0.0;
  poseError(t_grasp_target, tcpInBase(fk_state, ee_link), grasp_pos, grasp_ori);
  const Eigen::Vector3d pre_p = t_pre_target.translation();
  const Eigen::Vector3d grasp_p = t_grasp_target.translation();
  const Eigen::Vector3d grasp_line = grasp_p - pre_p;
  double grasp_lateral = 0.0;
  double grasp_path = 0.0;
  for (size_t i = 0; i < grasp_seg.points.size(); ++i)
  {
    applyJoints(fk_state, grasp_seg.points[i]);
    const Eigen::Vector3d p = tcpInBase(fk_state, ee_link).translation();
    if (i + 1 < grasp_seg.points.size())
    {
      applyJoints(fk_state, grasp_seg.points[i + 1]);
      grasp_path += (tcpInBase(fk_state, ee_link).translation() - p).norm();
    }
    const Eigen::Vector3d rel = p - pre_p;
    if (grasp_line.norm() > 1e-9)
    {
      grasp_lateral = std::max(
          grasp_lateral,
          (rel - grasp_line * (rel.dot(grasp_line) / grasp_line.squaredNorm())).norm());
    }
  }
  RCLCPP_INFO(node->get_logger(), "========== GRASP LIN ==========");
  RCLCPP_INFO(node->get_logger(), "planner: Pilz LIN id='%s'", grasp_seg.planner_id.c_str());
  RCLCPP_INFO(node->get_logger(), "trajectory points: %zu", grasp_seg.points.size());
  RCLCPP_INFO(node->get_logger(), "Cartesian path length: %.6f m", grasp_path);
  RCLCPP_INFO(node->get_logger(), "max lateral deviation: %.6f m", grasp_lateral);
  RCLCPP_INFO(node->get_logger(), "endpoint position error: %.6f m", grasp_pos);
  RCLCPP_INFO(node->get_logger(), "endpoint orientation error: %.6f deg", grasp_ori);

  bool before_world = false;
  bool before_attached = false;
  bool after_world = false;
  bool after_attached = false;
  std::string attached_link;
  Eigen::Isometry3d actual_tcp_object = Eigen::Isometry3d::Identity();
  std::set<std::string> actual_touch;
  if (grasp_sol && grasp_sol->end() && grasp_sol->end()->scene())
  {
    const auto& scene = *grasp_sol->end()->scene();
    before_world = static_cast<bool>(scene.getWorld()->getObject(object_id));
    before_attached = scene.getCurrentState().hasAttachedBody(object_id);
  }
  if (attach_sol && attach_sol->end() && attach_sol->end()->scene())
  {
    const auto& scene = *attach_sol->end()->scene();
    after_world = static_cast<bool>(scene.getWorld()->getObject(object_id));
    after_attached = scene.getCurrentState().hasAttachedBody(object_id);
    if (after_attached)
    {
      const auto* body = scene.getCurrentState().getAttachedBody(object_id);
      attached_link = body->getAttachedLinkName();
      actual_tcp_object = body->getPose();
      actual_touch = body->getTouchLinks();
    }
  }
  const Eigen::Isometry3d expected_tcp_object = poseToIso(tcp_object.pose);
  double attach_pos = 0.0;
  double attach_ori = 0.0;
  poseError(expected_tcp_object, actual_tcp_object, attach_pos, attach_ori);
  std::ostringstream touch_log;
  for (const auto& link : touch_links)
  {
    if (!touch_log.str().empty())
    {
      touch_log << ", ";
    }
    touch_log << link;
  }
  RCLCPP_INFO(node->get_logger(), "========== ATTACH VALIDATION ==========");
  RCLCPP_INFO(node->get_logger(), "Before attach world=%s attached=%s", before_world ? "YES" : "NO",
              before_attached ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "After attach world=%s attached=%s", after_world ? "YES" : "NO",
              after_attached ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "Attached link:\n%s", attached_link.c_str());
  RCLCPP_INFO(node->get_logger(), "Configured touch links:\n%s", touch_log.str().c_str());
  std::ostringstream actual_touch_log;
  for (const auto& link : actual_touch)
  {
    if (!actual_touch_log.str().empty())
    {
      actual_touch_log << ", ";
    }
    actual_touch_log << link;
  }
  RCLCPP_INFO(node->get_logger(), "Actual attached touch links:\n%s", actual_touch_log.str().c_str());
  RCLCPP_INFO(node->get_logger(), "Expected T_tcp_object:\n%s",
              formatIso(expected_tcp_object, attach_link).c_str());
  RCLCPP_INFO(node->get_logger(), "Actual attached T_tcp_object:\n%s",
              formatIso(actual_tcp_object, attached_link.empty() ? attach_link : attached_link)
                  .c_str());
  RCLCPP_INFO(node->get_logger(), "Translation error: %.6f m", attach_pos);
  RCLCPP_INFO(node->get_logger(), "Orientation error: %.6f deg", attach_ori);

  std::vector<ObjectPairContact> grasp_allowed;
  std::vector<ObjectPairContact> grasp_illegal;
  if (grasp_sol && grasp_sol->end() && grasp_sol->end()->scene())
  {
    classifyObjectContacts(
        rawObjectContacts(*grasp_sol->end()->scene(), object_id, table_name, touch_links),
        touch_links, grasp_allowed, grasp_illegal);
  }
  RCLCPP_INFO(node->get_logger(), "========== GRASP COLLISION PAIRS ==========");
  logContacts(node->get_logger(), "Allowed:", grasp_allowed);
  logContacts(node->get_logger(), "Illegal:", grasp_illegal);
  if (!grasp_illegal.empty())
  {
    RCLCPP_ERROR(node->get_logger(), "Grasp robot state:\n%s",
                 formatJoints(jointsFromState(grasp_sol->end()->scene()->getCurrentState())).c_str());
    RCLCPP_ERROR(node->get_logger(), "Object relative pose:\n%s",
                 formatIso(expected_tcp_object, attach_link).c_str());
  }

  std::vector<ObjectPairContact> attach_allowed;
  std::vector<ObjectPairContact> attach_illegal;
  if (attach_sol && attach_sol->end() && attach_sol->end()->scene())
  {
    classifyObjectContacts(
        rawObjectContacts(*attach_sol->end()->scene(), object_id, table_name, touch_links),
        touch_links, attach_allowed, attach_illegal);
  }
  RCLCPP_INFO(node->get_logger(), "========== ATTACHED STATE COLLISION ==========");
  logContacts(node->get_logger(), "Allowed attached contacts:", attach_allowed);
  logContacts(node->get_logger(), "Illegal attached-object robot collisions:", attach_illegal);

  bool lift_attached = false;
  if (lift_sol && lift_sol->start() && lift_sol->start()->scene())
  {
    const auto& scene = *lift_sol->start()->scene();
    lift_attached = scene.getCurrentState().hasAttachedBody(object_id) &&
                    !static_cast<bool>(scene.getWorld()->getObject(object_id));
  }

  applyJoints(fk_state, lift_seg.points.front());
  const Eigen::Isometry3d lift_start_tf = tcpInWorld(fk_state, ee_link, model_frame, t_world_base);
  const Eigen::Vector3d lift_start_world = lift_start_tf.translation();
  const Eigen::Quaterniond lift_start_q(lift_start_tf.rotation());
  applyJoints(fk_state, lift_seg.points.back());
  const Eigen::Isometry3d lift_end_tf = tcpInWorld(fk_state, ee_link, model_frame, t_world_base);
  const Eigen::Vector3d lift_end_world = lift_end_tf.translation();
  double lift_pos = 0.0;
  double lift_ori = 0.0;
  poseError(t_lift_target, tcpInBase(fk_state, ee_link), lift_pos, lift_ori);
  const Eigen::Vector3d lift_delta = lift_end_world - lift_start_world;
  double lift_path = 0.0;
  double lift_lateral = 0.0;
  double lift_ori_dev = 0.0;
  Eigen::Vector3d prev_world = lift_start_world;
  for (const auto& joints : lift_seg.points)
  {
    applyJoints(fk_state, joints);
    const Eigen::Isometry3d tcp_world = tcpInWorld(fk_state, ee_link, model_frame, t_world_base);
    lift_path += (tcp_world.translation() - prev_world).norm();
    prev_world = tcp_world.translation();
    const double dx = tcp_world.translation().x() - lift_start_world.x();
    const double dy = tcp_world.translation().y() - lift_start_world.y();
    lift_lateral = std::max(lift_lateral, std::sqrt(dx * dx + dy * dy));
    const Eigen::Quaterniond q(tcp_world.rotation());
    double qdot = std::abs(lift_start_q.normalized().dot(q.normalized()));
    qdot = std::min(1.0, qdot);
    lift_ori_dev = std::max(lift_ori_dev, 2.0 * std::acos(qdot) * 180.0 / M_PI);
  }

  RCLCPP_INFO(node->get_logger(), "========== LIFT VALIDATION ==========");
  RCLCPP_INFO(node->get_logger(), "Pipeline: pilz_industrial_motion_planner");
  RCLCPP_INFO(node->get_logger(), "Planner ID: %s", lift_seg.planner_id.c_str());
  RCLCPP_INFO(node->get_logger(), "Attached object present at start: %s",
              lift_attached ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "Trajectory points: %zu", lift_seg.points.size());
  RCLCPP_INFO(node->get_logger(), "Start TCP world: %.6f %.6f %.6f", lift_start_world.x(),
              lift_start_world.y(), lift_start_world.z());
  RCLCPP_INFO(node->get_logger(), "End TCP world: %.6f %.6f %.6f", lift_end_world.x(),
              lift_end_world.y(), lift_end_world.z());
  RCLCPP_INFO(node->get_logger(), "dx=%.6f dy=%.6f dz=%.6f", lift_delta.x(), lift_delta.y(),
              lift_delta.z());
  RCLCPP_INFO(node->get_logger(), "Target lift distance: %.6f", lift_distance);
  RCLCPP_INFO(node->get_logger(), "Cartesian path length: %.6f", lift_path);
  RCLCPP_INFO(node->get_logger(), "Max lateral deviation from world Z: %.6f m", lift_lateral);
  RCLCPP_INFO(node->get_logger(), "Max orientation deviation: %.6f deg", lift_ori_dev);
  RCLCPP_INFO(node->get_logger(), "Lift endpoint position error: %.6f m", lift_pos);
  RCLCPP_INFO(node->get_logger(), "Lift endpoint orientation error: %.6f deg", lift_ori);

  std::vector<ObjectPairContact> lift_illegal_all;
  size_t lift_states_checked = 0;
  if (lift_sol && lift_sol->start() && lift_sol->start()->scene())
  {
    for (size_t i = 0; i < lift_seg.points.size(); ++i)
    {
      const auto diag = lift_sol->start()->scene()->diff();
      applyJoints(diag->getCurrentStateNonConst(), lift_seg.points[i]);
      std::vector<ObjectPairContact> allowed_i;
      std::vector<ObjectPairContact> illegal_i;
      classifyObjectContacts(rawObjectContacts(*diag, object_id, table_name, touch_links),
                             touch_links, allowed_i, illegal_i);
      ++lift_states_checked;
      for (const auto& contact : illegal_i)
      {
        RCLCPP_ERROR(node->get_logger(),
                     "Lift waypoint %zu illegal: small_part <-> %s depth=%.6f", i,
                     contact.other.c_str(), contact.depth);
        lift_illegal_all.push_back(contact);
      }
    }
  }
  RCLCPP_INFO(node->get_logger(), "========== LIFT TRAJECTORY COLLISION CHECK ==========");
  RCLCPP_INFO(node->get_logger(), "trajectory states checked: %zu", lift_states_checked);
  logContacts(node->get_logger(), "illegal collisions:", lift_illegal_all);
  for (const auto& link : kArmBodyLinks)
  {
    bool hit = false;
    for (const auto& contact : lift_illegal_all)
    {
      hit = hit || contact.other == link;
    }
    RCLCPP_INFO(node->get_logger(), "attached part vs %s: %s", link.c_str(),
                hit ? "COLLISION" : "NO COLLISION");
  }

  const double pre_grasp_disc = maxJointError(ompl_seg.points.back(), grasp_seg.points.front());
  const double grasp_attach_disc =
      attach_sol && attach_sol->end() ?
          maxJointError(grasp_seg.points.back(),
                        jointsFromState(attach_sol->end()->scene()->getCurrentState())) :
          1.0;
  const double attach_lift_disc =
      attach_sol && attach_sol->end() ?
          maxJointError(jointsFromState(attach_sol->end()->scene()->getCurrentState()),
                        lift_seg.points.front()) :
          1.0;
  const double lift_inspect_disc = maxJointError(lift_seg.points.back(), inspect_seg.points.front());
  const double max_disc =
      std::max(pre_grasp_disc, std::max(grasp_attach_disc, std::max(attach_lift_disc, lift_inspect_disc)));
  RCLCPP_INFO(node->get_logger(), "========== STAGE CONTINUITY ==========");
  RCLCPP_INFO(node->get_logger(), "PreGrasp→Grasp: %.9f rad", pre_grasp_disc);
  RCLCPP_INFO(node->get_logger(), "Grasp→Attach: %.9f rad", grasp_attach_disc);
  RCLCPP_INFO(node->get_logger(), "Attach→Lift: %.9f rad", attach_lift_disc);
  RCLCPP_INFO(node->get_logger(), "Lift→View: %.9f rad", lift_inspect_disc);
  RCLCPP_INFO(node->get_logger(), "max joint discontinuity: %.9f rad", max_disc);

  bool restore_ok = false;
  if (restore_sol && restore_sol->end() && restore_sol->end()->scene())
  {
    restore_ok = !acmAllowed(*restore_sol->end()->scene(), object_id, table_name);
  }
  RCLCPP_INFO(node->get_logger(), "Restore part-table collision after lift: %s",
              restore_ok ? "YES" : "NO");

  bool inspect_attached_start = false;
  bool inspect_attached_end = false;
  bool inspect_part_table_allowed = true;
  if (inspect_sol && inspect_sol->start() && inspect_sol->start()->scene())
  {
    inspect_attached_start = inspect_sol->start()->scene()->getCurrentState().hasAttachedBody(object_id);
    inspect_part_table_allowed = acmAllowed(*inspect_sol->start()->scene(), object_id, table_name);
  }
  if (inspect_sol && inspect_sol->end() && inspect_sol->end()->scene())
  {
    inspect_attached_end = inspect_sol->end()->scene()->getCurrentState().hasAttachedBody(object_id);
  }

  applyJoints(fk_state, inspect_seg.points.back());
  const Eigen::Isometry3d actual_tcp = tcpInBase(fk_state, ee_link);
  const Eigen::Isometry3d t_inspect_tcp = poseToIso(inspect.pose);
  const Eigen::Isometry3d t_inspect_obj = poseToIso(inspect_object.pose);
  const Eigen::Isometry3d t_tcp_obj = poseToIso(tcp_object.pose);
  const Eigen::Isometry3d actual_object = actual_tcp * t_tcp_obj;
  double inspect_tcp_pos = 0.0;
  double inspect_tcp_ori = 0.0;
  double inspect_obj_pos = 0.0;
  double inspect_obj_ori = 0.0;
  poseError(t_inspect_tcp, actual_tcp, inspect_tcp_pos, inspect_tcp_ori);
  poseError(t_inspect_obj, actual_object, inspect_obj_pos, inspect_obj_ori);
  const Eigen::Vector3d actual_center =
      actual_object.translation() + actual_object.linear() * center_in_object;
  const Eigen::Vector3d actual_normal = (actual_object.linear() * normal_in_object).normalized();
  const Eigen::Vector3d actual_up = (actual_object.linear() * up_in_object).normalized();
  const double view_center_err = (actual_center - p1).norm();
  const double normal_err_deg =
      std::acos(std::min(1.0, std::max(-1.0, actual_normal.dot(d1)))) * 180.0 / M_PI;
  const double up_err_deg =
      std::acos(std::min(1.0, std::max(-1.0, actual_up.dot(preferred_up)))) * 180.0 / M_PI;

  double inspect_joint_len = 0.0;
  size_t inspect_bound_violations = 0;
  std::vector<ObjectPairContact> inspect_illegal_all;
  const auto* arm_jmg = robot_model->getJointModelGroup(group);
  for (size_t i = 0; i < inspect_seg.points.size(); ++i)
  {
    applyJoints(fk_state, inspect_seg.points[i]);
    // Group-only: gripper/slider defaults on a scratch RobotState are not task joints.
    if (arm_jmg && !fk_state.satisfiesBounds(arm_jmg))
    {
      ++inspect_bound_violations;
      if (inspect_bound_violations == 1)
      {
        for (const auto& name : kArmJoints)
        {
          const auto* jm = robot_model->getJointModel(name);
          if (!jm || fk_state.satisfiesBounds(jm))
          {
            continue;
          }
          const auto& bounds = jm->getVariableBounds();
          const double value = fk_state.getVariablePosition(name);
          RCLCPP_ERROR(node->get_logger(),
                       "first bounds violation waypoint %zu %s=%.6f bounds=[%.6f, %.6f]", i,
                       name.c_str(), value, bounds.empty() ? 0.0 : bounds.front().min_position_,
                       bounds.empty() ? 0.0 : bounds.front().max_position_);
        }
      }
    }
    if (i > 0)
    {
      inspect_joint_len += maxJointError(inspect_seg.points[i], inspect_seg.points[i - 1]);
    }
    if (inspect_sol && inspect_sol->start() && inspect_sol->start()->scene())
    {
      const auto diag = inspect_sol->start()->scene()->diff();
      applyJoints(diag->getCurrentStateNonConst(), inspect_seg.points[i]);
      std::vector<ObjectPairContact> allowed_i;
      std::vector<ObjectPairContact> illegal_i;
      classifyObjectContacts(rawObjectContacts(*diag, object_id, table_name, touch_links),
                             touch_links, allowed_i, illegal_i);
      inspect_illegal_all.insert(inspect_illegal_all.end(), illegal_i.begin(), illegal_i.end());
    }
  }
  // L2 joint path length.
  inspect_joint_len = 0.0;
  for (size_t i = 1; i < inspect_seg.points.size(); ++i)
  {
    double acc = 0.0;
    for (const auto& name : kArmJoints)
    {
      const double dq = inspect_seg.points[i].at(name) - inspect_seg.points[i - 1].at(name);
      acc += dq * dq;
    }
    inspect_joint_len += std::sqrt(acc);
  }
  double inspect_duration = 0.0;
  size_t arm_seg_index = 0;
  for (const auto& sub : solution_msg.sub_trajectory)
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
    if (!has_arm)
    {
      continue;
    }
    if (arm_seg_index == 3)
    {
      inspect_duration = rclcpp::Duration(jt.points.back().time_from_start).seconds();
    }
    ++arm_seg_index;
  }

  RCLCPP_INFO(node->get_logger(), "========== INSPECTION VIEW %s ==========", view_name.c_str());
  RCLCPP_INFO(node->get_logger(), "Lift→View planner: OMPL id='%s'", inspect_seg.planner_id.c_str());
  RCLCPP_INFO(node->get_logger(), "Attached at Lift end: %s", lift_attached ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "Attached at Inspection start: %s", inspect_attached_start ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "Attached at Inspection end: %s", inspect_attached_end ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "part-table during Lift→View allowed: %s",
              inspect_part_table_allowed ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "Trajectory points: %zu", inspect_seg.points.size());
  RCLCPP_INFO(node->get_logger(), "Joint path length: %.6f rad", inspect_joint_len);
  RCLCPP_INFO(node->get_logger(), "Predicted duration: %.6f s", inspect_duration);
  RCLCPP_INFO(node->get_logger(), "TCP endpoint error: position=%.6f m orientation=%.6f deg",
              inspect_tcp_pos, inspect_tcp_ori);
  RCLCPP_INFO(node->get_logger(), "Object endpoint error: position=%.6f m orientation=%.6f deg",
              inspect_obj_pos, inspect_obj_ori);
  RCLCPP_INFO(node->get_logger(), "View-center error: %.6f m", view_center_err);
  RCLCPP_INFO(node->get_logger(), "Normal error: %.6f deg", normal_err_deg);
  RCLCPP_INFO(node->get_logger(), "Canonical-up error: %.6f deg", up_err_deg);
  logContacts(node->get_logger(), "Illegal trajectory collision states:", inspect_illegal_all);
  RCLCPP_INFO(node->get_logger(), "Joint-bounds violations: %zu", inspect_bound_violations);

  const bool all_ok =
      pre_pos <= pos_tol && pre_ori <= ori_tol_deg && pregrasp_world && !part_touch_at_home &&
      part_table_allowed && part_touch_before_grasp && grasp_seg.planner_id == "LIN" &&
      grasp_pos <= pos_tol && grasp_ori <= ori_tol_deg && grasp_lateral <= kLateralTolM &&
      before_world && !before_attached && !after_world && after_attached &&
      attached_link == attach_link && attach_pos <= kAttachPosTolM &&
      attach_ori <= kAttachOriTolDeg && lift_attached && lift_seg.planner_id == "LIN" &&
      lift_lateral <= kLateralTolM && lift_ori_dev <= kLinOriTolDeg &&
      std::abs(lift_delta.z() - lift_distance) <= 0.005 && lift_delta.z() > 0.0 &&
      max_disc <= 1e-4 && lift_pos <= pos_tol && lift_ori <= ori_tol_deg && grasp_illegal.empty() &&
      attach_illegal.empty() && lift_illegal_all.empty() && restore_ok && !inspect_part_table_allowed &&
      inspect_attached_start && inspect_attached_end && inspect_tcp_pos <= pos_tol &&
      inspect_tcp_ori <= ori_tol_deg && inspect_obj_pos <= pos_tol && inspect_obj_ori <= ori_tol_deg &&
      view_center_err <= view_center_tol && normal_err_deg <= ori_tol_deg &&
      up_err_deg <= up_tol_deg && inspect_illegal_all.empty() && inspect_bound_violations == 0 &&
      inspect_seg.planner_id != "LIN";

  RCLCPP_INFO(node->get_logger(), "Physical gripper close: NOT EXECUTED");
  RCLCPP_INFO(node->get_logger(), "Gazebo weld: NOT PERFORMED");
  RCLCPP_INFO(node->get_logger(), "fallback used? NO");
  RCLCPP_INFO(node->get_logger(), "Complete Solution: YES (%zu)", num_solutions);
  RCLCPP_INFO(node->get_logger(), "FAILURE CLASSIFICATION:\nCOMPLETE TASK SUCCESS");
  RCLCPP_INFO(node->get_logger(), "STEP 7 view %s verification: %s", view_name.c_str(),
              all_ok ? "PASS" : "FAIL");

  task.publishAllSolutions(false);
  task.enableIntrospection(true);

  const auto after_msg = waitForFreshJoints(node, std::chrono::seconds(5));
  if (after_msg)
  {
    const auto after = jointsFromMsg(*after_msg);
    const double drift = maxJointError(after, actual);
    RCLCPP_INFO(node->get_logger(), "JOINT STATES BEFORE:\n%s", formatJoints(actual).c_str());
    RCLCPP_INFO(node->get_logger(), "JOINT STATES AFTER:\n%s", formatJoints(after).c_str());
    RCLCPP_INFO(node->get_logger(), "Max joint drift: %.6f rad", drift);
    RCLCPP_INFO(node->get_logger(), "Robot moved because of this node? %s",
                drift > 0.02 ? "YES" : "NO");
  }
  bool live_world_after = false;
  bool live_attached_after = false;
  inspectLiveScene(node, object_id, live_world_after, live_attached_after);
  RCLCPP_INFO(node->get_logger(), "LIVE SCENE after test: world=%s attached=%s (before world=%s attached=%s)",
              live_world_after ? "YES" : "NO", live_attached_after ? "YES" : "NO",
              live_world_before ? "YES" : "NO", live_attached_before ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "small_part permanently added? %s",
              (live_world_after && !live_world_before) ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(),
              "Introspection is active for RViz. THIS STEP IS PLAN-ONLY. Ctrl+C to exit.");
  spinner.join();
  return all_ok ? 0 : 4;
}
