#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
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

bool isRequiredView(const std::string& name)
{
  return name == "side_pos_y" || name == "side_neg_y" || name == "top_circle";
}

struct ViewSpec
{
  std::string name;
  geometry_msgs::msg::PoseStamped tcp;
  geometry_msgs::msg::PoseStamped object;
  Eigen::Vector3d center_in_object = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal_in_object = Eigen::Vector3d::UnitY();
  Eigen::Vector3d up_in_object = Eigen::Vector3d::UnitZ();
};

ViewSpec readViewSpec(const rclcpp::Node::SharedPtr& node, const std::string& prefix)
{
  ViewSpec spec;
  spec.name = getString(node, prefix + "_view", "");
  spec.tcp = readPose(node, prefix);
  spec.object = readPose(node, prefix + "_object");
  spec.center_in_object = Eigen::Vector3d(getDouble(node, prefix + "_center_in_object_x"),
                                          getDouble(node, prefix + "_center_in_object_y"),
                                          getDouble(node, prefix + "_center_in_object_z"));
  spec.normal_in_object = Eigen::Vector3d(getDouble(node, prefix + "_normal_in_object_x"),
                                          getDouble(node, prefix + "_normal_in_object_y"),
                                          getDouble(node, prefix + "_normal_in_object_z"));
  spec.up_in_object = Eigen::Vector3d(getDouble(node, prefix + "_up_in_object_x"),
                                     getDouble(node, prefix + "_up_in_object_y"),
                                     getDouble(node, prefix + "_up_in_object_z"));
  return spec;
}

struct EndpointCheck
{
  double tcp_pos = 0.0;
  double tcp_ori = 0.0;
  double obj_pos = 0.0;
  double obj_ori = 0.0;
  double view_center_err = 0.0;
  double normal_err_deg = 0.0;
  double up_err_deg = 0.0;
};

EndpointCheck checkEndpoint(const Eigen::Isometry3d& actual_tcp, const ViewSpec& spec,
                            const Eigen::Isometry3d& t_tcp_obj, const Eigen::Vector3d& p1,
                            const Eigen::Vector3d& d1, const Eigen::Vector3d& preferred_up,
                            const Eigen::Isometry3d& t_world_base)
{
  EndpointCheck out;
  const Eigen::Isometry3d actual_object = actual_tcp * t_tcp_obj;
  poseError(poseToIso(spec.tcp.pose), actual_tcp, out.tcp_pos, out.tcp_ori);
  poseError(poseToIso(spec.object.pose), actual_object, out.obj_pos, out.obj_ori);
  const Eigen::Isometry3d actual_object_world = t_world_base * actual_object;
  const Eigen::Vector3d actual_center =
      actual_object_world.translation() + actual_object_world.linear() * spec.center_in_object;
  const Eigen::Vector3d actual_normal =
      (actual_object_world.linear() * spec.normal_in_object).normalized();
  const Eigen::Vector3d actual_up = (actual_object_world.linear() * spec.up_in_object).normalized();
  out.view_center_err = (actual_center - p1).norm();
  out.normal_err_deg =
      std::acos(std::min(1.0, std::max(-1.0, actual_normal.dot(d1)))) * 180.0 / M_PI;
  out.up_err_deg =
      std::acos(std::min(1.0, std::max(-1.0, actual_up.dot(preferred_up)))) * 180.0 / M_PI;
  return out;
}

double jointPathLengthL2(const std::vector<std::map<std::string, double>>& points)
{
  double length = 0.0;
  for (size_t i = 1; i < points.size(); ++i)
  {
    double acc = 0.0;
    for (const auto& name : kArmJoints)
    {
      const double dq = points[i].at(name) - points[i - 1].at(name);
      acc += dq * dq;
    }
    length += std::sqrt(acc);
  }
  return length;
}

size_t countGroupBoundViolations(moveit::core::RobotState& state,
                                 const moveit::core::JointModelGroup* jmg,
                                 const std::vector<std::map<std::string, double>>& points)
{
  size_t n = 0;
  if (!jmg)
  {
    return 0;
  }
  for (const auto& joints : points)
  {
    applyJoints(state, joints);
    if (!state.satisfiesBounds(jmg))
    {
      ++n;
    }
  }
  return n;
}

std::vector<ObjectPairContact> illegalOnSegment(
    const planning_scene::PlanningScene& start_scene, const std::string& object_id,
    const std::string& table_name, const std::vector<std::string>& touch_links,
    const std::vector<std::map<std::string, double>>& points)
{
  std::vector<ObjectPairContact> illegal;
  for (const auto& joints : points)
  {
    const auto diag = start_scene.diff();
    applyJoints(diag->getCurrentStateNonConst(), joints);
    std::vector<ObjectPairContact> allowed_i;
    std::vector<ObjectPairContact> illegal_i;
    classifyObjectContacts(rawObjectContacts(*diag, object_id, table_name, touch_links),
                           touch_links, allowed_i, illegal_i);
    illegal.insert(illegal.end(), illegal_i.begin(), illegal_i.end());
  }
  return illegal;
}

std::string diagnosePoseFailure(const moveit::core::RobotModelConstPtr& robot_model,
                                const std::string& group, const std::string& ee_link,
                                const std::string& attach_link, const std::string& object_id,
                                const std::vector<std::string>& touch_links,
                                const geometry_msgs::msg::PoseStamped& pose,
                                const std::map<std::string, double>& seed, const std::string& kind)
{
  auto* jmg = robot_model->getJointModelGroup(group);
  if (!jmg)
  {
    return "OTHER";
  }
  moveit::core::RobotState ik_state(robot_model);
  ik_state.setToDefaultValues();
  applyJoints(ik_state, seed);
  const bool ik_ok = ik_state.setFromIK(jmg, pose.pose, ee_link, 0.2);
  if (!ik_ok)
  {
    return kind + " IK FAILURE";
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
    return kind + " COLLISION";
  }
  return kind == "TARGET" ? "SOURCE→TARGET OMPL FAILURE" : kind + " TRANSIT FAILURE";
}

std::string classifyTransitionFailure(const std::string& explain,
                                      const moveit::core::RobotModelConstPtr& robot_model,
                                      const std::string& group, const std::string& ee_link,
                                      const std::string& attach_link, const std::string& object_id,
                                      const std::vector<std::string>& touch_links,
                                      const ViewSpec& source, const ViewSpec& target,
                                      const std::map<std::string, double>& seed)
{
  const std::string source_cls =
      diagnosePoseFailure(robot_model, group, ee_link, attach_link, object_id, touch_links,
                          source.tcp, seed, "SOURCE");
  const std::string target_cls =
      diagnosePoseFailure(robot_model, group, ee_link, attach_link, object_id, touch_links,
                          target.tcp, seed, "TARGET");
  if (explain.find("Source Inspection") != std::string::npos ||
      explain.find("MoveTo Source") != std::string::npos)
  {
    return source_cls;
  }
  if (explain.find("Target Inspection") != std::string::npos ||
      explain.find("MoveTo Target") != std::string::npos)
  {
    return target_cls;
  }
  if (source_cls.find("IK FAILURE") != std::string::npos ||
      source_cls.find("COLLISION") != std::string::npos)
  {
    return source_cls;
  }
  if (target_cls.find("IK FAILURE") != std::string::npos ||
      target_cls.find("COLLISION") != std::string::npos)
  {
    return target_cls;
  }
  return "OTHER";
}

double armSegmentDuration(const moveit_task_constructor_msgs::msg::Solution& solution_msg,
                          size_t arm_index)
{
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
    if (arm_seg_index == arm_index)
    {
      return rclcpp::Duration(jt.points.back().time_from_start).seconds();
    }
    ++arm_seg_index;
  }
  return 0.0;
}

std::vector<std::string> parseViewOrder(const std::string& raw)
{
  std::vector<std::string> out;
  std::string token;
  std::istringstream iss(raw);
  while (std::getline(iss, token, ','))
  {
    token.erase(0, token.find_first_not_of(" \t"));
    if (!token.empty())
    {
      const auto last = token.find_last_not_of(" \t");
      token.erase(last + 1);
      out.push_back(token);
    }
  }
  return out;
}

bool isPermutationOfRequired(const std::vector<std::string>& names)
{
  if (names.size() != 3)
  {
    return false;
  }
  std::set<std::string> unique(names.begin(), names.end());
  return unique.size() == 3 &&
         unique.count("side_pos_y") && unique.count("side_neg_y") && unique.count("top_circle");
}

std::string orderLabel(const std::vector<std::string>& names)
{
  auto letter = [](const std::string& name) -> std::string {
    if (name == "side_pos_y")
    {
      return "A";
    }
    if (name == "side_neg_y")
    {
      return "B";
    }
    return "C";
  };
  return letter(names[0]) + "_" + letter(names[1]) + "_" + letter(names[2]);
}

std::string jointsDigest(const std::vector<TrajectorySegment>& segs)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(6);
  for (const auto& seg : segs)
  {
    for (const auto& point : seg.points)
    {
      for (const auto& name : kArmJoints)
      {
        oss << name << "=" << point.at(name) << ";";
      }
      oss << "|";
    }
    oss << "#";
  }
  const std::string text = oss.str();
  uint64_t hash = 1469598103934665603ull;
  for (unsigned char c : text)
  {
    hash ^= c;
    hash *= 1099511628211ull;
  }
  std::ostringstream hex;
  hex << std::hex << hash;
  return hex.str();
}

double terminalSpeed(const moveit_task_constructor_msgs::msg::Solution& solution_msg,
                     size_t arm_index, bool& available)
{
  available = false;
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
    if (arm_seg_index == arm_index)
    {
      const auto& last = jt.points.back();
      if (last.velocities.empty())
      {
        return 0.0;
      }
      available = true;
      double acc = 0.0;
      for (double v : last.velocities)
      {
        acc += v * v;
      }
      return std::sqrt(acc);
    }
    ++arm_seg_index;
  }
  return 0.0;
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("fr3_mtc_full_order_test", options);
  RCLCPP_INFO(node->get_logger(), "THIS STEP IS PLAN-ONLY. NO TRAJECTORY EXECUTION IS PERFORMED.");
  RCLCPP_INFO(node->get_logger(), "Physical gripper close: NOT EXECUTED");
  RCLCPP_INFO(node->get_logger(), "Predicted grasp scene transition: YES");
  RCLCPP_INFO(node->get_logger(),
              "STEP 9: one complete three-view order. Canonical orientations. No roll sampling.");

  const auto home = readHome(node);
  const auto pregrasp = readPose(node, "pregrasp");
  const auto grasp = readPose(node, "grasp");
  const auto lift = readPose(node, "lift");
  const auto object_world = readPose(node, "object_world");
  const auto object_base = readPose(node, "object");
  const auto world_base = readPose(node, "world_base");
  const auto tcp_object = readPose(node, "tcp_object");
  const ViewSpec view1 = readViewSpec(node, "view1");
  const ViewSpec view2 = readViewSpec(node, "view2");
  const ViewSpec view3 = readViewSpec(node, "view3");
  const std::string view_order_raw = getString(node, "view_order", "");
  const auto order_names = parseViewOrder(view_order_raw);
  const Eigen::Vector3d p1(getDouble(node, "p1_x"), getDouble(node, "p1_y"), getDouble(node, "p1_z"));
  const Eigen::Vector3d d1 =
      Eigen::Vector3d(getDouble(node, "d1_x"), getDouble(node, "d1_y"), getDouble(node, "d1_z"))
          .normalized();
  const Eigen::Vector3d preferred_up =
      Eigen::Vector3d(getDouble(node, "preferred_up_x"), getDouble(node, "preferred_up_y"),
                      getDouble(node, "preferred_up_z"))
          .normalized();
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
  const int64_t solutions_per_order =
      node->has_parameter("solutions_per_order") ?
          node->get_parameter("solutions_per_order").as_int() :
          (node->has_parameter("max_solutions") ? node->get_parameter("max_solutions").as_int() : 5);
  const double hold_seconds =
      node->has_parameter("hold_seconds") ? node->get_parameter("hold_seconds").as_double() : 2.0;
  const int64_t trial_index =
      node->has_parameter("trial_index") ? node->get_parameter("trial_index").as_int() : 0;
  const std::string diagnostic_output_path = getString(node, "diagnostic_output_path", "");
  const bool hold_for_introspection =
      node->has_parameter("hold_for_introspection") &&
      node->get_parameter("hold_for_introspection").as_bool();
  if (!isPermutationOfRequired(order_names) || view1.name != order_names[0] ||
      view2.name != order_names[1] || view3.name != order_names[2])
  {
    RCLCPP_ERROR(node->get_logger(),
                 "ABORT: view_order must be a permutation of "
                 "{side_pos_y, side_neg_y, top_circle}. got '%s' specs='%s,%s,%s'",
                 view_order_raw.c_str(), view1.name.c_str(), view2.name.c_str(),
                 view3.name.c_str());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  RCLCPP_INFO(node->get_logger(), "Parameters loaded. Overlaying robot_description from /move_group.");
  auto param_node =
      rclcpp::Node::make_shared("fr3_mtc_full_order_params_" + std::to_string(node->get_clock()->now().nanoseconds()));
  if (!overlayRobotDescriptionFromMoveGroup(node, param_node))
  {
    shutdownSpinner(executor, spinner);
    return 1;
  }
  RCLCPP_INFO(node->get_logger(), "robot_description overlay complete.");

  bool live_world_before = false;
  bool live_attached_before = false;
  inspectLiveScene(node, object_id, live_world_before, live_attached_before);

  const auto before_msg = waitForFreshJoints(node, std::chrono::seconds(10));
  if (!before_msg)
  {
    RCLCPP_ERROR(node->get_logger(), "/joint_states missing. STEP 9 FAIL.");
    shutdownSpinner(executor, spinner);
    return 1;
  }
  const auto actual = jointsFromMsg(*before_msg);
  if (jointsAreAllZero(actual))
  {
    RCLCPP_ERROR(node->get_logger(), "CurrentState is all zeros. STEP 9 FAIL.");
    shutdownSpinner(executor, spinner);
    return 1;
  }

  double max_home_error = 0.0;
  for (const auto& name : kArmJoints)
  {
    max_home_error = std::max(max_home_error, std::abs(actual.at(name) - home.at(name)));
  }

  RCLCPP_INFO(node->get_logger(), "========== STEP 9 PREFLIGHT ==========");
  RCLCPP_INFO(node->get_logger(), "view_order:\n%s → %s → %s", view1.name.c_str(), view2.name.c_str(),
              view3.name.c_str());
  RCLCPP_INFO(node->get_logger(), "solutions_per_order:\n%ld", static_cast<long>(solutions_per_order));
  RCLCPP_INFO(node->get_logger(), "Canonical View1 TCP:\n%s", formatPose(view1.tcp).c_str());
  RCLCPP_INFO(node->get_logger(), "Canonical View2 TCP:\n%s", formatPose(view2.tcp).c_str());
  RCLCPP_INFO(node->get_logger(), "Canonical View3 TCP:\n%s", formatPose(view3.tcp).c_str());
  RCLCPP_INFO(node->get_logger(), "hold_seconds (report only, not executed): %.3f", hold_seconds);
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
  task.setName("FR3 Full Inspection Order Task");
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

  const std::array<ViewSpec, 3> views = { view1, view2, view3 };
  std::array<std::string, 3> view_stage_names;
  for (size_t i = 0; i < 3; ++i)
  {
    auto ompl_view =
        std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
    ompl_view->setTimeout(planning_time);
    view_stage_names[i] =
        "MoveTo Inspection View " + std::to_string(i + 1) + " " + views[i].name;
    auto move_view =
        std::make_unique<moveit::task_constructor::stages::MoveTo>(view_stage_names[i], ompl_view);
    move_view->setGroup(group);
    move_view->setIKFrame(ee_link);
    move_view->setGoal(views[i].tcp);
    move_view->setTimeout(planning_time);
    task.add(std::move(move_view));
  }

  RCLCPP_INFO(node->get_logger(), "========== MTC TASK ==========");
  RCLCPP_INFO(node->get_logger(),
              "CurrentState\n↓ Prepare Object On Table\n↓ MoveTo PreGrasp / OMPL\n↓ Allow "
              "Gripper-Part Contact\n↓ MoveTo Grasp / Pilz LIN\n↓ Attach Part To TCP\n↓ MoveTo "
              "Lift / Pilz LIN\n↓ Restore Part-Table Collision\n↓ %s / OMPL\n↓ %s / OMPL\n↓ %s / "
              "OMPL",
              view_stage_names[0].c_str(), view_stage_names[1].c_str(),
              view_stage_names[2].c_str());
  RCLCPP_INFO(node->get_logger(), "Collision checking: ENABLED");
  RCLCPP_INFO(node->get_logger(),
              "Inspection endpoints are stop-required. Hold time is not executed.");

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

  const auto plan_t0 = std::chrono::steady_clock::now();
  const auto plan_result =
      task.plan(static_cast<size_t>(std::max<int64_t>(1, solutions_per_order)));
  const double planning_computation_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - plan_t0).count();
  const size_t num_solutions = task.numSolutions();
  std::ostringstream state;
  task.printState(state);
  RCLCPP_INFO(node->get_logger(), "planning result: %s", plan_result ? "SUCCESS" : "FAIL");
  RCLCPP_INFO(node->get_logger(), "requested: %ld", static_cast<long>(solutions_per_order));
  RCLCPP_INFO(node->get_logger(), "Complete solutions returned: %zu", num_solutions);
  RCLCPP_INFO(node->get_logger(), "planning computation time: %.6f s", planning_computation_s);
  RCLCPP_INFO(node->get_logger(), "MTC task state:\n%s", state.str().c_str());

  std::string failure_class = "COMPLETE TASK SUCCESS";
  std::string failure_stage = "";
  if (!plan_result || num_solutions < 1)
  {
    std::ostringstream failure;
    task.explainFailure(failure);
    RCLCPP_ERROR(node->get_logger(), "planning failure:\n%s", failure.str().c_str());
    const std::string fail_text = failure.str();
    if (fail_text.find("MoveTo PreGrasp") != std::string::npos ||
        fail_text.find("MoveTo Grasp") != std::string::npos ||
        fail_text.find("MoveTo Lift") != std::string::npos)
    {
      failure_class = "prefix failure";
      if (fail_text.find("MoveTo PreGrasp") != std::string::npos)
      {
        failure_stage = "MoveTo PreGrasp";
      }
      else if (fail_text.find("MoveTo Grasp") != std::string::npos)
      {
        failure_stage = "MoveTo Grasp";
      }
      else
      {
        failure_stage = "MoveTo Lift";
      }
    }
    else if (fail_text.find(view_stage_names[2]) != std::string::npos)
    {
      failure_class = "View2→View3 failure";
      failure_stage = view_stage_names[2];
    }
    else if (fail_text.find(view_stage_names[1]) != std::string::npos)
    {
      failure_class = "View1→View2 failure";
      failure_stage = view_stage_names[1];
    }
    else if (fail_text.find(view_stage_names[0]) != std::string::npos)
    {
      failure_class = "View1 failure";
      failure_stage = view_stage_names[0];
    }
    else
    {
      failure_class = "OMPL stochastic failure within budget";
    }
    RCLCPP_ERROR(node->get_logger(), "FAILURE CLASSIFICATION:\n%s", failure_class.c_str());
    if (!failure_stage.empty())
    {
      RCLCPP_ERROR(node->get_logger(), "FAILURE STAGE:\n%s", failure_stage.c_str());
    }
  }


  struct CandRec
  {
    int index = 0;
    bool valid = false;
    std::string reason;
    std::string digest;
    double T = 0.0;
    double L = 0.0;
    double T_hold = 0.0;
    double max_disc = 0.0;
    size_t collisions = 0;
    size_t bounds = 0;
    std::array<double, 6> seg_T{};
    std::array<double, 6> seg_L{};
    std::array<size_t, 6> seg_pts{};
    std::array<EndpointCheck, 3> views{};
    std::array<double, 3> view_term_speed{};
    std::array<bool, 3> view_term_available{};
  };

  const auto* arm_jmg = robot_model->getJointModelGroup(group);
  const Eigen::Isometry3d t_tcp_obj = poseToIso(tcp_object.pose);
  const Eigen::Isometry3d t_pre_target = poseToIso(pregrasp.pose);
  const Eigen::Isometry3d t_grasp_target = poseToIso(grasp.pose);
  const Eigen::Isometry3d t_lift_target = poseToIso(lift.pose);
  std::vector<CandRec> recs;
  int cand_index = 0;
  for (const auto& sol_ptr : task.solutions())
  {
    CandRec rec;
    rec.index = cand_index++;
    rec.reason = "ok";
    const auto& solution = *sol_ptr;
    std::vector<const moveit::task_constructor::SolutionBase*> leaves;
    flattenSolutions(solution, leaves);
    const auto* attach_sol = findStageSolution(leaves, "Attach Part To TCP");
    const auto* lift_sol = findStageSolution(leaves, "MoveTo Lift");
    const auto* restore_sol = findStageSolution(leaves, "Restore Part-Table Collision");
    std::array<const moveit::task_constructor::SolutionBase*, 3> view_sols{};
    for (size_t i = 0; i < 3; ++i)
    {
      view_sols[i] = findStageSolution(leaves, view_stage_names[i]);
    }
    moveit_task_constructor_msgs::msg::Solution solution_msg;
    solution.toMsg(solution_msg);
    const auto segments = extractArmSegments(solution_msg);
    if (segments.size() < 6)
    {
      rec.reason = "missing motion segments";
      recs.push_back(rec);
      continue;
    }
    rec.digest = jointsDigest(segments);
    for (size_t i = 0; i < 6; ++i)
    {
      rec.seg_pts[i] = segments[i].points.size();
      rec.seg_T[i] = armSegmentDuration(solution_msg, i);
      rec.seg_L[i] = jointPathLengthL2(segments[i].points);
      rec.T += rec.seg_T[i];
      rec.L += rec.seg_L[i];
    }
    rec.T_hold = rec.T + 3.0 * hold_seconds;

    moveit::core::RobotState fk_state(robot_model);
    fk_state.setToDefaultValues();
    applyJoints(fk_state, segments[0].points.back());
    double pre_pos = 0.0;
    double pre_ori = 0.0;
    poseError(t_pre_target, tcpInBase(fk_state, ee_link), pre_pos, pre_ori);
    applyJoints(fk_state, segments[1].points.back());
    double grasp_pos = 0.0;
    double grasp_ori = 0.0;
    poseError(t_grasp_target, tcpInBase(fk_state, ee_link), grasp_pos, grasp_ori);
    applyJoints(fk_state, segments[2].points.front());
    const Eigen::Vector3d lift_start = tcpInWorld(fk_state, ee_link, model_frame, t_world_base).translation();
    applyJoints(fk_state, segments[2].points.back());
    const Eigen::Vector3d lift_end = tcpInWorld(fk_state, ee_link, model_frame, t_world_base).translation();
    const Eigen::Vector3d lift_delta = lift_end - lift_start;
    double lift_pos = 0.0;
    double lift_ori = 0.0;
    poseError(t_lift_target, tcpInBase(fk_state, ee_link), lift_pos, lift_ori);

    bool after_attached = false;
    std::string attached_link;
    Eigen::Isometry3d actual_tcp_object = Eigen::Isometry3d::Identity();
    if (attach_sol && attach_sol->end() && attach_sol->end()->scene() &&
        attach_sol->end()->scene()->getCurrentState().hasAttachedBody(object_id))
    {
      after_attached = true;
      const auto* body = attach_sol->end()->scene()->getCurrentState().getAttachedBody(object_id);
      attached_link = body->getAttachedLinkName();
      actual_tcp_object = body->getPose();
    }
    double attach_pos = 0.0;
    double attach_ori = 0.0;
    poseError(t_tcp_obj, actual_tcp_object, attach_pos, attach_ori);
    bool restore_ok = restore_sol && restore_sol->end() && restore_sol->end()->scene() &&
                      !acmAllowed(*restore_sol->end()->scene(), object_id, table_name);

    const double pre_grasp_disc = maxJointError(segments[0].points.back(), segments[1].points.front());
    const double attach_lift_disc =
        attach_sol && attach_sol->end() ?
            maxJointError(jointsFromState(attach_sol->end()->scene()->getCurrentState()),
                          segments[2].points.front()) :
            1.0;
    const double lift_v1_disc = maxJointError(segments[2].points.back(), segments[3].points.front());
    const double v12_disc = maxJointError(segments[3].points.back(), segments[4].points.front());
    const double v23_disc = maxJointError(segments[4].points.back(), segments[5].points.front());
    rec.max_disc = std::max({pre_grasp_disc, attach_lift_disc, lift_v1_disc, v12_disc, v23_disc});

    bool views_attached = true;
    bool views_geom = true;
    bool views_table = true;
    for (size_t i = 0; i < 3; ++i)
    {
      if (!(view_sols[i] && view_sols[i]->start() && view_sols[i]->end()))
      {
        views_attached = false;
        rec.reason = "missing view stage";
        break;
      }
      views_attached = views_attached &&
                       view_sols[i]->start()->scene()->getCurrentState().hasAttachedBody(object_id) &&
                       view_sols[i]->end()->scene()->getCurrentState().hasAttachedBody(object_id);
      views_table = views_table &&
                    !acmAllowed(*view_sols[i]->start()->scene(), object_id, table_name);
      applyJoints(fk_state, segments[3 + i].points.back());
      rec.views[i] = checkEndpoint(tcpInBase(fk_state, ee_link), views[i], t_tcp_obj, p1, d1,
                                   preferred_up, t_world_base);
      rec.view_term_speed[i] = terminalSpeed(solution_msg, 3 + i, rec.view_term_available[i]);
      views_geom = views_geom && rec.views[i].tcp_pos <= pos_tol &&
                   rec.views[i].tcp_ori <= ori_tol_deg && rec.views[i].obj_pos <= pos_tol &&
                   rec.views[i].obj_ori <= ori_tol_deg &&
                   rec.views[i].view_center_err <= view_center_tol &&
                   rec.views[i].normal_err_deg <= ori_tol_deg && rec.views[i].up_err_deg <= up_tol_deg;
      rec.bounds += countGroupBoundViolations(fk_state, arm_jmg, segments[3 + i].points);
      if (view_sols[i]->start()->scene())
      {
        const auto illegal = illegalOnSegment(*view_sols[i]->start()->scene(), object_id, table_name,
                                              touch_links, segments[3 + i].points);
        rec.collisions += illegal.size();
      }
    }
    rec.bounds += countGroupBoundViolations(fk_state, arm_jmg, segments[0].points);
    rec.bounds += countGroupBoundViolations(fk_state, arm_jmg, segments[1].points);
    rec.bounds += countGroupBoundViolations(fk_state, arm_jmg, segments[2].points);
    if (lift_sol && lift_sol->start() && lift_sol->start()->scene())
    {
      rec.collisions +=
          illegalOnSegment(*lift_sol->start()->scene(), object_id, table_name, touch_links,
                           segments[2].points)
              .size();
    }

    rec.valid = pre_pos <= pos_tol && pre_ori <= ori_tol_deg && grasp_pos <= pos_tol &&
                grasp_ori <= ori_tol_deg && segments[1].planner_id == "LIN" &&
                segments[2].planner_id == "LIN" && segments[3].planner_id != "LIN" &&
                segments[4].planner_id != "LIN" && segments[5].planner_id != "LIN" &&
                lift_delta.z() > 0.0 && std::abs(lift_delta.z() - lift_distance) <= 0.005 &&
                lift_pos <= pos_tol && lift_ori <= ori_tol_deg && after_attached &&
                attached_link == attach_link && attach_pos <= kAttachPosTolM &&
                attach_ori <= kAttachOriTolDeg && restore_ok && rec.max_disc <= 1e-4 &&
                views_attached && views_table && views_geom && rec.collisions == 0 &&
                rec.bounds == 0;
    if (!rec.valid && rec.reason == "ok")
    {
      rec.reason = "independent validation failed";
    }
    RCLCPP_INFO(node->get_logger(),
                "ORDER %s candidate%d T=%.6f L=%.6f valid=%s reason=%s digest=%s",
                orderLabel(order_names).c_str(), rec.index, rec.T, rec.L,
                rec.valid ? "true" : "false", rec.reason.c_str(), rec.digest.c_str());
    recs.push_back(rec);
  }

  size_t valid_count = 0;
  for (const auto& rec : recs)
  {
    valid_count += rec.valid ? 1 : 0;
  }
  RCLCPP_INFO(node->get_logger(), "requested=%ld returned=%zu valid=%zu",
              static_cast<long>(solutions_per_order), recs.size(), valid_count);
  RCLCPP_INFO(node->get_logger(), "Physical gripper close: NOT EXECUTED");
  RCLCPP_INFO(node->get_logger(), "Gazebo weld: NOT PERFORMED");
  RCLCPP_INFO(node->get_logger(), "fallback used? NO");
  RCLCPP_INFO(node->get_logger(), "FAILURE CLASSIFICATION:\n%s", failure_class.c_str());
  RCLCPP_INFO(node->get_logger(), "STEP 9 order %s verification: %s",
              orderLabel(order_names).c_str(), valid_count > 0 ? "PASS" : "FAIL");

  {
    const std::string yaml_path =
        diagnostic_output_path.empty() ?
            ("/tmp/fr3_step9_" + orderLabel(order_names) + ".yaml") :
            diagnostic_output_path;
    std::ofstream yaml(yaml_path);
    yaml << "order: " << view1.name << "," << view2.name << "," << view3.name << "\n";
    yaml << "order_key: " << orderLabel(order_names) << "\n";
    yaml << "trial_index: " << trial_index << "\n";
    yaml << "planning_success: " << (num_solutions > 0 ? "true" : "false") << "\n";
    yaml << "complete_solution_count: " << num_solutions << "\n";
    yaml << "requested: " << solutions_per_order << "\n";
    yaml << "returned: " << recs.size() << "\n";
    yaml << "valid: " << valid_count << "\n";
    yaml << "planning_computation_time: " << planning_computation_s << "\n";
    yaml << "hold_seconds: " << hold_seconds << "\n";
    yaml << "failure_classification: " << failure_class << "\n";
    yaml << "failure_stage: " << (failure_stage.empty() ? "none" : failure_stage) << "\n";
    yaml << "candidates:\n";
    for (const auto& rec : recs)
    {
      yaml << "  - candidate_index: " << rec.index << "\n";
      yaml << "    valid: " << (rec.valid ? "true" : "false") << "\n";
      yaml << "    reason: " << rec.reason << "\n";
      yaml << "    digest: " << rec.digest << "\n";
      yaml << "    total_predicted_motion_duration: " << rec.T << "\n";
      yaml << "    total_joint_path_length: " << rec.L << "\n";
      yaml << "    predicted_with_hold_time: " << rec.T_hold << "\n";
      yaml << "    max_joint_discontinuity: " << rec.max_disc << "\n";
      yaml << "    illegal_collision_states: " << rec.collisions << "\n";
      yaml << "    joint_bounds_violations: " << rec.bounds << "\n";
      const char* names[6] = {"home_to_pregrasp", "pregrasp_to_grasp", "grasp_to_lift",
                              "lift_to_view1",    "view1_to_view2",    "view2_to_view3"};
      yaml << "    segment_metrics:\n";
      for (size_t i = 0; i < 6; ++i)
      {
        yaml << "      " << names[i] << ":\n";
        yaml << "        points: " << rec.seg_pts[i] << "\n";
        yaml << "        duration: " << rec.seg_T[i] << "\n";
        yaml << "        path_length: " << rec.seg_L[i] << "\n";
      }
      yaml << "    view_errors:\n";
      for (size_t i = 0; i < 3; ++i)
      {
        yaml << "      - name: " << views[i].name << "\n";
        yaml << "        view_center: " << rec.views[i].view_center_err << "\n";
        yaml << "        normal_deg: " << rec.views[i].normal_err_deg << "\n";
        yaml << "        up_deg: " << rec.views[i].up_err_deg << "\n";
        if (rec.view_term_available[i])
        {
          yaml << "        terminal_joint_speed: " << rec.view_term_speed[i] << "\n";
        }
        else
        {
          yaml << "        terminal_joint_speed: unavailable\n";
        }
      }
    }
    RCLCPP_INFO(node->get_logger(), "Wrote order diagnostics: %s", yaml_path.c_str());
  }

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
  RCLCPP_INFO(node->get_logger(),
              "LIVE SCENE after test: world=%s attached=%s (before world=%s attached=%s)",
              live_world_after ? "YES" : "NO", live_attached_after ? "YES" : "NO",
              live_world_before ? "YES" : "NO", live_attached_before ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "small_part permanently added? %s",
              (live_world_after && !live_world_before) ? "YES" : "NO");
  if (hold_for_introspection)
  {
    RCLCPP_INFO(node->get_logger(),
                "Introspection is active for RViz. THIS STEP IS PLAN-ONLY. Ctrl+C to exit.");
    spinner.join();
  }
  else
  {
    shutdownSpinner(executor, spinner);
  }
  return valid_count > 0 ? 0 : 4;
}
