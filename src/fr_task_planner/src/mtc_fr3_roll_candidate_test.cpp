#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
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
#include <moveit/task_constructor/stages/compute_ik.h>
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/fixed_cartesian_poses.h>
#include <moveit/task_constructor/stages/fixed_state.h>
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
const std::vector<std::string> kRequiredViews = { "side_pos_y", "side_neg_y", "top_circle" };

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

geometry_msgs::msg::Pose isoToPose(const Eigen::Isometry3d& transform)
{
  geometry_msgs::msg::Pose pose;
  const Eigen::Vector3d p = transform.translation();
  const Eigen::Quaterniond q(transform.rotation());
  pose.position.x = p.x();
  pose.position.y = p.y();
  pose.position.z = p.z();
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();
  return pose;
}

geometry_msgs::msg::PoseStamped tcpInModelFrame(const geometry_msgs::msg::PoseStamped& tcp_base,
                                                const Eigen::Isometry3d& t_model_base,
                                                const std::string& model_frame)
{
  geometry_msgs::msg::PoseStamped out;
  out.header.frame_id = model_frame;
  out.pose = isoToPose(t_model_base * poseToIso(tcp_base.pose));
  return out;
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

double jointL2(const std::map<std::string, double>& a, const std::map<std::string, double>& b)
{
  double acc = 0.0;
  for (const auto& name : kArmJoints)
  {
    const double dq = a.at(name) - b.at(name);
    acc += dq * dq;
  }
  return std::sqrt(acc);
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
                      bool& world_present, bool& attached_present, bool& table_present,
                      bool& column_present, const std::string& table_name,
                      const std::string& column_name)
{
  world_present = false;
  attached_present = false;
  table_present = false;
  column_present = false;
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
    if (obj.id == table_name)
    {
      table_present = true;
    }
    if (obj.id == column_name)
    {
      column_present = true;
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

struct ViewGeom
{
  std::string name;
  Eigen::Vector3d center_in_object = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal_in_object = Eigen::Vector3d::Zero();
  Eigen::Vector3d up_in_object = Eigen::Vector3d::Zero();
  geometry_msgs::msg::PoseStamped canonical_object;
  geometry_msgs::msg::PoseStamped canonical_tcp;
};

struct RollPose
{
  std::string view;
  double roll_deg = 0.0;
  int pose_index = 0;
  geometry_msgs::msg::PoseStamped object;
  geometry_msgs::msg::PoseStamped tcp;
  double requested_center_err = 0.0;
  double requested_normal_err = 0.0;
  double requested_up_err = 0.0;
};

struct EndpointCandidate
{
  std::string view;
  double roll_deg = 0.0;
  int pose_index = 0;
  int ik_index = 0;
  geometry_msgs::msg::PoseStamped object_target;
  geometry_msgs::msg::PoseStamped tcp_target;
  std::map<std::string, double> joints;
  double view_center_error = 0.0;
  double normal_error = 0.0;
  double up_error = 0.0;
  double tcp_object_error = 0.0;
  double joint_distance_from_lift = 0.0;
  bool bounds_ok = false;
  bool collision_free = false;
  bool attached_object_present = false;
  bool valid = false;
  std::string failure_reason = "ok";
};

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
  view.canonical_object = readPose(node, name + "_canonical_object");
  view.canonical_tcp = readPose(node, name + "_canonical_tcp");
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
  const auto cerr = node->get_parameter("roll_center_err").as_double_array();
  const auto nerr = node->get_parameter("roll_normal_err").as_double_array();
  const auto uerr = node->get_parameter("roll_up_err").as_double_array();
  const std::string frame = getString(node, "planning_frame", kPlanningFrame);
  if (views.size() != rolls.size() || views.empty())
  {
    throw std::runtime_error("roll pose arrays are empty or mismatched");
  }
  std::vector<RollPose> out;
  out.reserve(views.size());
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
    pose.requested_center_err = cerr[i];
    pose.requested_normal_err = nerr[i];
    pose.requested_up_err = uerr[i];
    out.push_back(pose);
  }
  return out;
}

const ViewGeom* findView(const std::map<std::string, ViewGeom>& views, const std::string& name)
{
  const auto it = views.find(name);
  return it == views.end() ? nullptr : &it->second;
}

int matchRollIndex(const std::vector<RollPose>& rolls, const Eigen::Isometry3d& tcp)
{
  int best = -1;
  double best_score = 1e9;
  for (size_t i = 0; i < rolls.size(); ++i)
  {
    double pos = 0.0;
    double ori = 0.0;
    poseError(poseToIso(rolls[i].tcp.pose), tcp, pos, ori);
    const double score = pos + 0.001 * ori;
    if (score < best_score)
    {
      best_score = score;
      best = static_cast<int>(i);
    }
  }
  if (best >= 0 && best_score < 0.02)
  {
    return best;
  }
  return -1;
}

void validateCandidate(EndpointCandidate& cand, const planning_scene::PlanningScene& lift_scene,
                       const moveit::core::RobotModelConstPtr& robot_model, const std::string& group,
                       const std::string& ee_link, const std::string& object_id,
                       const std::string& table_name, const std::vector<std::string>& touch_links,
                       const Eigen::Isometry3d& t_tcp_object, const ViewGeom& view,
                       const Eigen::Vector3d& p1, const Eigen::Vector3d& d1,
                       const Eigen::Vector3d& preferred_up, const std::map<std::string, double>& q_lift,
                       double pos_tol, double ori_tol_deg)
{
  auto* jmg = robot_model->getJointModelGroup(group);
  moveit::core::RobotState fk(robot_model);
  fk.setToDefaultValues();
  applyJoints(fk, cand.joints);
  cand.bounds_ok = static_cast<bool>(jmg) && fk.satisfiesBounds(jmg);
  const Eigen::Isometry3d actual_tcp = tcpInBase(fk, ee_link);
  const Eigen::Isometry3d actual_object = actual_tcp * t_tcp_object;
  const Eigen::Vector3d actual_center =
      actual_object.translation() + actual_object.linear() * view.center_in_object;
  const Eigen::Vector3d actual_normal = (actual_object.linear() * view.normal_in_object).normalized();
  const Eigen::Vector3d actual_up = (actual_object.linear() * view.up_in_object).normalized();
  cand.view_center_error = (actual_center - p1).norm();
  cand.normal_error =
      std::acos(std::min(1.0, std::max(-1.0, actual_normal.dot(d1.normalized())))) * 180.0 / M_PI;
  cand.up_error =
      std::acos(std::min(1.0, std::max(-1.0, actual_up.dot(preferred_up.normalized())))) * 180.0 /
      M_PI;
  double tcp_pos = 0.0;
  double tcp_ori = 0.0;
  poseError(poseToIso(cand.tcp_target.pose), actual_tcp, tcp_pos, tcp_ori);
  cand.tcp_object_error = tcp_pos;
  cand.joint_distance_from_lift = jointL2(cand.joints, q_lift);

  const auto diag = lift_scene.diff();
  applyJoints(diag->getCurrentStateNonConst(), cand.joints);
  cand.attached_object_present = diag->getCurrentState().hasAttachedBody(object_id);
  collision_detection::CollisionRequest req;
  collision_detection::CollisionResult res;
  req.contacts = true;
  req.max_contacts = 40;
  diag->checkCollision(req, res);
  std::vector<ObjectPairContact> allowed;
  std::vector<ObjectPairContact> illegal;
  classifyObjectContacts(rawObjectContacts(*diag, object_id, table_name, touch_links), touch_links,
                         allowed, illegal);
  cand.collision_free = !res.collision && illegal.empty();

  if (!cand.bounds_ok)
  {
    cand.failure_reason = "joint bounds";
  }
  else if (!cand.attached_object_present)
  {
    cand.failure_reason = "attached object missing";
  }
  else if (!cand.collision_free)
  {
    cand.failure_reason = res.collision ? "collision" : "illegal attached contact";
  }
  else if (cand.view_center_error > pos_tol || cand.normal_error > ori_tol_deg)
  {
    cand.failure_reason = "fk geometry";
  }
  else
  {
    cand.valid = true;
    cand.failure_reason = "ok";
  }
}

bool collectComputeIK(const planning_scene::PlanningScenePtr& lift_scene,
                      const rclcpp::Node::SharedPtr& node, const std::string& group,
                      const std::string& ee_link, uint32_t max_ik, double min_dist,
                      const std::vector<RollPose>& rolls, const Eigen::Isometry3d& t_model_base,
                      const std::string& model_frame, std::vector<EndpointCandidate>& raw,
                      std::string& error)
{
  moveit::task_constructor::Task diag("", false);
  diag.setName("FR3 Roll Endpoint IK");
  diag.loadRobotModel(node);
  diag.setProperty("group", group);
  diag.setProperty("eef", std::string("hkv_gripper"));

  auto fixed = std::make_unique<moveit::task_constructor::stages::FixedState>("Lift Scene");
  fixed->setState(lift_scene);
  fixed->setIgnoreCollisions(false);
  auto* fixed_ptr = fixed.get();
  diag.add(std::move(fixed));

  auto poses = std::make_unique<moveit::task_constructor::stages::FixedCartesianPoses>(
      "Inspection Roll Poses");
  for (const auto& roll : rolls)
  {
    poses->addPose(tcpInModelFrame(roll.tcp, t_model_base, model_frame));
  }
  poses->setMonitoredStage(fixed_ptr);

  auto ik = std::make_unique<moveit::task_constructor::stages::ComputeIK>("ComputeIK Roll Endpoints",
                                                                         std::move(poses));
  ik->setGroup(group);
  ik->setEndEffector("hkv_gripper");
  ik->setIKFrame(ee_link);
  ik->setMaxIKSolutions(max_ik);
  ik->setMinSolutionDistance(min_dist);
  ik->setIgnoreCollisions(false);
  ik->properties().configureInitFrom(moveit::task_constructor::Stage::INTERFACE, { "target_pose" });
  diag.add(std::move(ik));

  try
  {
    diag.init();
  }
  catch (const std::exception& ex)
  {
    error = std::string("diagnostic task.init() failed: ") + ex.what();
    return false;
  }

  const size_t budget = std::max<size_t>(1, rolls.size() * static_cast<size_t>(max_ik));
  const auto plan_ok = diag.plan(budget);
  std::ostringstream state;
  diag.printState(state);
  RCLCPP_INFO(node->get_logger(), "ComputeIK diagnostic result: %s solutions=%zu",
              plan_ok ? "SUCCESS" : "FAIL", diag.numSolutions());
  RCLCPP_INFO(node->get_logger(), "ComputeIK task state:\n%s", state.str().c_str());
  if (!plan_ok && diag.numSolutions() == 0)
  {
    std::ostringstream failure;
    diag.explainFailure(failure);
    error = failure.str();
    return false;
  }

  std::map<int, int> per_pose;
  for (const auto& solution_ptr : diag.solutions())
  {
    std::vector<const moveit::task_constructor::SolutionBase*> leaves;
    flattenSolutions(*solution_ptr, leaves);
    const auto* ik_sol = findStageSolution(leaves, "ComputeIK Roll Endpoints");
    const moveit::task_constructor::InterfaceState* state_ptr = nullptr;
    if (ik_sol && ik_sol->end())
    {
      state_ptr = ik_sol->end();
    }
    else if (solution_ptr->end())
    {
      state_ptr = solution_ptr->end();
    }
    if (!state_ptr || !state_ptr->scene())
    {
      continue;
    }
    EndpointCandidate cand;
    cand.joints = jointsFromState(state_ptr->scene()->getCurrentState());
    Eigen::Isometry3d tcp = tcpInBase(state_ptr->scene()->getCurrentState(), ee_link);
    if (state_ptr->properties().hasProperty("target_pose"))
    {
      const auto target =
          state_ptr->properties().get<geometry_msgs::msg::PoseStamped>("target_pose");
      tcp = poseToIso(target.pose);
    }
    const int idx = matchRollIndex(rolls, tcp);
    if (idx < 0)
    {
      continue;
    }
    cand.view = rolls[idx].view;
    cand.roll_deg = rolls[idx].roll_deg;
    cand.pose_index = rolls[idx].pose_index;
    cand.object_target = rolls[idx].object;
    cand.tcp_target = rolls[idx].tcp;
    cand.ik_index = per_pose[idx]++;
    raw.push_back(cand);
  }
  return true;
}

void collectSetFromIK(const planning_scene::PlanningScene& lift_scene,
                      const rclcpp::Logger& logger, const std::string& group,
                      const std::string& ee_link, uint32_t max_ik, double min_dist,
                      const std::vector<RollPose>& rolls, const Eigen::Isometry3d& t_model_base,
                      std::vector<EndpointCandidate>& raw)
{
  const auto robot_model = lift_scene.getRobotModel();
  auto* jmg = robot_model->getJointModelGroup(group);
  if (!jmg)
  {
    return;
  }
  moveit::core::RobotState seed(lift_scene.getCurrentState());
  for (size_t i = 0; i < rolls.size(); ++i)
  {
    const Eigen::Isometry3d t_model_tcp = t_model_base * poseToIso(rolls[i].tcp.pose);
    std::vector<std::map<std::string, double>> found;
    const uint32_t attempts = std::max<uint32_t>(max_ik * 4, 8);
    for (uint32_t attempt = 0; attempt < attempts && found.size() < max_ik; ++attempt)
    {
      moveit::core::RobotState ik_state(seed);
      if (attempt > 0)
      {
        ik_state.setToRandomPositionsNearBy(jmg, seed, 2.0);
      }
      const double timeout = (attempt == 0) ? 0.25 : 0.05;
      if (!ik_state.setFromIK(jmg, t_model_tcp, ee_link, timeout))
      {
        if (attempt == 0 && std::abs(rolls[i].roll_deg) < 1e-9)
        {
          RCLCPP_WARN(logger,
                      "canonical setFromIK miss %s model_tcp xyz=(%.4f, %.4f, %.4f) timeout=%.2f",
                      rolls[i].view.c_str(), t_model_tcp.translation().x(),
                      t_model_tcp.translation().y(), t_model_tcp.translation().z(), timeout);
        }
        continue;
      }
      const auto q = jointsFromState(ik_state);
      bool duplicate = false;
      for (const auto& prev : found)
      {
        if (jointL2(prev, q) < min_dist)
        {
          duplicate = true;
          break;
        }
      }
      if (duplicate)
      {
        continue;
      }
      found.push_back(q);
      EndpointCandidate cand;
      cand.view = rolls[i].view;
      cand.roll_deg = rolls[i].roll_deg;
      cand.pose_index = rolls[i].pose_index;
      cand.object_target = rolls[i].object;
      cand.tcp_target = rolls[i].tcp;
      cand.ik_index = static_cast<int>(found.size() - 1);
      cand.joints = q;
      raw.push_back(cand);
    }
  }
}

void writeYaml(const std::string& path, const std::string& view, const std::vector<RollPose>& rolls,
               const std::vector<EndpointCandidate>& all)
{
  std::ofstream yaml(path);
  yaml.setf(std::ios::fixed);
  yaml.precision(9);
  yaml << "view_name: " << view << "\n";
  yaml << "candidates:\n";
  for (const auto& cand : all)
  {
    if (cand.view != view)
    {
      continue;
    }
    yaml << "  - view_name: " << cand.view << "\n";
    yaml << "    roll_deg: " << cand.roll_deg << "\n";
    yaml << "    pose_index: " << cand.pose_index << "\n";
    yaml << "    object_target:\n";
    yaml << "      frame: " << cand.object_target.header.frame_id << "\n";
    yaml << "      xyz: [" << cand.object_target.pose.position.x << ", "
         << cand.object_target.pose.position.y << ", " << cand.object_target.pose.position.z
         << "]\n";
    yaml << "      xyzw: [" << cand.object_target.pose.orientation.x << ", "
         << cand.object_target.pose.orientation.y << ", " << cand.object_target.pose.orientation.z
         << ", " << cand.object_target.pose.orientation.w << "]\n";
    yaml << "    tcp_target:\n";
    yaml << "      frame: " << cand.tcp_target.header.frame_id << "\n";
    yaml << "      xyz: [" << cand.tcp_target.pose.position.x << ", "
         << cand.tcp_target.pose.position.y << ", " << cand.tcp_target.pose.position.z << "]\n";
    yaml << "      xyzw: [" << cand.tcp_target.pose.orientation.x << ", "
         << cand.tcp_target.pose.orientation.y << ", " << cand.tcp_target.pose.orientation.z << ", "
         << cand.tcp_target.pose.orientation.w << "]\n";
    yaml << "    view_center_error: " << cand.view_center_error << "\n";
    yaml << "    normal_error: " << cand.normal_error << "\n";
    yaml << "    up_error: " << cand.up_error << "\n";
    yaml << "    ik_index: " << cand.ik_index << "\n";
    yaml << "    joints: [";
    for (size_t i = 0; i < kArmJoints.size(); ++i)
    {
      yaml << cand.joints.at(kArmJoints[i]);
      if (i + 1 < kArmJoints.size())
      {
        yaml << ", ";
      }
    }
    yaml << "]\n";
    yaml << "    bounds_ok: " << (cand.bounds_ok ? "true" : "false") << "\n";
    yaml << "    collision_free: " << (cand.collision_free ? "true" : "false") << "\n";
    yaml << "    attached_object_present: " << (cand.attached_object_present ? "true" : "false")
         << "\n";
    yaml << "    T_tcp_object_error: " << cand.tcp_object_error << "\n";
    yaml << "    joint_distance_from_lift: " << cand.joint_distance_from_lift << "\n";
    yaml << "    joint_distance_note: NOT a path cost / NOT used for ranking\n";
    yaml << "    valid: " << (cand.valid ? "true" : "false") << "\n";
    yaml << "    failure_reason: " << cand.failure_reason << "\n";
  }
  yaml << "requested_poses: " << rolls.size() << "\n";
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("fr3_mtc_roll_candidate_test", options);
  RCLCPP_INFO(node->get_logger(), "THIS STEP IS PLAN-ONLY. NO TRAJECTORY EXECUTION IS PERFORMED.");
  RCLCPP_INFO(node->get_logger(), "STEP 10: roll / IK endpoint candidates. No order search.");

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
  const std::string planning_frame = getString(node, "planning_frame", kPlanningFrame);
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
  const double roll_step_deg = node->has_parameter("roll_step_deg") ?
                                   node->get_parameter("roll_step_deg").as_double() :
                                   30.0;
  const int64_t max_ik = node->has_parameter("max_ik_solutions_per_pose") ?
                             node->get_parameter("max_ik_solutions_per_pose").as_int() :
                             8;
  const double min_ik_dist = node->has_parameter("min_ik_solution_distance") ?
                                 node->get_parameter("min_ik_solution_distance").as_double() :
                                 0.1;
  const std::string output_dir = getString(node, "output_dir", "/tmp");
  const bool hold_for_introspection = node->has_parameter("hold_for_introspection") &&
                                      node->get_parameter("hold_for_introspection").as_bool();

  std::map<std::string, ViewGeom> views;
  for (const auto& name : kRequiredViews)
  {
    views[name] = readViewGeom(node, name);
  }
  const auto rolls = readRollPoses(node);

  RCLCPP_INFO(node->get_logger(), "Parameters loaded. Overlaying robot_description from /move_group.");
  auto param_node = rclcpp::Node::make_shared("fr3_mtc_roll_candidate_params");
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
    RCLCPP_ERROR(node->get_logger(), "/joint_states missing. STEP 10 FAIL.");
    shutdownSpinner(executor, spinner);
    return 1;
  }
  const auto actual = jointsFromMsg(*before_msg);
  if (jointsAreAllZero(actual))
  {
    RCLCPP_ERROR(node->get_logger(), "CurrentState is all zeros. STEP 10 FAIL.");
    shutdownSpinner(executor, spinner);
    return 1;
  }
  double max_home_error = 0.0;
  for (const auto& name : kArmJoints)
  {
    max_home_error = std::max(max_home_error, std::abs(actual.at(name) - home.at(name)));
  }
  RCLCPP_INFO(node->get_logger(), "========== STEP 10 PREFLIGHT ==========");
  RCLCPP_INFO(node->get_logger(), "roll_step_deg=%.3f samples=%zu max_ik=%ld min_dist=%.3f",
              roll_step_deg, rolls.size(), static_cast<long>(max_ik), min_ik_dist);
  RCLCPP_INFO(node->get_logger(), "Current joints:\n%s", formatJoints(actual).c_str());
  RCLCPP_INFO(node->get_logger(), "Max Home error: %.6f rad %s", max_home_error,
              max_home_error <= home_tol ? "PASS" : "FAIL");
  if (max_home_error > home_tol)
  {
    RCLCPP_ERROR(node->get_logger(), "Current is not Stage4 Home. Planning aborted.");
    shutdownSpinner(executor, spinner);
    return 2;
  }

  for (const auto& link : touch_links)
  {
    if (isArmBodyLink(link))
    {
      RCLCPP_ERROR(node->get_logger(), "Arm link incorrectly in touch_links: %s", link.c_str());
      shutdownSpinner(executor, spinner);
      return 1;
    }
  }

  double max_center = 0.0;
  double max_normal = 0.0;
  bool canonical_regressed = true;
  for (const auto& roll : rolls)
  {
    const auto* view = findView(views, roll.view);
    if (!view)
    {
      RCLCPP_ERROR(node->get_logger(), "Unknown view %s", roll.view.c_str());
      shutdownSpinner(executor, spinner);
      return 1;
    }
    const Eigen::Isometry3d t_obj = poseToIso(roll.object.pose);
    const Eigen::Vector3d center = t_obj.translation() + t_obj.linear() * view->center_in_object;
    const Eigen::Vector3d normal = (t_obj.linear() * view->normal_in_object).normalized();
    const double center_err = (center - p1).norm();
    const double normal_err =
        std::acos(std::min(1.0, std::max(-1.0, normal.dot(d1)))) * 180.0 / M_PI;
    max_center = std::max(max_center, center_err);
    max_normal = std::max(max_normal, normal_err);
    if (std::abs(roll.roll_deg) < 1e-9)
    {
      double obj_pos = 0.0;
      double obj_ori = 0.0;
      double tcp_pos = 0.0;
      double tcp_ori = 0.0;
      poseError(poseToIso(view->canonical_object.pose), t_obj, obj_pos, obj_ori);
      poseError(poseToIso(view->canonical_tcp.pose), poseToIso(roll.tcp.pose), tcp_pos, tcp_ori);
      RCLCPP_INFO(node->get_logger(),
                  "canonical regression %s object pos=%.3e ori=%.3e deg tcp pos=%.3e ori=%.3e deg",
                  roll.view.c_str(), obj_pos, obj_ori, tcp_pos, tcp_ori);
      if (obj_pos > 1e-9 || obj_ori > 1e-6 || tcp_pos > 1e-9 || tcp_ori > 1e-6)
      {
        canonical_regressed = false;
      }
    }
  }
  RCLCPP_INFO(node->get_logger(), "Requested pose max view-center error: %.3e m", max_center);
  RCLCPP_INFO(node->get_logger(), "Requested pose max normal error: %.3e deg", max_normal);
  if (!canonical_regressed || max_center > 1e-6 || max_normal > 1e-6)
  {
    RCLCPP_ERROR(node->get_logger(), "STEP 10 FAIL: geometry regression before IK.");
    shutdownSpinner(executor, spinner);
    return 4;
  }

  moveit::task_constructor::Task prefix("", true);
  prefix.setName("FR3 Lift Prefix For Roll IK");
  prefix.loadRobotModel(node);
  const auto robot_model = prefix.getRobotModel();
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

  prefix.add(std::make_unique<moveit::task_constructor::stages::CurrentState>("CurrentState"));
  auto prepare = std::make_unique<moveit::task_constructor::stages::ModifyPlanningScene>(
      "Prepare Object On Table");
  prepare->addObject(makeCylinder(object_id, object_scene, object_height, object_radius));
  prepare->allowCollisions(object_id, table_name, true);
  prefix.add(std::move(prepare));

  auto ompl = std::make_shared<moveit::task_constructor::solvers::PipelinePlanner>(node, kOmplPipeline);
  ompl->setTimeout(planning_time);
  auto move_pregrasp =
      std::make_unique<moveit::task_constructor::stages::MoveTo>("MoveTo PreGrasp", ompl);
  move_pregrasp->setGroup(group);
  move_pregrasp->setIKFrame(ee_link);
  move_pregrasp->setGoal(pregrasp);
  move_pregrasp->setTimeout(planning_time);
  prefix.add(std::move(move_pregrasp));

  auto allow_touch = std::make_unique<moveit::task_constructor::stages::ModifyPlanningScene>(
      "Allow Gripper-Part Contact");
  allow_touch->allowCollisions(object_id, touch_links, true);
  prefix.add(std::move(allow_touch));

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
  prefix.add(std::move(move_grasp));

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
  prefix.add(std::move(attach));

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
  prefix.add(std::move(move_lift));

  auto restore = std::make_unique<moveit::task_constructor::stages::ModifyPlanningScene>(
      "Restore Part-Table Collision");
  restore->allowCollisions(object_id, table_name, false);
  prefix.add(std::move(restore));

  RCLCPP_INFO(node->get_logger(),
              "========== PREFIX TASK ==========\nCurrentState → Prepare → PreGrasp/OMPL → "
              "Allow touch → Grasp/LIN → Attach → Lift/LIN → Restore\n"
              "No Lift→View path planning. No order search.");
  try
  {
    prefix.init();
  }
  catch (const std::exception& ex)
  {
    RCLCPP_ERROR(node->get_logger(), "prefix.init() failed: %s", ex.what());
    shutdownSpinner(executor, spinner);
    return 1;
  }
  const auto prefix_ok = prefix.plan(1);
  if (!prefix_ok || prefix.numSolutions() < 1)
  {
    std::ostringstream failure;
    prefix.explainFailure(failure);
    RCLCPP_ERROR(node->get_logger(), "prefix planning failed:\n%s", failure.str().c_str());
    shutdownSpinner(executor, spinner);
    return 3;
  }

  const auto& prefix_sol = *prefix.solutions().front();
  std::vector<const moveit::task_constructor::SolutionBase*> leaves;
  flattenSolutions(prefix_sol, leaves);
  const auto* restore_sol = findStageSolution(leaves, "Restore Part-Table Collision");
  if (!restore_sol || !restore_sol->end() || !restore_sol->end()->scene())
  {
    RCLCPP_ERROR(node->get_logger(), "Restore scene missing");
    shutdownSpinner(executor, spinner);
    return 3;
  }
  const auto lift_scene = planning_scene::PlanningScene::clone(restore_sol->end()->scene());
  const Eigen::Isometry3d t_model_base =
      lift_scene->getCurrentState().getGlobalLinkTransform("base_link");
  RCLCPP_INFO(node->get_logger(), "RobotModel model frame: %s", model_frame.c_str());
  RCLCPP_INFO(node->get_logger(),
              "T_model_base xyz=(%.6f, %.6f, %.6f)  (setFromIK pose frame = model frame)",
              t_model_base.translation().x(), t_model_base.translation().y(),
              t_model_base.translation().z());
  const bool attached = lift_scene->getCurrentState().hasAttachedBody(object_id);
  std::string attached_link;
  if (attached)
  {
    attached_link = lift_scene->getCurrentState().getAttachedBody(object_id)->getAttachedLinkName();
  }
  const bool table_present = static_cast<bool>(lift_scene->getWorld()->getObject(table_name));
  const bool column_present = static_cast<bool>(lift_scene->getWorld()->getObject(column_name));
  const bool part_table_allowed = acmAllowed(*lift_scene, object_id, table_name);
  const auto q_lift = jointsFromState(lift_scene->getCurrentState());
  RCLCPP_INFO(node->get_logger(), "========== LIFT SOURCE SCENE ==========");
  RCLCPP_INFO(node->get_logger(), "How obtained: one prefix Task.plan(1) through Restore");
  RCLCPP_INFO(node->get_logger(), "small_part attached: %s", attached ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "attached link: %s", attached_link.c_str());
  RCLCPP_INFO(node->get_logger(), "table present: %s", table_present ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "mounting_column present: %s", column_present ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "part-table collision allowed: %s",
              part_table_allowed ? "YES" : "NO");
  RCLCPP_INFO(node->get_logger(), "Lift joints:\n%s", formatJoints(q_lift).c_str());
  if (!attached || attached_link != attach_link || part_table_allowed || !table_present)
  {
    RCLCPP_ERROR(node->get_logger(), "Lift scene is not the required attached Restore scene.");
    shutdownSpinner(executor, spinner);
    return 3;
  }

  std::vector<EndpointCandidate> raw;
  std::string ik_error;
  bool used_compute_ik = collectComputeIK(lift_scene, node, group, ee_link,
                                          static_cast<uint32_t>(std::max<int64_t>(1, max_ik)),
                                          min_ik_dist, rolls, t_model_base, model_frame, raw,
                                          ik_error);
  std::string ik_backend = "ComputeIK";
  if (!used_compute_ik || raw.empty())
  {
    RCLCPP_WARN(node->get_logger(),
                "ComputeIK diagnostic produced no usable solutions (%s). Falling back to "
                "RobotState::setFromIK on the same cloned Lift scene.",
                ik_error.empty() ? "empty" : ik_error.c_str());
    raw.clear();
    collectSetFromIK(*lift_scene, node->get_logger(), group, ee_link,
                     static_cast<uint32_t>(std::max<int64_t>(1, max_ik)), min_ik_dist, rolls,
                     t_model_base, raw);
    ik_backend = "RobotState::setFromIK";
    used_compute_ik = false;
  }
  RCLCPP_INFO(node->get_logger(), "IK backend: %s", ik_backend.c_str());
  RCLCPP_INFO(node->get_logger(),
              "roll/IK branching integrated in same prefix Task? NO");
  RCLCPP_INFO(node->get_logger(),
              "Humble restriction: Serial prefix ends at Restore (propagator). A Generator "
              "cannot sit after it as a sibling. ComputeIK+FixedCartesianPoses therefore runs "
              "in a diagnostic Task seeded by PlanningScene::clone(Restore). "
              "Whole-task endpoint branching is deferred.");

  const Eigen::Isometry3d t_tcp_object = poseToIso(tcp_object.pose);
  std::vector<EndpointCandidate> validated;
  for (auto cand : raw)
  {
    const auto* view = findView(views, cand.view);
    if (!view)
    {
      continue;
    }
    validateCandidate(cand, *lift_scene, robot_model, group, ee_link, object_id, table_name,
                      touch_links, t_tcp_object, *view, p1, d1, preferred_up, q_lift, pos_tol,
                      ori_tol_deg);
    validated.push_back(cand);
    RCLCPP_INFO(node->get_logger(),
                "candidate %s roll=%.1f ik=%d valid=%s reason=%s joints=%.3f %.3f %.3f %.3f %.3f "
                "%.3f up=%.3f coll=%s",
                cand.view.c_str(), cand.roll_deg, cand.ik_index, cand.valid ? "YES" : "NO",
                cand.failure_reason.c_str(), cand.joints.at("j1"), cand.joints.at("j2"),
                cand.joints.at("j3"), cand.joints.at("j4"), cand.joints.at("j5"),
                cand.joints.at("j6"), cand.up_error, cand.collision_free ? "YES" : "NO");
  }

  struct ViewStats
  {
    int poses = 0;
    int geom_valid = 0;
    int raw_ik = 0;
    int valid_ik = 0;
    int unique_ik = 0;
    int feasible_rolls = 0;
    bool canonical_valid = false;
    std::vector<double> feasible_angles;
  };
  std::map<std::string, ViewStats> stats;
  for (const auto& name : kRequiredViews)
  {
    stats[name] = ViewStats();
  }
  for (const auto& roll : rolls)
  {
    stats[roll.view].poses += 1;
    stats[roll.view].geom_valid += 1;
  }
  for (const auto& cand : validated)
  {
    stats[cand.view].raw_ik += 1;
    if (cand.valid)
    {
      stats[cand.view].valid_ik += 1;
    }
  }

  int accepted_bound_violations = 0;
  int cross_roll_same_q = 0;
  for (const auto& name : kRequiredViews)
  {
    std::vector<const EndpointCandidate*> valid;
    for (const auto& cand : validated)
    {
      if (cand.view == name && cand.valid)
      {
        valid.push_back(&cand);
        if (!cand.bounds_ok)
        {
          ++accepted_bound_violations;
        }
      }
    }
    std::vector<const EndpointCandidate*> unique;
    for (const auto* cand : valid)
    {
      bool dup = false;
      for (const auto* prev : unique)
      {
        if (jointL2(prev->joints, cand->joints) < min_ik_dist)
        {
          dup = true;
          if (std::abs(prev->roll_deg - cand->roll_deg) > 1e-6)
          {
            double pos = 0.0;
            double ori = 0.0;
            poseError(poseToIso(prev->tcp_target.pose), poseToIso(cand->tcp_target.pose), pos, ori);
            RCLCPP_WARN(node->get_logger(),
                        "same q for different rolls %s %.1f vs %.1f pose_err=%.6f m / %.6f deg",
                        name.c_str(), prev->roll_deg, cand->roll_deg, pos, ori);
            ++cross_roll_same_q;
          }
          break;
        }
      }
      if (!dup)
      {
        unique.push_back(cand);
      }
    }
    stats[name].unique_ik = static_cast<int>(unique.size());
    std::set<int> angles;
    for (const auto* cand : unique)
    {
      angles.insert(static_cast<int>(std::lround(cand->roll_deg)));
      if (std::abs(cand->roll_deg) < 1e-9)
      {
        stats[name].canonical_valid = true;
      }
    }
    stats[name].feasible_rolls = static_cast<int>(angles.size());
    for (int angle : angles)
    {
      stats[name].feasible_angles.push_back(static_cast<double>(angle));
    }
  }

  std::string diversity = "NONE";
  const bool ready = stats["side_pos_y"].feasible_rolls >= 2 &&
                     stats["side_neg_y"].feasible_rolls >= 2 &&
                     stats["top_circle"].feasible_rolls >= 2;
  const bool limited = !ready && (stats["side_pos_y"].feasible_rolls >= 2 ||
                                  stats["side_neg_y"].feasible_rolls >= 2 ||
                                  stats["top_circle"].feasible_rolls >= 2);
  if (ready)
  {
    diversity = "READY";
  }
  else if (limited)
  {
    diversity = "LIMITED";
  }

  int total_valid = 0;
  for (const auto& name : kRequiredViews)
  {
    total_valid += stats[name].valid_ik;
    std::ostringstream angles;
    for (size_t i = 0; i < stats[name].feasible_angles.size(); ++i)
    {
      if (i)
      {
        angles << ",";
      }
      angles << stats[name].feasible_angles[i];
    }
    RCLCPP_INFO(node->get_logger(),
                "VIEW %s poses=%d geom=%d ik_feasible_rolls=%d raw_ik=%d valid_ik=%d unique_ik=%d "
                "canonical0=%s angles=[%s]",
                name.c_str(), stats[name].poses, stats[name].geom_valid, stats[name].feasible_rolls,
                stats[name].raw_ik, stats[name].valid_ik, stats[name].unique_ik,
                stats[name].canonical_valid ? "YES" : "NO", angles.str().c_str());
  }

  writeYaml(output_dir + "/fr3_step10_side_pos_y.yaml", "side_pos_y", rolls, validated);
  writeYaml(output_dir + "/fr3_step10_side_neg_y.yaml", "side_neg_y", rolls, validated);
  writeYaml(output_dir + "/fr3_step10_top_circle.yaml", "top_circle", rolls, validated);
  {
    std::ofstream yaml(output_dir + "/fr3_step10_endpoint_candidates.yaml");
    yaml.setf(std::ios::fixed);
    yaml.precision(6);
    yaml << "step: 10\n";
    yaml << "ik_backend: " << ik_backend << "\n";
    yaml << "same_prefix_task_integration: false\n";
    yaml << "same_lift_scene: true\n";
    yaml << "roll_step_deg: " << roll_step_deg << "\n";
    yaml << "max_ik_solutions_per_pose: " << max_ik << "\n";
    yaml << "min_ik_solution_distance: " << min_ik_dist << "\n";
    yaml << "collision_checking: true\n";
    yaml << "ignore_collisions: false\n";
    yaml << "ik_frame: " << ee_link << "\n";
    yaml << "attached_link: " << attached_link << "\n";
    yaml << "part_table_allowed: false\n";
    yaml << "max_requested_view_center_error: " << max_center << "\n";
    yaml << "max_requested_normal_error: " << max_normal << "\n";
    yaml << "canonical_regression: " << (canonical_regressed ? "PASS" : "FAIL") << "\n";
    yaml << "total_valid_ik: " << total_valid << "\n";
    yaml << "roll_diversity: " << diversity << "\n";
    yaml << "cross_roll_same_q: " << cross_roll_same_q << "\n";
    yaml << "accepted_bounds_violations: " << accepted_bound_violations << "\n";
    yaml << "path_planning_to_endpoints: false\n";
    yaml << "order_search: false\n";
    yaml << "ranking: false\n";
    yaml << "execution: false\n";
    yaml << "views:\n";
    for (const auto& name : kRequiredViews)
    {
      yaml << "  " << name << ":\n";
      yaml << "    roll_poses: " << stats[name].poses << "\n";
      yaml << "    geometry_valid: " << stats[name].geom_valid << "\n";
      yaml << "    ik_feasible_rolls: " << stats[name].feasible_rolls << "\n";
      yaml << "    raw_ik: " << stats[name].raw_ik << "\n";
      yaml << "    valid_ik: " << stats[name].valid_ik << "\n";
      yaml << "    unique_ik: " << stats[name].unique_ik << "\n";
      yaml << "    canonical_0_valid: " << (stats[name].canonical_valid ? "true" : "false") << "\n";
      yaml << "    feasible_roll_angles: [";
      for (size_t i = 0; i < stats[name].feasible_angles.size(); ++i)
      {
        yaml << stats[name].feasible_angles[i];
        if (i + 1 < stats[name].feasible_angles.size())
        {
          yaml << ", ";
        }
      }
      yaml << "]\n";
    }
  }

  const bool all_views_have_ik = stats["side_pos_y"].valid_ik >= 1 &&
                                 stats["side_neg_y"].valid_ik >= 1 &&
                                 stats["top_circle"].valid_ik >= 1;
  const bool more_than_three = total_valid > 3;
  const bool canonical_ok = stats["side_pos_y"].canonical_valid &&
                            stats["side_neg_y"].canonical_valid &&
                            stats["top_circle"].canonical_valid;
  const bool pass = canonical_regressed && all_views_have_ik && more_than_three &&
                    accepted_bound_violations == 0 && attached && !part_table_allowed;
  if (!canonical_ok)
  {
    RCLCPP_ERROR(node->get_logger(),
                 "canonical roll=0 has no valid IK on this Lift scene. Investigate.");
  }
  RCLCPP_INFO(node->get_logger(), "ROLL DIVERSITY: %s", diversity.c_str());
  RCLCPP_INFO(node->get_logger(), "STEP 10 verification: %s", pass ? "PASS" : "FAIL");

  prefix.publishAllSolutions(false);
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
  bool live_table_after = false;
  bool live_column_after = false;
  inspectLiveScene(node, object_id, live_world_after, live_attached_after, live_table_after,
                   live_column_after, table_name, column_name);
  RCLCPP_INFO(node->get_logger(),
              "LIVE SCENE after test: world=%s attached=%s table=%s column=%s",
              live_world_after ? "YES" : "NO", live_attached_after ? "YES" : "NO",
              live_table_after ? "YES" : "NO", live_column_after ? "YES" : "NO");
  if (hold_for_introspection)
  {
    RCLCPP_INFO(node->get_logger(), "Introspection hold. Ctrl+C to exit.");
    spinner.join();
  }
  else
  {
    shutdownSpinner(executor, spinner);
  }
  return pass ? 0 : 4;
}
