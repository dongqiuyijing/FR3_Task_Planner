#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <geometric_shapes/shapes.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <yaml-cpp/yaml.h>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/collision_detection/collision_matrix.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/planning_scene_components.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>

using namespace std::chrono_literals;

namespace
{
constexpr char kExpectedModel[] = "fairino3_dual_robot";
constexpr char kModelFrame[] = "world";
constexpr char kGroupA[] = "arm_a";
constexpr char kGroupB[] = "arm_b";
constexpr char kTcpA[] = "arm_a_gripper_tcp";
constexpr char kTcpB[] = "arm_b_gripper_tcp";
constexpr char kTableName[] = "table";
constexpr char kColumnName[] = "mounting_column";
constexpr char kGripperJointA[] = "arm_a_gripper_joint";
constexpr char kGripperJointB[] = "arm_b_gripper_joint";
constexpr char kSliderLA[] = "arm_a_rail_2_slider_l";
constexpr char kSliderRA[] = "arm_a_rail_2_slider_r";
constexpr char kSliderLB[] = "arm_b_rail_2_slider_l";
constexpr char kSliderRB[] = "arm_b_rail_2_slider_r";

const std::vector<std::string> kArmAJoints = {
    "arm_a_j1", "arm_a_j2", "arm_a_j3", "arm_a_j4", "arm_a_j5", "arm_a_j6"};
const std::vector<std::string> kArmBJoints = {
    "arm_b_j1", "arm_b_j2", "arm_b_j3", "arm_b_j4", "arm_b_j5", "arm_b_j6"};

const std::vector<std::pair<std::string, std::string>> kFocusPairs = {
    {"arm_a_base_link", "mounting_column"},
    {"arm_b_base_link", "mounting_column"},
    {"arm_a_finger_l", "arm_a_finger_r"},
    {"arm_b_finger_l", "arm_b_finger_r"},
    {"arm_a_upperarm_link", "mounting_column"},
    {"arm_b_upperarm_link", "mounting_column"},
};

struct HandoverGeometry
{
  Eigen::Isometry3d world_tcp_a = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d world_tcp_b = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d world_pre_b = Eigen::Isometry3d::Identity();
};

struct IkCandidate
{
  std::vector<double> joints;
  bool within_bounds = false;
  bool fk_ok = false;
  double fk_position_error = 0.0;
  double fk_orientation_error_deg = 0.0;
  std::string seed_name;
};

struct TargetResult
{
  std::string label;
  std::string group;
  std::string tcp;
  Eigen::Isometry3d target = Eigen::Isometry3d::Identity();
  int attempts = 0;
  std::vector<IkCandidate> unique;
  bool pass = false;
  std::string fail_reason;
};

struct SceneIds
{
  std::vector<std::string> world;
  std::vector<std::string> attached;
};

void emit(const std::string& line)
{
  std::cout << line << std::endl;
}

std::string fmtVec(const std::vector<double>& values, int precision = 6)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss << std::setprecision(precision) << "[";
  for (size_t i = 0; i < values.size(); ++i)
  {
    if (i)
      oss << ", ";
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

std::string fmtScalar(double value, int precision = 6)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss << std::setprecision(precision) << value;
  return oss.str();
}

std::string fmtXyz(const Eigen::Vector3d& p)
{
  return fmtVec({p.x(), p.y(), p.z()});
}

std::string joinNames(const std::vector<std::string>& names)
{
  if (names.empty())
    return "(none)";
  std::ostringstream oss;
  for (size_t i = 0; i < names.size(); ++i)
  {
    if (i)
      oss << ", ";
    oss << names[i];
  }
  return oss.str();
}

std::string pairKey(const std::string& a, const std::string& b)
{
  return (a <= b) ? (a + " <-> " + b) : (b + " <-> " + a);
}

bool inList(const std::vector<std::string>& names, const std::string& value)
{
  return std::find(names.begin(), names.end(), value) != names.end();
}

bool startsWith(const std::string& value, const std::string& prefix)
{
  return value.rfind(prefix, 0) == 0;
}

std::string getString(const rclcpp::Node::SharedPtr& node, const std::string& name,
                      const std::string& fallback)
{
  if (!node->has_parameter(name))
    return fallback;
  return node->get_parameter(name).as_string();
}

double getDouble(const rclcpp::Node::SharedPtr& node, const std::string& name, double fallback)
{
  if (!node->has_parameter(name))
    return fallback;
  const auto parameter = node->get_parameter(name);
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
    return parameter.as_double();
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
    return static_cast<double>(parameter.as_int());
  return fallback;
}

int getInt(const rclcpp::Node::SharedPtr& node, const std::string& name, int fallback)
{
  if (!node->has_parameter(name))
    return fallback;
  const auto parameter = node->get_parameter(name);
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
    return static_cast<int>(parameter.as_int());
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
    return static_cast<int>(parameter.as_double());
  return fallback;
}

void setStringParam(const rclcpp::Node::SharedPtr& node, const std::string& name,
                    const std::string& value)
{
  if (!node->has_parameter(name))
    node->declare_parameter<std::string>(name, value);
  else
    node->set_parameter(rclcpp::Parameter(name, value));
}

void setDoubleParam(const rclcpp::Node::SharedPtr& node, const std::string& name, double value)
{
  if (!node->has_parameter(name))
    node->declare_parameter<double>(name, value);
  else
    node->set_parameter(rclcpp::Parameter(name, value));
}

void poseError(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b, double& position_m,
               double& orientation_deg)
{
  position_m = (a.translation() - b.translation()).norm();
  const Eigen::Quaterniond qa(a.rotation());
  const Eigen::Quaterniond qb(b.rotation());
  const double qdot = std::min(1.0, std::abs(qa.normalized().dot(qb.normalized())));
  orientation_deg = 2.0 * std::acos(qdot) * 180.0 / M_PI;
}

std::vector<double> jointsOf(const moveit::core::RobotState& state,
                             const std::vector<std::string>& names)
{
  std::vector<double> values;
  for (const auto& name : names)
    values.push_back(state.getVariablePosition(name));
  return values;
}

double jointL2(const std::vector<double>& a, const std::vector<double>& b)
{
  const size_t n = std::min(a.size(), b.size());
  double acc = 0.0;
  for (size_t i = 0; i < n; ++i)
    acc += (a[i] - b[i]) * (a[i] - b[i]);
  return std::sqrt(acc);
}

int findDuplicate(const std::vector<IkCandidate>& cands, const std::vector<double>& joints,
                  double min_distance)
{
  for (size_t i = 0; i < cands.size(); ++i)
  {
    if (jointL2(cands[i].joints, joints) < min_distance)
      return static_cast<int>(i);
  }
  return -1;
}

Eigen::Isometry3d makeTransform(const Eigen::Matrix3d& R, const Eigen::Vector3d& p)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = R;
  T.translation() = p;
  return T;
}

void applyNamedJoints(moveit::core::RobotState& state, const std::map<std::string, double>& joints)
{
  for (const auto& item : joints)
  {
    if (state.getRobotModel()->hasJointModel(item.first))
      state.setVariablePosition(item.first, item.second);
  }
  state.update();
}

void applyGroupJoints(moveit::core::RobotState& state, const std::vector<std::string>& names,
                      const std::vector<double>& values)
{
  const size_t n = std::min(names.size(), values.size());
  for (size_t i = 0; i < n; ++i)
    state.setVariablePosition(names[i], values[i]);
  state.update();
}

void perturbGroup(moveit::core::RobotState& state, const moveit::core::JointModelGroup* group,
                  double radius, std::mt19937& rng)
{
  std::uniform_real_distribution<double> dist(-radius, radius);
  for (const auto* joint : group->getActiveJointModels())
  {
    double q = state.getVariablePosition(joint->getName()) + dist(rng);
    const auto& bounds = joint->getVariableBounds();
    if (!bounds.empty() && bounds.front().position_bounded_)
    {
      q = std::min(bounds.front().max_position_, std::max(bounds.front().min_position_, q));
    }
    state.setVariablePosition(joint->getName(), q);
  }
  state.update();
}

void randomInBounds(moveit::core::RobotState& state, const moveit::core::JointModelGroup* group,
                    std::mt19937& rng)
{
  for (const auto* joint : group->getActiveJointModels())
  {
    const auto& bounds = joint->getVariableBounds();
    double q = 0.0;
    if (!bounds.empty() && bounds.front().position_bounded_)
    {
      std::uniform_real_distribution<double> dist(bounds.front().min_position_,
                                                  bounds.front().max_position_);
      q = dist(rng);
    }
    else
    {
      std::uniform_real_distribution<double> dist(-M_PI, M_PI);
      q = dist(rng);
    }
    state.setVariablePosition(joint->getName(), q);
  }
  state.update();
}

bool overlayFromMoveGroup(const rclcpp::Node::SharedPtr& node, bool* move_group_ok)
{
  *move_group_ok = false;
  auto client = std::make_shared<rclcpp::SyncParametersClient>(node, "/move_group");
  if (!client->wait_for_service(10s))
    return false;
  *move_group_ok = true;
  const auto values =
      client->get_parameters({"robot_description", "robot_description_semantic"});
  if (values.size() != 2)
    return false;
  for (const auto& parameter : values)
  {
    if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING ||
        parameter.as_string().empty())
      return false;
    setStringParam(node, parameter.get_name(), parameter.as_string());
  }
  return true;
}

bool loadKinematicsYaml(const rclcpp::Node::SharedPtr& node, const std::string& path)
{
  YAML::Node root;
  try
  {
    root = YAML::LoadFile(path);
  }
  catch (...)
  {
    return false;
  }
  if (!root || !root.IsMap())
    return false;
  for (auto git = root.begin(); git != root.end(); ++git)
  {
    const std::string group = git->first.as<std::string>();
    const YAML::Node block = git->second;
    if (!block || !block.IsMap())
      continue;
    for (auto pit = block.begin(); pit != block.end(); ++pit)
    {
      const std::string key = pit->first.as<std::string>();
      const YAML::Node value = pit->second;
      if (!value || !value.IsScalar())
        continue;
      const std::string full = "robot_description_kinematics." + group + "." + key;
      if (key == "kinematics_solver")
        setStringParam(node, full, value.as<std::string>());
      else
        setDoubleParam(node, full, value.as<double>());
    }
  }
  return true;
}

bool loadHandoverGeometry(const std::string& path, HandoverGeometry& geo, std::string& error)
{
  YAML::Node cfg;
  try
  {
    cfg = YAML::LoadFile(path);
  }
  catch (const std::exception& e)
  {
    error = e.what();
    return false;
  }
  if (cfg["execution"] && cfg["execution"]["enabled"].as<bool>())
  {
    error = "execution.enabled is true";
    return false;
  }
  const Eigen::Vector3d object_center(cfg["object"]["center_world"]["x"].as<double>(),
                                      cfg["object"]["center_world"]["y"].as<double>(),
                                      cfg["object"]["center_world"]["z"].as<double>());
  const double arm_b_offset_x = cfg["handover"]["arm_b_grasp_offset_x_m"].as<double>();
  const double pre = cfg["handover"]["prehandover_b_distance_m"].as<double>();
  Eigen::Matrix3d R_world_object;
  R_world_object << 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  Eigen::Matrix3d R_tcpA_object;
  R_tcpA_object << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0;
  const Eigen::Isometry3d world_object = makeTransform(R_world_object, object_center);
  const Eigen::Isometry3d tcp_a_object = makeTransform(R_tcpA_object, Eigen::Vector3d::Zero());
  geo.world_tcp_a = world_object * tcp_a_object.inverse();
  Eigen::Matrix3d R_world_tcpB;
  R_world_tcpB << 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0;
  Eigen::Vector3d b_position = object_center;
  b_position.x() += arm_b_offset_x;
  geo.world_tcp_b = makeTransform(R_world_tcpB, b_position);
  geo.world_pre_b = geo.world_tcp_b;
  geo.world_pre_b.translation().x() -= pre;
  return true;
}

bool loadStep14ArmASeed(const std::string& path, std::vector<double>& joints)
{
  YAML::Node yaml;
  try
  {
    yaml = YAML::LoadFile(path);
  }
  catch (...)
  {
    return false;
  }
  YAML::Node values = yaml["c_joints_rad"];
  if (!values || !values.IsSequence() || values.size() != 6)
    return false;
  joints.clear();
  for (int i = 0; i < 6; ++i)
    joints.push_back(values[i].as<double>());
  return true;
}

bool waitForJointStates(const rclcpp::Node::SharedPtr& node, double timeout_sec, double max_age_sec,
                        sensor_msgs::msg::JointState& out, std::string& error)
{
  sensor_msgs::msg::JointState::SharedPtr latest;
  auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(),
      [&latest](const sensor_msgs::msg::JointState::SharedPtr msg) { latest = msg; });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec);
  while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline)
  {
    if (latest)
    {
      const rclcpp::Time stamp(latest->header.stamp);
      if (stamp.nanoseconds() != 0 && (node->now() - stamp).seconds() > max_age_sec)
      {
        latest.reset();
        std::this_thread::sleep_for(20ms);
        continue;
      }
      out = *latest;
      return true;
    }
    std::this_thread::sleep_for(20ms);
  }
  error = "/joint_states timeout";
  return false;
}

bool extractNamedPositions(const sensor_msgs::msg::JointState& msg,
                           std::map<std::string, double>& joints, std::string& error)
{
  joints.clear();
  const size_t count = std::min(msg.name.size(), msg.position.size());
  for (size_t i = 0; i < count; ++i)
  {
    if (!std::isfinite(msg.position[i]))
    {
      error = "non-finite " + msg.name[i];
      return false;
    }
    joints[msg.name[i]] = msg.position[i];
  }
  return true;
}

void validateCandidate(IkCandidate& cand, const moveit::core::RobotState& seed_state,
                       const moveit::core::JointModelGroup* group, const std::string& tcp,
                       const Eigen::Isometry3d& target, const std::vector<std::string>& joint_names,
                       double pos_tol, double ori_tol_deg)
{
  moveit::core::RobotState fk(seed_state);
  applyGroupJoints(fk, joint_names, cand.joints);
  cand.within_bounds = fk.satisfiesBounds(group);
  poseError(target, fk.getGlobalLinkTransform(tcp), cand.fk_position_error,
            cand.fk_orientation_error_deg);
  cand.fk_ok = cand.fk_position_error <= pos_tol && cand.fk_orientation_error_deg <= ori_tol_deg;
}

void searchIk(TargetResult& result, const moveit::core::RobotState& current,
              const std::vector<std::pair<std::string, std::vector<double>>>& seeds,
              int max_attempts, int max_unique, double timeout_exact, double timeout_nearby,
              double nearby_radius, double nearby_radius_local, double min_distance, double pos_tol,
              double ori_tol_deg, unsigned int rng_seed)
{
  const auto model = current.getRobotModel();
  const auto* group = model->getJointModelGroup(result.group);
  const auto joint_names = (result.group == kGroupA) ? kArmAJoints : kArmBJoints;
  std::mt19937 rng(rng_seed);
  auto consider = [&](moveit::core::RobotState& ik_state, const std::string& seed_name,
                      double timeout) {
    if (result.attempts >= max_attempts || static_cast<int>(result.unique.size()) >= max_unique)
      return;
    ++result.attempts;
    if (!ik_state.setFromIK(group, result.target, result.tcp, timeout))
      return;
    IkCandidate cand;
    cand.joints = jointsOf(ik_state, joint_names);
    cand.seed_name = seed_name;
    validateCandidate(cand, current, group, result.tcp, result.target, joint_names, pos_tol,
                      ori_tol_deg);
    if (!cand.within_bounds || !cand.fk_ok)
      return;
    if (findDuplicate(result.unique, cand.joints, min_distance) >= 0)
      return;
    result.unique.push_back(std::move(cand));
  };
  for (const auto& seed : seeds)
  {
    moveit::core::RobotState exact(current);
    applyGroupJoints(exact, joint_names, seed.second);
    consider(exact, seed.first + "_exact", timeout_exact);
    const int nearby_n = (seed.first == "current") ? 8 : 4;
    for (int i = 0; i < nearby_n; ++i)
    {
      moveit::core::RobotState nearby(current);
      applyGroupJoints(nearby, joint_names, seed.second);
      perturbGroup(nearby, group, (i < nearby_n / 2) ? nearby_radius_local : nearby_radius, rng);
      consider(nearby, seed.first + "_nearby", timeout_nearby);
    }
  }
  while (result.attempts < max_attempts && static_cast<int>(result.unique.size()) < max_unique)
  {
    moveit::core::RobotState random(current);
    randomInBounds(random, group, rng);
    consider(random, "random_bounds", timeout_nearby);
  }
  result.pass = !result.unique.empty();
}

uint32_t fullSceneComponents()
{
  using C = moveit_msgs::msg::PlanningSceneComponents;
  return C::SCENE_SETTINGS | C::ROBOT_STATE | C::ROBOT_STATE_ATTACHED_OBJECTS |
         C::WORLD_OBJECT_NAMES | C::WORLD_OBJECT_GEOMETRY | C::OCTOMAP | C::TRANSFORMS |
         C::ALLOWED_COLLISION_MATRIX | C::LINK_PADDING_AND_SCALING;
}

bool fetchPlanningSceneMsg(const rclcpp::Node::SharedPtr& node, moveit_msgs::msg::PlanningScene& out,
                           std::string& error)
{
  auto client = node->create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");
  if (!client->wait_for_service(10s))
  {
    error = "/get_planning_scene unavailable";
    return false;
  }
  auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
  request->components.components = fullSceneComponents();
  auto future = client->async_send_request(request);
  if (future.wait_for(10s) != std::future_status::ready)
  {
    error = "/get_planning_scene timeout";
    return false;
  }
  auto response = future.get();
  if (!response)
  {
    error = "empty GetPlanningScene";
    return false;
  }
  out = response->scene;
  return true;
}

SceneIds idsFromMsg(const moveit_msgs::msg::PlanningScene& msg)
{
  SceneIds ids;
  for (const auto& object : msg.world.collision_objects)
    ids.world.push_back(object.id);
  for (const auto& attached : msg.robot_state.attached_collision_objects)
    ids.attached.push_back(attached.object.id + "@" + attached.link_name);
  std::sort(ids.world.begin(), ids.world.end());
  std::sort(ids.attached.begin(), ids.attached.end());
  return ids;
}

std::string acmStatus(const collision_detection::AllowedCollisionMatrix& acm,
                      const std::string& a, const std::string& b)
{
  collision_detection::AllowedCollision::Type type;
  if (!acm.getEntry(a, b, type))
    return "ABSENT (default: checked)";
  if (type == collision_detection::AllowedCollision::ALWAYS)
    return "ALWAYS allowed";
  if (type == collision_detection::AllowedCollision::NEVER)
    return "NEVER (checked)";
  return "CONDITIONAL";
}

std::set<std::string> contactPairs(const collision_detection::CollisionResult& result)
{
  std::set<std::string> pairs;
  for (const auto& item : result.contacts)
    pairs.insert(pairKey(item.first.first, item.first.second));
  return pairs;
}

collision_detection::CollisionResult checkRaw(planning_scene::PlanningScene& scene,
                                              moveit::core::RobotState& state, int max_contacts,
                                              int max_per_pair,
                                              const collision_detection::AllowedCollisionMatrix* acm)
{
  collision_detection::CollisionRequest request;
  collision_detection::CollisionResult result;
  request.contacts = true;
  request.max_contacts = static_cast<std::size_t>(max_contacts);
  request.max_contacts_per_pair = static_cast<std::size_t>(max_per_pair);
  request.group_name.clear();
  if (acm)
    scene.checkCollision(request, result, state, *acm);
  else
    scene.checkCollision(request, result, state);
  return result;
}

bool hasPair(const std::set<std::string>& pairs, const std::string& a, const std::string& b)
{
  return pairs.count(pairKey(a, b)) > 0;
}

void printFocus(const std::set<std::string>& pairs)
{
  for (const auto& item : kFocusPairs)
  {
    emit("  " + pairKey(item.first, item.second) + ": " +
         (hasPair(pairs, item.first, item.second) ? "YES" : "NO"));
  }
}

void printAllPairs(const std::set<std::string>& pairs, const std::string& title)
{
  emit(title + " (" + std::to_string(pairs.size()) + " unique pairs):");
  if (pairs.empty())
    emit("  (none)");
  for (const auto& pair : pairs)
    emit("  " + pair);
}

bool isGripperLink(const std::string& name, const std::string& prefix)
{
  static const std::vector<std::string> tails = {
      "gripper_base_link", "rail_155", "slider_l", "slider_r", "finger_l", "finger_r",
      "gripper_gap_link", "gripper_tcp"};
  for (const auto& tail : tails)
  {
    if (name == prefix + tail)
      return true;
  }
  return false;
}

std::set<std::string> interGripperPairs(const std::set<std::string>& pairs)
{
  std::set<std::string> out;
  for (const auto& pair : pairs)
  {
    const auto pos = pair.find(" <-> ");
    if (pos == std::string::npos)
      continue;
    const std::string a = pair.substr(0, pos);
    const std::string b = pair.substr(pos + 5);
    if ((isGripperLink(a, "arm_a_") && isGripperLink(b, "arm_b_")) ||
        (isGripperLink(a, "arm_b_") && isGripperLink(b, "arm_a_")))
    {
      out.insert(pair);
    }
  }
  return out;
}

void setGripper(moveit::core::RobotState& state, const std::string& joint, double value)
{
  if (state.getRobotModel()->hasJointModel(joint))
    state.setVariablePosition(joint, value);
  state.update();
}

void printGripper(const moveit::core::RobotState& state, const std::string& tag)
{
  auto val = [&](const std::string& name) {
    if (!state.getRobotModel()->hasJointModel(name))
      return std::string("MISSING");
    return fmtScalar(state.getVariablePosition(name));
  };
  emit(tag + ":");
  emit("  arm_a_gripper_joint=" + val(kGripperJointA) + "  slider_l=" + val(kSliderLA) +
       "  slider_r=" + val(kSliderRA));
  emit("  arm_b_gripper_joint=" + val(kGripperJointB) + "  slider_l=" + val(kSliderLB) +
       "  slider_r=" + val(kSliderRB));
  if (state.getRobotModel()->hasLinkModel("arm_a_finger_l") &&
      state.getRobotModel()->hasLinkModel("arm_a_finger_r"))
  {
    const Eigen::Vector3d gap_a = state.getGlobalLinkTransform("arm_a_finger_r").translation() -
                                  state.getGlobalLinkTransform("arm_a_finger_l").translation();
    emit("  arm_a finger origin separation: " + fmtScalar(gap_a.norm()) + " m");
  }
  if (state.getRobotModel()->hasLinkModel("arm_b_finger_l") &&
      state.getRobotModel()->hasLinkModel("arm_b_finger_r"))
  {
    const Eigen::Vector3d gap_b = state.getGlobalLinkTransform("arm_b_finger_r").translation() -
                                  state.getGlobalLinkTransform("arm_b_finger_l").translation();
    emit("  arm_b finger origin separation: " + fmtScalar(gap_b.norm()) + " m");
  }
}

void printWorldObject(const planning_scene::PlanningScene& scene, const std::string& id)
{
  const auto object = scene.getWorld()->getObject(id);
  emit("World object " + id + ":");
  if (!object)
  {
    emit("  ABSENT from local scene copy");
    return;
  }
  if (object->shape_poses_.empty())
  {
    emit("  present, but no shape pose");
    return;
  }
  Eigen::Isometry3d T = object->pose_ * object->shape_poses_.front();
  if (!object->global_shape_poses_.empty())
    T = object->global_shape_poses_.front();
  emit("  pose xyz: " + fmtXyz(T.translation()));
  Eigen::Quaterniond q(T.rotation());
  emit("  pose xyzw: " + fmtVec({q.x(), q.y(), q.z(), q.w()}));
  if (!object->shapes_.empty() && object->shapes_.front())
  {
    const auto* shape = object->shapes_.front().get();
    if (shape->type == shapes::BOX)
    {
      const auto* box = static_cast<const shapes::Box*>(shape);
      emit("  box size: " + fmtVec({box->size[0], box->size[1], box->size[2]}));
      emit("  AABB x: " + fmtVec({T.translation().x() - 0.5 * box->size[0],
                                  T.translation().x() + 0.5 * box->size[0]}));
      emit("  AABB y: " + fmtVec({T.translation().y() - 0.5 * box->size[1],
                                  T.translation().y() + 0.5 * box->size[1]}));
      emit("  AABB z: " + fmtVec({T.translation().z() - 0.5 * box->size[2],
                                  T.translation().z() + 0.5 * box->size[2]}));
    }
    else
      emit(std::string("  shape type: ") + std::to_string(static_cast<int>(shape->type)));
  }
}

void printLinkPose(const moveit::core::RobotState& state, const std::string& link)
{
  if (!state.getRobotModel()->hasLinkModel(link))
  {
    emit(link + ": MISSING");
    return;
  }
  const Eigen::Isometry3d T = state.getGlobalLinkTransform(link);
  Eigen::Quaterniond q(T.rotation());
  emit(link + " xyz=" + fmtXyz(T.translation()) + " xyzw=" + fmtVec({q.x(), q.y(), q.z(), q.w()}));
  const Eigen::Vector3d z_axis = T.rotation() * Eigen::Vector3d::UnitZ();
  emit("  +Z_world=" + fmtXyz(z_axis));
}

collision_detection::AllowedCollisionMatrix isolationAcm(
    const collision_detection::AllowedCollisionMatrix& src)
{
  collision_detection::AllowedCollisionMatrix acm = src;
  acm.setEntry("arm_a_base_link", kColumnName, true);
  acm.setEntry("arm_b_base_link", kColumnName, true);
  acm.setEntry("arm_a_finger_l", "arm_a_finger_r", true);
  acm.setEntry("arm_b_finger_l", "arm_b_finger_r", true);
  return acm;
}

int blocked(const std::string& why)
{
  emit("BUILD PASS");
  emit("RUNTIME BLOCKED:");
  emit("existing dual_bringup unavailable");
  emit("reason: " + why);
  emit("Global PlanningScene write issued: NO");
  emit("Real robot motion commands: ZERO");
  emit("Gripper commands: ZERO");
  return 1;
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("dual_collision_source_diagnostic", options);

  rclcpp::executors::MultiThreadedExecutor executor;
  std::thread spinner;
  bool spinning = false;
  auto stop = [&]() {
    if (spinning)
    {
      executor.cancel();
      if (spinner.joinable())
        spinner.join();
      spinning = false;
    }
    if (rclcpp::ok())
      rclcpp::shutdown();
  };

  try
  {
    emit("========== DUAL-4B-2A COLLISION SOURCE DIAGNOSTIC ==========");
    emit("READ ONLY. NO /apply_planning_scene. NO MOTION. NO GRIPPER COMMANDS.");
    emit("This diagnostic does not attach objects and does not change handover geometry.");

    const std::string home = std::getenv("HOME") ? std::getenv("HOME") : "";
    const std::string handover_yaml =
        getString(node, "handover_yaml", home + "/fr_task_ws/src/fr_task_planner/config/dual_handover.yaml");
    const std::string kinematics_yaml = getString(
        node, "kinematics_yaml",
        home + "/fairino_ws/src/fairino3_dual_moveit_config/config/kinematics.yaml");
    const std::string step14_yaml = getString(
        node, "step14_winner_yaml",
        home + "/fr_task_ws/src/fr_task_planner/config/step14_optimized_grasp_winner.yaml");
    const int max_attempts = getInt(node, "max_ik_attempts", 48);
    const int max_unique = getInt(node, "max_unique_candidates", 8);
    const double timeout_exact = getDouble(node, "ik_timeout_exact", 0.25);
    const double timeout_nearby = getDouble(node, "ik_timeout_nearby", 0.05);
    const double nearby_radius = getDouble(node, "nearby_radius", 2.0);
    const double nearby_radius_local = getDouble(node, "nearby_radius_local", 0.8);
    const double min_distance = getDouble(node, "min_ik_solution_distance", 0.1);
    const double pos_tol = getDouble(node, "tcp_position_tol_m", 0.001);
    const double ori_tol = getDouble(node, "tcp_orientation_tol_deg", 1.0);
    const unsigned int rng_seed = static_cast<unsigned int>(getInt(node, "ik_random_seed", 42));
    const int max_contacts = getInt(node, "max_contacts", 1000);
    const int max_per_pair = getInt(node, "max_contacts_per_pair", 20);

    loadKinematicsYaml(node, kinematics_yaml);
    bool move_group_ok = false;
    if (!overlayFromMoveGroup(node, &move_group_ok) || !move_group_ok)
    {
      stop();
      return blocked("existing dual_bringup unavailable");
    }

    executor.add_node(node);
    spinner = std::thread([&executor]() { executor.spin(); });
    spinning = true;

    robot_model_loader::RobotModelLoader loader(node);
    auto model = loader.getModel();
    if (!model || model->getName() != kExpectedModel || model->getModelFrame() != kModelFrame)
    {
      emit("Wrong or missing RobotModel");
      stop();
      return 1;
    }

    std::string error;
    sensor_msgs::msg::JointState js;
    if (!waitForJointStates(node, 10.0, 2.0, js, error))
    {
      stop();
      return blocked(error);
    }
    std::map<std::string, double> named;
    if (!extractNamedPositions(js, named, error))
    {
      emit(error);
      stop();
      return 1;
    }

    moveit_msgs::msg::PlanningScene scene_msg;
    if (!fetchPlanningSceneMsg(node, scene_msg, error))
    {
      stop();
      return blocked(error);
    }
    const SceneIds before = idsFromMsg(scene_msg);

    auto fetched = std::make_shared<planning_scene::PlanningScene>(model);
    const auto srdf_acm = fetched->getAllowedCollisionMatrix();
    fetched->setPlanningSceneMsg(scene_msg);
    applyNamedJoints(fetched->getCurrentStateNonConst(), named);
    auto local = planning_scene::PlanningScene::clone(fetched);

    emit("");
    emit("----------------------------------------");
    emit("SOURCES");
    emit("----------------------------------------");
    emit("RobotModel: live /move_group robot_description");
    emit("Joints: /joint_states overlay onto GetPlanningScene robot_state");
    emit("World geometry: /get_planning_scene WORLD_OBJECT_GEOMETRY");
    emit("ACM: /get_planning_scene ALLOWED_COLLISION_MATRIX");
    emit("SRDF ACM retained only as a comparison copy; not used to replace the scene ACM.");
    emit("GetPlanningScene ACM entry_names: " +
         std::to_string(scene_msg.allowed_collision_matrix.entry_names.size()));
    emit("Local SRDF/model ACM size: " + std::to_string(srdf_acm.getSize()));
    emit("Local scene ACM size after setPlanningSceneMsg: " +
         std::to_string(local->getAllowedCollisionMatrix().getSize()));

    emit("");
    emit("Before world IDs: " + joinNames(before.world));
    emit("Before attached IDs: " + joinNames(before.attached));

    emit("");
    emit("----------------------------------------");
    emit("ACM FOCUS PAIRS");
    emit("----------------------------------------");
    const auto& scene_acm = local->getAllowedCollisionMatrix();
    for (const auto& item : kFocusPairs)
    {
      emit(pairKey(item.first, item.second) + ":");
      emit("  scene ACM: " + acmStatus(scene_acm, item.first, item.second));
      emit("  SRDF-only ACM: " + acmStatus(srdf_acm, item.first, item.second));
    }
    emit("arm_a_finger_l <-> arm_b_finger_l:");
    emit("  scene ACM: " + acmStatus(scene_acm, "arm_a_finger_l", "arm_b_finger_l"));
    emit("arm_a_gripper_base_link <-> arm_b_gripper_base_link:");
    emit("  scene ACM: " +
         acmStatus(scene_acm, "arm_a_gripper_base_link", "arm_b_gripper_base_link"));

    emit("");
    emit("----------------------------------------");
    emit("INSTALLATION GEOMETRY");
    emit("----------------------------------------");
    printWorldObject(*local, kColumnName);
    printWorldObject(*local, kTableName);
    auto& state = local->getCurrentStateNonConst();
    printLinkPose(state, "arm_a_base_link");
    printLinkPose(state, "arm_b_base_link");
    printLinkPose(state, "arm_a_upperarm_link");
    printLinkPose(state, "arm_b_upperarm_link");

    emit("");
    emit("----------------------------------------");
    emit("GRIPPER JOINTS FROM /joint_states");
    emit("----------------------------------------");
    auto jsVal = [&](const std::string& name) {
      auto it = named.find(name);
      return it == named.end() ? std::string("missing from /joint_states") : fmtScalar(it->second);
    };
    emit("  " + std::string(kGripperJointA) + ": " + jsVal(kGripperJointA));
    emit("  " + std::string(kSliderLA) + ": " + jsVal(kSliderLA));
    emit("  " + std::string(kSliderRA) + ": " + jsVal(kSliderRA));
    emit("  " + std::string(kGripperJointB) + ": " + jsVal(kGripperJointB));
    emit("  " + std::string(kSliderLB) + ": " + jsVal(kSliderLB));
    emit("  " + std::string(kSliderRB) + ": " + jsVal(kSliderRB));
    emit("These /joint_states values are NOT claimed as true hardware stroke.");
    printGripper(state, "CURRENT STATE after overlay (NOMINAL if JS missing gripper)");

    emit("");
    emit("----------------------------------------");
    emit("A. BASELINE: current dual-arm state, NO handover object, NO IK");
    emit("----------------------------------------");
    auto baseline = checkRaw(*local, state, max_contacts, max_per_pair, nullptr);
    auto baseline_pairs = contactPairs(baseline);
    emit(std::string("collision flag: ") + (baseline.collision ? "YES" : "NO"));
    emit("contact_count: " + std::to_string(baseline.contact_count) +
         (static_cast<int>(baseline.contact_count) >= max_contacts ? " TRUNCATED" : ""));
    printFocus(baseline_pairs);
    printAllPairs(baseline_pairs, "Baseline unique pairs");

    emit("");
    emit("----------------------------------------");
    emit("B. NOMINAL GRIPPER OPEN vs CLOSED (local RobotState only)");
    emit("----------------------------------------");
    moveit::core::RobotState open_state(state);
    setGripper(open_state, kGripperJointA, 0.0);
    setGripper(open_state, kGripperJointB, 0.0);
    printGripper(open_state, "NOMINAL GRIPPER STATE open=0.0 (SRDF open / URDF lower limit)");
    auto open_res = checkRaw(*local, open_state, max_contacts, max_per_pair, nullptr);
    auto open_pairs = contactPairs(open_res);
    emit("open focus:");
    printFocus(open_pairs);

    moveit::core::RobotState closed_state(state);
    setGripper(closed_state, kGripperJointA, 0.1);
    setGripper(closed_state, kGripperJointB, 0.1);
    printGripper(closed_state, "NOMINAL GRIPPER STATE closed=0.1 (SRDF closed / DUAL-4B-2 test)");
    auto closed_res = checkRaw(*local, closed_state, max_contacts, max_per_pair, nullptr);
    auto closed_pairs = contactPairs(closed_res);
    emit("closed focus:");
    printFocus(closed_pairs);

    emit("STL-derived note (source mesh, not FCL): finger AABB x=[-0.012, 0.0165] m after 0.001 scale.");
    emit("At q=0.1 slider frames are 0.0309 m apart; inner mesh extents cross by ~2 mm.");
    emit("At q=0.0 inner gap is ~0.098 m. 15 mm object implies q≈0.083, not 0.1.");

    emit("");
    emit("----------------------------------------");
    emit("C. ISOLATED REMAINING CONTACTS at DUAL-4B-2 closed grippers");
    emit("----------------------------------------");
    emit("Local diagnostic ACM only (not written globally):");
    emit("  allow arm_a/b_base_link <-> mounting_column");
    emit("  allow same-gripper finger_l <-> finger_r");
    auto iso_acm = isolationAcm(scene_acm);
    auto iso_res = checkRaw(*local, closed_state, max_contacts, max_per_pair, &iso_acm);
    auto iso_pairs = contactPairs(iso_res);
    emit(std::string("remaining collision flag: ") + (iso_res.collision ? "YES" : "NO"));
    printAllPairs(iso_pairs, "Remaining pairs after isolating baseline contacts");

    HandoverGeometry geo;
    if (!loadHandoverGeometry(handover_yaml, geo, error))
    {
      emit("IK skipped, geometry load failed: " + error);
    }
    else
    {
      moveit::core::RobotState current(model);
      current.setToDefaultValues();
      applyNamedJoints(current, named);
      current.update();

      std::vector<double> step14_c;
      std::vector<std::pair<std::string, std::vector<double>>> seeds_a;
      seeds_a.push_back({"current", jointsOf(current, kArmAJoints)});
      if (loadStep14ArmASeed(step14_yaml, step14_c))
        seeds_a.push_back({"step14_c_face", step14_c});
      std::vector<std::pair<std::string, std::vector<double>>> seeds_b;
      seeds_b.push_back({"current", jointsOf(current, kArmBJoints)});

      TargetResult arm_a{"Arm A Handover", kGroupA, kTcpA, geo.world_tcp_a, 0, {}, false, ""};
      TargetResult arm_b_pre{"Arm B PreHandover", kGroupB, kTcpB, geo.world_pre_b, 0, {}, false, ""};
      TargetResult arm_b_han{"Arm B Handover", kGroupB, kTcpB, geo.world_tcp_b, 0, {}, false, ""};
      searchIk(arm_a, current, seeds_a, max_attempts, max_unique, timeout_exact, timeout_nearby,
               nearby_radius, nearby_radius_local, min_distance, pos_tol, ori_tol, rng_seed);
      searchIk(arm_b_pre, current, seeds_b, max_attempts, max_unique, timeout_exact, timeout_nearby,
               nearby_radius, nearby_radius_local, min_distance, pos_tol, ori_tol, rng_seed + 17);
      searchIk(arm_b_han, current, seeds_b, max_attempts, max_unique, timeout_exact, timeout_nearby,
               nearby_radius, nearby_radius_local, min_distance, pos_tol, ori_tol, rng_seed + 31);

      emit("");
      emit("----------------------------------------");
      emit("D. UPPERARM <-> COLUMN BY IK CANDIDATE (no object, current ACM)");
      emit("----------------------------------------");
      emit("A candidates: " + std::to_string(arm_a.unique.size()) +
           "  B pre: " + std::to_string(arm_b_pre.unique.size()) +
           "  B handover: " + std::to_string(arm_b_han.unique.size()));

      auto scan = [&](const std::string& label, const TargetResult& target,
                      const std::vector<std::string>& joint_names, const std::string& upperarm) {
        emit("");
        emit(label + ":");
        if (target.unique.empty())
          emit("  (no IK)");
        for (size_t i = 0; i < target.unique.size(); ++i)
        {
          moveit::core::RobotState cand(state);
          applyGroupJoints(cand, joint_names, target.unique[i].joints);
          auto raw = checkRaw(*local, cand, max_contacts, max_per_pair, nullptr);
          auto pairs = contactPairs(raw);
          auto iso = checkRaw(*local, cand, max_contacts, max_per_pair, &iso_acm);
          auto iso_p = contactPairs(iso);
          emit("  candidate " + std::to_string(i + 1) + " seed=" + target.unique[i].seed_name);
          emit("    " + upperarm + " <-> column: " +
               (hasPair(pairs, upperarm, kColumnName) ? "YES" : "NO"));
          emit("    base/finger baseline still present: " +
               std::string((hasPair(pairs, "arm_a_base_link", kColumnName) ||
                            hasPair(pairs, "arm_b_base_link", kColumnName) ||
                            hasPair(pairs, "arm_a_finger_l", "arm_a_finger_r") ||
                            hasPair(pairs, "arm_b_finger_l", "arm_b_finger_r")) ?
                               "YES" :
                               "NO"));
          emit("    remaining after isolating baseline: " + joinNames(std::vector<std::string>(
                                                                iso_p.begin(), iso_p.end())));
        }
      };
      scan("Arm A Handover", arm_a, kArmAJoints, "arm_a_upperarm_link");
      scan("Arm B PreHandover", arm_b_pre, kArmBJoints, "arm_b_upperarm_link");
      scan("Arm B Handover", arm_b_han, kArmBJoints, "arm_b_upperarm_link");

      emit("");
      emit("----------------------------------------");
      emit("E. INTER-GRIPPER at first A+B handover combo");
      emit("----------------------------------------");
      if (arm_a.unique.empty() || arm_b_han.unique.empty())
        emit("skipped: missing IK");
      else
      {
        for (double q : {0.0, 0.1})
        {
          moveit::core::RobotState combo(state);
          applyGroupJoints(combo, kArmAJoints, arm_a.unique.front().joints);
          applyGroupJoints(combo, kArmBJoints, arm_b_han.unique.front().joints);
          setGripper(combo, kGripperJointA, q);
          setGripper(combo, kGripperJointB, q);
          auto raw = checkRaw(*local, combo, max_contacts, max_per_pair, nullptr);
          auto pairs = contactPairs(raw);
          auto ig = interGripperPairs(pairs);
          emit(std::string("NOMINAL GRIPPER STATE q=") + fmtScalar(q));
          emit("  truncated: " +
               std::string(static_cast<int>(raw.contact_count) >= max_contacts ? "YES" : "NO"));
          emit("  contact_count: " + std::to_string(raw.contact_count));
          emit("  unique pairs: " + std::to_string(pairs.size()));
          emit("  modeled A-gripper <-> B-gripper pairs: " +
               (ig.empty() ? std::string("(none listed)") : joinNames(std::vector<std::string>(
                                                                ig.begin(), ig.end()))));
          if (static_cast<int>(raw.contact_count) >= max_contacts)
            emit("  Inter-gripper collision: INCONCLUSIVE — contact enumeration incomplete");
        }
        moveit::core::RobotState combo(state);
        applyGroupJoints(combo, kArmAJoints, arm_a.unique.front().joints);
        applyGroupJoints(combo, kArmBJoints, arm_b_han.unique.front().joints);
        setGripper(combo, kGripperJointA, 0.1);
        setGripper(combo, kGripperJointB, 0.0);
        auto mixed = checkRaw(*local, combo, max_contacts, max_per_pair, &iso_acm);
        auto mixed_pairs = contactPairs(mixed);
        auto mixed_ig = interGripperPairs(mixed_pairs);
        emit("NOMINAL mixed: A closed 0.1, B open 0.0, baseline isolated");
        emit("  remaining pairs: " + joinNames(std::vector<std::string>(mixed_pairs.begin(),
                                                                        mixed_pairs.end())));
        emit("  inter-gripper: " +
             (mixed_ig.empty() ? std::string("(none listed)") :
                                 joinNames(std::vector<std::string>(mixed_ig.begin(), mixed_ig.end()))));
      }

      int combo_total = 0;
      int combo_only_baseline = 0;
      int combo_has_upperarm = 0;
      int combo_has_other = 0;
      for (const auto& a : arm_a.unique)
      {
        for (const auto& b : arm_b_han.unique)
        {
          ++combo_total;
          moveit::core::RobotState combo(state);
          applyGroupJoints(combo, kArmAJoints, a.joints);
          applyGroupJoints(combo, kArmBJoints, b.joints);
          setGripper(combo, kGripperJointA, 0.1);
          setGripper(combo, kGripperJointB, 0.1);
          auto raw = contactPairs(checkRaw(*local, combo, max_contacts, max_per_pair, nullptr));
          const bool upper =
              hasPair(raw, "arm_a_upperarm_link", kColumnName) ||
              hasPair(raw, "arm_b_upperarm_link", kColumnName);
          auto iso = contactPairs(checkRaw(*local, combo, max_contacts, max_per_pair, &iso_acm));
          if (upper)
            ++combo_has_upperarm;
          if (iso.empty())
            ++combo_only_baseline;
          else
            ++combo_has_other;
        }
      }
      emit("");
      emit("A Handover x B Handover combinations at DUAL-4B-2 closed=0.1:");
      emit("  tested: " + std::to_string(combo_total));
      emit("  only baseline pairs (base-column and/or same-gripper fingers): " +
           std::to_string(combo_only_baseline));
      emit("  also upperarm <-> column: " + std::to_string(combo_has_upperarm));
      emit("  remaining non-baseline pairs after isolation: " + std::to_string(combo_has_other));
    }

    emit("");
    emit("----------------------------------------");
    emit("F. GLOBAL PLANNING SCENE ID CHECK");
    emit("----------------------------------------");
    std::this_thread::sleep_for(2s);
    moveit_msgs::msg::PlanningScene after_msg;
    const bool after_ok = fetchPlanningSceneMsg(node, after_msg, error);
    if (!after_ok)
      emit("post-check GetPlanningScene failed: " + error);
    else
    {
      const SceneIds after = idsFromMsg(after_msg);
      emit("After world IDs: " + joinNames(after.world));
      emit("After attached IDs: " + joinNames(after.attached));
      std::vector<std::string> added_w, removed_w, added_a, removed_a;
      std::set_difference(after.world.begin(), after.world.end(), before.world.begin(),
                          before.world.end(), std::back_inserter(added_w));
      std::set_difference(before.world.begin(), before.world.end(), after.world.begin(),
                          after.world.end(), std::back_inserter(removed_w));
      std::set_difference(after.attached.begin(), after.attached.end(), before.attached.begin(),
                          before.attached.end(), std::back_inserter(added_a));
      std::set_difference(before.attached.begin(), before.attached.end(), after.attached.begin(),
                          after.attached.end(), std::back_inserter(removed_a));
      emit("Added world IDs: " + joinNames(added_w));
      emit("Removed world IDs: " + joinNames(removed_w));
      emit("Added attached IDs: " + joinNames(added_a));
      emit("Removed attached IDs: " + joinNames(removed_a));
      const bool changed = !(added_w.empty() && removed_w.empty() && added_a.empty() &&
                             removed_a.empty());
      emit(std::string("Global scene changed during test: ") + (changed ? "YES" : "NO"));
      if (changed)
        emit("Cause: UNDETERMINED from IDs alone");
    }
    emit("This diagnostic issued global scene write: NO, based on verified code path");
    emit("DUAL-4B-2 / this diagnostic create only GetPlanningScene clients.");
    emit("Observed writers on this machine include workcell_scene_loader (ApplyPlanningScene).");

    emit("");
    emit("Handover geometry changed: NO");
    emit("URDF/SRDF changed: NO");
    emit("Original collision rules changed: NO");
    emit("Global PlanningScene write issued: NO");
    emit("Real robot motion commands: ZERO");
    emit("Gripper commands: ZERO");
    emit("DUAL-4B-3 started: NO");
    stop();
    return 0;
  }
  catch (const std::exception& e)
  {
    emit(std::string("exception: ") + e.what());
    try
    {
      stop();
    }
    catch (...)
    {
      if (rclcpp::ok())
        rclcpp::shutdown();
    }
    return 1;
  }
}
