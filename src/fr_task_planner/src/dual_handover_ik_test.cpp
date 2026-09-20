
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
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <yaml-cpp/yaml.h>

#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
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

const std::vector<std::string> kArmAJoints = {
    "arm_a_j1", "arm_a_j2", "arm_a_j3", "arm_a_j4", "arm_a_j5", "arm_a_j6"};
const std::vector<std::string> kArmBJoints = {
    "arm_b_j1", "arm_b_j2", "arm_b_j3", "arm_b_j4", "arm_b_j5", "arm_b_j6"};

struct PoseXYZW
{
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
};

struct HandoverGeometry
{
  Eigen::Isometry3d world_object = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d world_tcp_a = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d world_tcp_b = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d world_pre_b = Eigen::Isometry3d::Identity();
  double arm_b_offset_x = 0.0;
  double prehandover_distance = 0.0;
  bool yaml_matches_dual4a_doc = true;
  std::string yaml_diff;
};

struct IkCandidate
{
  std::vector<double> joints;
  bool within_bounds = false;
  bool fk_ok = false;
  double fk_position_error = 0.0;
  double fk_orientation_error_deg = 0.0;
  double seed_l2 = 0.0;
  std::string seed_name;
};

struct TargetResult
{
  std::string label;
  std::string group;
  std::string tcp;
  Eigen::Isometry3d target = Eigen::Isometry3d::Identity();
  std::string solver_name = "MISSING";
  std::string solver_tip;
  std::string solver_base;
  int attempts = 0;
  int ik_success = 0;
  std::vector<IkCandidate> unique;
  bool pass = false;
  std::string fail_reason;
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

std::string fmtXyz(const Eigen::Vector3d& p)
{
  return fmtVec({p.x(), p.y(), p.z()});
}

Eigen::Quaterniond rotationToQuaternion(const Eigen::Matrix3d& R)
{
  const double trace = R.trace();
  double w, x, y, z;

  if (trace > 0.0)
  {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    w = 0.25 * s;
    x = (R(2, 1) - R(1, 2)) / s;
    y = (R(0, 2) - R(2, 0)) / s;
    z = (R(1, 0) - R(0, 1)) / s;
  }
  else
  {
    int i = 0;
    if (R(1, 1) > R(0, 0))
      i = 1;
    if (R(2, 2) > R(i, i))
      i = 2;

    if (i == 0)
    {
      const double s = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;
      w = (R(2, 1) - R(1, 2)) / s;
      x = 0.25 * s;
      y = (R(0, 1) + R(1, 0)) / s;
      z = (R(0, 2) + R(2, 0)) / s;
    }
    else if (i == 1)
    {
      const double s = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;
      w = (R(0, 2) - R(2, 0)) / s;
      x = (R(0, 1) + R(1, 0)) / s;
      y = 0.25 * s;
      z = (R(1, 2) + R(2, 1)) / s;
    }
    else
    {
      const double s = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;
      w = (R(1, 0) - R(0, 1)) / s;
      x = (R(0, 2) + R(2, 0)) / s;
      y = (R(1, 2) + R(2, 1)) / s;
      z = 0.25 * s;
    }
  }

  Eigen::Quaterniond q(w, x, y, z);
  q.normalize();
  return q;
}

std::string fmtXyzw(const Eigen::Quaterniond& q)
{
  return fmtVec({q.x(), q.y(), q.z(), q.w()});
}

PoseXYZW posePrint(const Eigen::Isometry3d& T)
{
  PoseXYZW out;
  out.xyz = T.translation();
  out.q = rotationToQuaternion(T.rotation());
  return out;
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

bool nearlyEqual(double a, double b, double tol)
{
  return std::abs(a - b) <= tol;
}

std::vector<double> jointsOf(const moveit::core::RobotState& state,
                             const std::vector<std::string>& names)
{
  std::vector<double> values;
  values.reserve(names.size());
  for (const auto& name : names)
    values.push_back(state.getVariablePosition(name));
  return values;
}

double jointL2(const std::vector<double>& a, const std::vector<double>& b)
{
  const size_t n = std::min(a.size(), b.size());
  double acc = 0.0;
  for (size_t i = 0; i < n; ++i)
  {
    const double dq = a[i] - b[i];
    acc += dq * dq;
  }
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

bool checkRotation(const Eigen::Matrix3d& R, const std::string& name, std::string& error)
{
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  if ((R.transpose() * R - I).norm() > 1e-8)
  {
    error = name + ": rotation is not orthogonal";
    return false;
  }
  if (std::abs(R.determinant() - 1.0) > 1e-8)
  {
    error = name + ": invalid rotation determinant";
    return false;
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
    error = std::string("Failed to load handover YAML: ") + e.what();
    return false;
  }

  if (!cfg["execution"] || !cfg["execution"]["enabled"])
  {
    error = "handover YAML missing execution.enabled";
    return false;
  }
  if (cfg["execution"]["enabled"].as<bool>())
  {
    error = "DUAL-4B-1 must remain IK-only; execution.enabled is true";
    return false;
  }

  const Eigen::Vector3d object_center(
      cfg["object"]["center_world"]["x"].as<double>(),
      cfg["object"]["center_world"]["y"].as<double>(),
      cfg["object"]["center_world"]["z"].as<double>());
  const double length = cfg["object"]["length_m"].as<double>();
  geo.arm_b_offset_x = cfg["handover"]["arm_b_grasp_offset_x_m"].as<double>();
  geo.prehandover_distance = cfg["handover"]["prehandover_b_distance_m"].as<double>();

  std::ostringstream diff;
  if (!nearlyEqual(object_center.x(), 0.0, 1e-12) ||
      !nearlyEqual(object_center.y(), 0.4, 1e-12) ||
      !nearlyEqual(object_center.z(), 0.9, 1e-12))
  {
    geo.yaml_matches_dual4a_doc = false;
    diff << "object.center_world differs from DUAL-4A (0, 0.4, 0.9); ";
  }
  if (!nearlyEqual(geo.arm_b_offset_x, -0.008, 1e-12))
  {
    geo.yaml_matches_dual4a_doc = false;
    diff << "arm_b_grasp_offset_x_m=" << geo.arm_b_offset_x << " (DUAL-4A used -0.008); ";
  }
  if (!nearlyEqual(geo.prehandover_distance, 0.20, 1e-12))
  {
    geo.yaml_matches_dual4a_doc = false;
    diff << "prehandover_b_distance_m=" << geo.prehandover_distance
         << " (DUAL-4A used 0.20); ";
  }
  geo.yaml_diff = diff.str();

  if (!(geo.arm_b_offset_x > -length / 2.0 && geo.arm_b_offset_x < 0.0))
  {
    error = "Arm B offset must be inside the negative-X half of the object";
    return false;
  }
  if (geo.prehandover_distance <= 0.0)
  {
    error = "PreHandover distance must be positive";
    return false;
  }

  // Same R_world_object as dual_handover_geometry.py:
  // object +Z -> world +X, +X -> world +Y, +Y -> world +Z.
  Eigen::Matrix3d R_world_object;
  R_world_object << 0.0, 0.0, 1.0,
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0;
  if (!checkRotation(R_world_object, "R_world_object", error))
    return false;
  geo.world_object = makeTransform(R_world_object, object_center);

  // Existing Arm A grasp: T_tcpA_object = Rx(pi).
  Eigen::Matrix3d R_tcpA_object;
  R_tcpA_object << 1.0, 0.0, 0.0,
                   0.0, -1.0, 0.0,
                   0.0, 0.0, -1.0;
  const Eigen::Isometry3d T_tcpA_object =
      makeTransform(R_tcpA_object, Eigen::Vector3d::Zero());
  geo.world_tcp_a = geo.world_object * T_tcpA_object.inverse();

  // Arm B: TCP +Z -> world +X, +X -> world +Z, +Y -> world -Y.
  Eigen::Matrix3d R_world_tcpB;
  R_world_tcpB << 0.0, 0.0, 1.0,
                  0.0, -1.0, 0.0,
                  1.0, 0.0, 0.0;
  if (!checkRotation(R_world_tcpB, "R_world_tcpB", error))
    return false;
  Eigen::Vector3d b_position = object_center;
  b_position.x() += geo.arm_b_offset_x;
  geo.world_tcp_b = makeTransform(R_world_tcpB, b_position);

  geo.world_pre_b = geo.world_tcp_b;
  geo.world_pre_b.translation().x() -= geo.prehandover_distance;

  const Eigen::Vector3d a_tip = geo.world_tcp_a.rotation() * Eigen::Vector3d(0.0, 0.0, 1.0);
  const Eigen::Vector3d b_tip = geo.world_tcp_b.rotation() * Eigen::Vector3d(0.0, 0.0, 1.0);
  if ((a_tip - Eigen::Vector3d(-1.0, 0.0, 0.0)).norm() > 1e-8)
  {
    error = "Arm A fingertip is not world -X";
    return false;
  }
  if ((b_tip - Eigen::Vector3d(1.0, 0.0, 0.0)).norm() > 1e-8)
  {
    error = "Arm B fingertip is not world +X";
    return false;
  }

  return true;
}

bool loadStep14ArmASeed(const std::string& path, std::vector<double>& joints, std::string& error)
{
  YAML::Node yaml;
  try
  {
    yaml = YAML::LoadFile(path);
  }
  catch (const std::exception& e)
  {
    error = std::string("STEP14 winner YAML unreadable: ") + e.what();
    return false;
  }

  YAML::Node values = yaml["c_joints_rad"];
  if (!values || !values.IsSequence() || values.size() != 6)
  {
    error = "STEP14 winner YAML missing c_joints_rad[6]";
    return false;
  }

  joints.clear();
  for (int i = 0; i < 6; ++i)
  {
    const double q = values[i].as<double>();
    if (!std::isfinite(q))
    {
      error = "STEP14 C-face seed has a non-finite joint";
      return false;
    }
    joints.push_back(q);
  }
  return true;
}

bool loadKinematicsYaml(const rclcpp::Node::SharedPtr& node, const std::string& path,
                        std::string& error)
{
  YAML::Node root;
  try
  {
    root = YAML::LoadFile(path);
  }
  catch (const std::exception& e)
  {
    error = std::string("Failed to load kinematics YAML: ") + e.what();
    return false;
  }

  if (!root || !root.IsMap())
  {
    error = "kinematics YAML is not a map";
    return false;
  }

  for (auto git = root.begin(); git != root.end(); ++git)
  {
    const std::string group = git->first.as<std::string>();
    const YAML::Node block = git->second;
    if (!block || !block.IsMap())
      continue;

    for (auto pit = block.begin(); pit != block.end(); ++pit)
    {
      const std::string key = pit->first.as<std::string>();
      const std::string full = "robot_description_kinematics." + group + "." + key;
      const YAML::Node value = pit->second;
      if (!value || !value.IsScalar())
        continue;

      if (key == "kinematics_solver")
        setStringParam(node, full, value.as<std::string>());
      else
        setDoubleParam(node, full, value.as<double>());
    }
  }
  return true;
}

// SyncParametersClient creates a private executor and add_node()s the given
// node while waiting for get_parameters(). That is only safe BEFORE this
// test node is added to our MultiThreadedExecutor.
bool overlayFromMoveGroup(const rclcpp::Node::SharedPtr& node, bool* move_group_ok)
{
  *move_group_ok = false;
  auto client = std::make_shared<rclcpp::SyncParametersClient>(node, "/move_group");
  if (!client->wait_for_service(10s))
    return false;

  *move_group_ok = true;

  const auto values = client->get_parameters(
      {"robot_description", "robot_description_semantic"});
  if (values.size() != 2)
    return false;

  for (const auto& parameter : values)
  {
    if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING ||
        parameter.as_string().empty())
    {
      return false;
    }
    setStringParam(node, parameter.get_name(), parameter.as_string());
  }

  const std::vector<std::string> groups = {kGroupA, kGroupB};
  const std::vector<std::string> keys = {
      "kinematics_solver",
      "kinematics_solver_search_resolution",
      "kinematics_solver_timeout"};

  try
  {
    std::vector<std::string> names;
    for (const auto& group : groups)
    {
      for (const auto& key : keys)
        names.push_back("robot_description_kinematics." + group + "." + key);
    }
    const auto kin = client->get_parameters(names);
    for (const auto& parameter : kin)
    {
      if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET)
        continue;
      if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
        setStringParam(node, parameter.get_name(), parameter.as_string());
      else if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
        setDoubleParam(node, parameter.get_name(), parameter.as_double());
    }
  }
  catch (const std::exception& e)
  {
    emit(std::string("STAGE overlay: /move_group kinematics params skipped: ") +
         e.what());
  }

  return true;
}

std::string solverParam(const rclcpp::Node::SharedPtr& node, const std::string& group)
{
  const std::string name = "robot_description_kinematics." + group + ".kinematics_solver";
  return getString(node, name, "");
}

bool waitForJointStates(const rclcpp::Node::SharedPtr& node, double timeout_sec, double max_age_sec,
                        sensor_msgs::msg::JointState& out, std::string& error)
{
  // Subscription callbacks are delivered by the single background executor.
  sensor_msgs::msg::JointState::SharedPtr latest;
  auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(),
      [&latest](const sensor_msgs::msg::JointState::SharedPtr msg) { latest = msg; });

  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(timeout_sec);
  while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline)
  {
    if (latest)
    {
      const auto received_at = std::chrono::steady_clock::now();
      (void)received_at;

      const rclcpp::Time stamp(latest->header.stamp);
      const bool stamp_valid = stamp.nanoseconds() != 0;
      if (stamp_valid)
      {
        const double age = (node->now() - stamp).seconds();
        if (age > max_age_sec)
        {
          latest.reset();
          std::this_thread::sleep_for(20ms);
          continue;
        }
      }

      out = *latest;
      return true;
    }
    std::this_thread::sleep_for(20ms);
  }

  error = "/joint_states timeout or incomplete";
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
      error = "Non-finite position for joint " + msg.name[i];
      return false;
    }
    joints[msg.name[i]] = msg.position[i];
  }
  return true;
}

bool requireArmJoints(const std::map<std::string, double>& joints, std::string& error)
{
  for (const auto& names : {kArmAJoints, kArmBJoints})
  {
    for (const auto& name : names)
    {
      auto it = joints.find(name);
      if (it == joints.end())
      {
        error = "Missing joint in /joint_states: " + name;
        return false;
      }
      if (!std::isfinite(it->second))
      {
        error = "Non-finite joint: " + name;
        return false;
      }
    }
  }
  return true;
}

void applyNamedJoints(moveit::core::RobotState& state,
                      const std::map<std::string, double>& joints)
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
      q = std::min(bounds.front().max_position_,
                   std::max(bounds.front().min_position_, q));
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

bool describeSolver(const rclcpp::Node::SharedPtr& node, const moveit::core::RobotModelConstPtr& model,
                    const std::string& group_name, const std::string& tcp_name, TargetResult& result)
{
  const auto* group = model->getJointModelGroup(group_name);
  if (!group)
  {
    result.fail_reason = "missing group " + group_name;
    return false;
  }
  if (!model->hasLinkModel(tcp_name))
  {
    result.fail_reason = "missing TCP " + tcp_name;
    return false;
  }

  const auto solver = group->getSolverInstance();
  result.solver_name = solverParam(node, group_name);
  if (result.solver_name.empty())
    result.solver_name = "UNSET";

  if (!solver)
  {
    result.fail_reason = "no IK solver loaded for " + group_name;
    return false;
  }

  result.solver_tip = solver->getTipFrame();
  result.solver_base = solver->getBaseFrame();
  if (!result.solver_name.empty() && result.solver_name != "UNSET")
  {
    result.solver_name += " (tip=" + result.solver_tip + ", base=" + result.solver_base + ")";
  }

  if (!group->canSetStateFromIK(tcp_name))
  {
    result.fail_reason = group_name + " cannot setFromIK for " + tcp_name +
                         " (solver tip=" + result.solver_tip + ")";
    return false;
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

  const Eigen::Isometry3d actual = fk.getGlobalLinkTransform(tcp);
  poseError(target, actual, cand.fk_position_error, cand.fk_orientation_error_deg);
  cand.fk_ok = cand.fk_position_error <= pos_tol &&
               cand.fk_orientation_error_deg <= ori_tol_deg;
  cand.seed_l2 = jointL2(cand.joints, jointsOf(seed_state, joint_names));
}

void searchIk(TargetResult& result, const moveit::core::RobotState& current,
              const std::vector<std::pair<std::string, std::vector<double>>>& seeds,
              int max_attempts, int max_unique, double timeout_exact, double timeout_nearby,
              double nearby_radius, double nearby_radius_local, double min_distance,
              double pos_tol, double ori_tol_deg, unsigned int rng_seed)
{
  const auto model = current.getRobotModel();
  const auto* group = model->getJointModelGroup(result.group);
  const auto joint_names = (result.group == kGroupA) ? kArmAJoints : kArmBJoints;
  std::mt19937 rng(rng_seed);

  auto consider = [&](moveit::core::RobotState& ik_state, const std::string& seed_name,
                      double timeout) {
    if (result.attempts >= max_attempts)
      return;
    if (static_cast<int>(result.unique.size()) >= max_unique)
      return;

    ++result.attempts;
    if (!ik_state.setFromIK(group, result.target, result.tcp, timeout))
      return;

    ++result.ik_success;
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
    if (result.attempts >= max_attempts ||
        static_cast<int>(result.unique.size()) >= max_unique)
    {
      break;
    }

    moveit::core::RobotState exact(current);
    applyGroupJoints(exact, joint_names, seed.second);
    consider(exact, seed.first + "_exact", timeout_exact);

    const int nearby_n = (seed.first == "current") ? 8 : 4;
    for (int i = 0; i < nearby_n; ++i)
    {
      if (result.attempts >= max_attempts ||
          static_cast<int>(result.unique.size()) >= max_unique)
      {
        break;
      }
      moveit::core::RobotState nearby(current);
      applyGroupJoints(nearby, joint_names, seed.second);
      const double radius = (i < nearby_n / 2) ? nearby_radius_local : nearby_radius;
      perturbGroup(nearby, group, radius, rng);
      consider(nearby, seed.first + "_nearby", timeout_nearby);
    }
  }

  while (result.attempts < max_attempts &&
         static_cast<int>(result.unique.size()) < max_unique)
  {
    moveit::core::RobotState random(current);
    randomInBounds(random, group, rng);
    consider(random, "random_bounds", timeout_nearby);
  }

  result.pass = !result.unique.empty();
  if (!result.pass)
  {
    result.fail_reason = "no unique in-bounds FK-valid IK";
  }
}

void printTarget(const TargetResult& result)
{
  const auto printed = posePrint(result.target);
  emit("");
  emit(result.label + ":");
  emit("  target xyz: " + fmtXyz(printed.xyz));
  emit("  target xyzw: " + fmtXyzw(printed.q));
  emit("  IK solver: " + result.solver_name);
  emit("  attempts: " + std::to_string(result.attempts));
  emit("  raw IK success: " + std::to_string(result.ik_success));
  emit("  unique candidates: " + std::to_string(result.unique.size()));
  if (!result.fail_reason.empty() && !result.pass)
    emit("  fail reason: " + result.fail_reason);

  for (size_t i = 0; i < result.unique.size(); ++i)
  {
    const auto& cand = result.unique[i];
    std::vector<double> deg;
    deg.reserve(cand.joints.size());
    for (double q : cand.joints)
      deg.push_back(q * 180.0 / M_PI);

    emit("  candidate " + std::to_string(i + 1) + ":");
    emit("    seed: " + cand.seed_name);
    emit("    joints_rad: " + fmtVec(cand.joints));
    emit("    joints_deg: " + fmtVec(deg, 3));
    emit(std::string("    within_bounds: ") + (cand.within_bounds ? "true" : "false"));
    std::ostringstream fk;
    fk.setf(std::ios::fixed);
    fk << std::setprecision(6) << cand.fk_position_error;
    emit("    FK position error: " + fk.str() + " m");
    fk.str("");
    fk << std::setprecision(6) << cand.fk_orientation_error_deg;
    emit("    FK orientation error: " + fk.str() + " deg");
    fk.str("");
    fk << std::setprecision(6) << cand.seed_l2;
    emit("    joint L2 from seed-state: " + fk.str() + " rad");
  }
}

int blocked(const std::string& why)
{
  emit("BUILD PASS");
  emit("RUNTIME TEST BLOCKED: " + why);
  emit("Collision check: NOT PERFORMED");
  emit("Gripper interference check: NOT PERFORMED");
  emit("Grasp feasibility: NOT VERIFIED");
  emit("Real robot commands sent: ZERO");
  return 1;
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("dual_handover_ik_test", options);

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
  emit("========== DUAL-4B-1 IK ONLY ==========");
  emit("PLAN / IK ONLY. NO EXECUTION. NO GRIPPER COMMANDS.");
  emit("STAGE: init");

  const std::string handover_yaml = getString(
      node, "handover_yaml",
      std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
          "/fr_task_ws/src/fr_task_planner/config/dual_handover.yaml");
  const std::string kinematics_yaml = getString(
      node, "kinematics_yaml",
      std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
          "/fairino_ws/src/fairino3_dual_moveit_config/config/kinematics.yaml");
  const std::string step14_yaml = getString(
      node, "step14_winner_yaml",
      std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
          "/fr_task_ws/src/fr_task_planner/config/step14_optimized_grasp_winner.yaml");

  const int max_attempts = getInt(node, "max_ik_attempts", 48);
  const int max_unique = getInt(node, "max_unique_candidates", 8);
  const double timeout_exact = getDouble(node, "ik_timeout_exact", 0.25);
  const double timeout_nearby = getDouble(node, "ik_timeout_nearby", 0.05);
  const double nearby_radius = getDouble(node, "nearby_radius", 2.0);
  const double nearby_radius_local = getDouble(node, "nearby_radius_local", 0.8);
  const double min_distance = getDouble(node, "min_ik_solution_distance", 0.1);
  const double pos_tol = getDouble(node, "tcp_position_tol_m", 0.001);
  const double ori_tol = getDouble(node, "tcp_orientation_tol_deg", 1.0);
  const double joint_timeout = getDouble(node, "joint_state_timeout_sec", 10.0);
  const double joint_max_age = getDouble(node, "joint_state_max_age_sec", 2.0);
  const unsigned int rng_seed =
      static_cast<unsigned int>(getInt(node, "ik_random_seed", 42));

  emit("STAGE: load handover geometry");
  HandoverGeometry geo;
  std::string error;
  if (!loadHandoverGeometry(handover_yaml, geo, error))
  {
    emit("Geometry load failed: " + error);
    stop();
    return 1;
  }

  if (!geo.yaml_matches_dual4a_doc)
  {
    emit("NOTE: local dual_handover.yaml differs from documented DUAL-4A values.");
    emit("  " + geo.yaml_diff);
    emit("  Using the local YAML as-is. Geometry was not reverted.");
  }

  emit("STAGE: load kinematics YAML");
  std::string kin_error;
  const bool kin_file_ok = loadKinematicsYaml(node, kinematics_yaml, kin_error);
  if (!kin_file_ok)
  {
    emit("Kinematics YAML note: " + kin_error);
  }

  emit("STAGE: overlay robot_description from /move_group");
  bool move_group_ok = false;
  if (!overlayFromMoveGroup(node, &move_group_ok) || !move_group_ok)
  {
    stop();
    return blocked("existing dual_bringup unavailable");
  }
  emit("STAGE: overlay robot_description complete");

  emit("STAGE: start executor");
  executor.add_node(node);
  spinner = std::thread([&executor]() { executor.spin(); });
  spinning = true;

  emit("STAGE: load RobotModel");
  robot_model_loader::RobotModelLoader loader(node);
  auto model = loader.getModel();
  if (!model)
  {
    emit("RobotModel is null");
    stop();
    return 1;
  }
  emit("STAGE: RobotModel loaded");

  emit("");
  emit(std::string("RobotModel: ") + model->getName());
  emit(std::string("Model frame: ") + model->getModelFrame());

  if (model->getName() != kExpectedModel)
  {
    emit("Wrong robot model. Expected fairino3_dual_robot.");
    stop();
    return 1;
  }
  if (model->getModelFrame() != kModelFrame)
  {
    emit("Wrong model frame. Expected world.");
    stop();
    return 1;
  }
  if (!model->hasJointModelGroup(kGroupA) || !model->hasJointModelGroup(kGroupB) ||
      !model->hasJointModelGroup("dual_arms"))
  {
    emit("Missing arm_a, arm_b or dual_arms group");
    stop();
    return 1;
  }
  if (!model->hasLinkModel(kTcpA) || !model->hasLinkModel(kTcpB))
  {
    emit("Missing TCP_A or TCP_B");
    stop();
    return 1;
  }

  TargetResult arm_a;
  arm_a.label = "Arm A Handover";
  arm_a.group = kGroupA;
  arm_a.tcp = kTcpA;
  arm_a.target = geo.world_tcp_a;

  TargetResult arm_b_pre;
  arm_b_pre.label = "Arm B PreHandover";
  arm_b_pre.group = kGroupB;
  arm_b_pre.tcp = kTcpB;
  arm_b_pre.target = geo.world_pre_b;

  TargetResult arm_b_handover;
  arm_b_handover.label = "Arm B Handover";
  arm_b_handover.group = kGroupB;
  arm_b_handover.tcp = kTcpB;
  arm_b_handover.target = geo.world_tcp_b;

  emit("STAGE: check IK solvers");
  bool solvers_ok = true;
  solvers_ok &= describeSolver(node, model, kGroupA, kTcpA, arm_a);
  solvers_ok &= describeSolver(node, model, kGroupB, kTcpB, arm_b_pre);
  arm_b_handover.solver_name = arm_b_pre.solver_name;
  arm_b_handover.solver_tip = arm_b_pre.solver_tip;
  arm_b_handover.solver_base = arm_b_pre.solver_base;
  if (!solvers_ok)
  {
    emit("IK solver missing or cannot solve for the required TCP.");
    emit("  Arm A: " + (arm_a.fail_reason.empty() ? arm_a.solver_name : arm_a.fail_reason));
    emit("  Arm B: " +
         (arm_b_pre.fail_reason.empty() ? arm_b_pre.solver_name : arm_b_pre.fail_reason));
    stop();
    return 1;
  }

  emit("STAGE: read RobotState from /joint_states");
  sensor_msgs::msg::JointState js;
  if (!waitForJointStates(node, joint_timeout, joint_max_age, js, error))
  {
    emit("Current state read failed: " + error);
    stop();
    return blocked("existing dual_bringup unavailable");
  }

  std::map<std::string, double> named;
  if (!extractNamedPositions(js, named, error) || !requireArmJoints(named, error))
  {
    emit("Current state read failed: " + error);
    stop();
    return 1;
  }

  moveit::core::RobotState current(model);
  current.setToDefaultValues();
  applyNamedJoints(current, named);
  current.update();
  emit("STAGE: RobotState read complete");

  emit("");
  emit("Current dual-arm joints from /joint_states (by name):");
  for (const auto& name : kArmAJoints)
  {
    std::ostringstream line;
    line.setf(std::ios::fixed);
    line << std::setprecision(6) << "  " << name << " = " << named[name] << " rad";
    emit(line.str());
  }
  for (const auto& name : kArmBJoints)
  {
    std::ostringstream line;
    line.setf(std::ios::fixed);
    line << std::setprecision(6) << "  " << name << " = " << named[name] << " rad";
    emit(line.str());
  }

  emit("STAGE: query GetPlanningScene (async, wait on existing executor)");
  auto scene_client =
      node->create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");
  if (scene_client->wait_for_service(5s))
  {
    auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
    request->components.components =
        moveit_msgs::msg::PlanningSceneComponents::ROBOT_STATE;
    // Do not call spin_until_future_complete(): the node already belongs to
    // the background MultiThreadedExecutor, which delivers this response.
    auto future = scene_client->async_send_request(request);
    if (future.wait_for(5s) == std::future_status::ready)
    {
      auto response = future.get();
      if (response)
      {
        const auto& scene_js = response->scene.robot_state.joint_state;
        const size_t count = std::min(scene_js.name.size(), scene_js.position.size());
        double max_delta = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
          auto it = named.find(scene_js.name[i]);
          if (it == named.end() || !std::isfinite(scene_js.position[i]))
            continue;
          max_delta = std::max(max_delta, std::abs(it->second - scene_js.position[i]));
        }
        std::ostringstream line;
        line.setf(std::ios::fixed);
        line << std::setprecision(6)
             << "GetPlanningScene vs /joint_states max |dq| = " << max_delta << " rad";
        emit(line.str());
      }
    }
  }

  std::vector<double> step14_c;
  std::string seed_error;
  const bool have_step14 = loadStep14ArmASeed(step14_yaml, step14_c, seed_error);
  if (!have_step14)
    emit("STEP14 C-face seed skipped: " + seed_error);
  else
    emit("STEP14 Arm A C-face joints used only as IK seed, not as current state.");

  std::vector<std::pair<std::string, std::vector<double>>> seeds_a;
  seeds_a.push_back({"current", jointsOf(current, kArmAJoints)});
  if (have_step14)
    seeds_a.push_back({"step14_c_face", step14_c});

  std::vector<std::pair<std::string, std::vector<double>>> seeds_b;
  seeds_b.push_back({"current", jointsOf(current, kArmBJoints)});

  emit("STAGE: IK solve");
  searchIk(arm_a, current, seeds_a, max_attempts, max_unique, timeout_exact, timeout_nearby,
           nearby_radius, nearby_radius_local, min_distance, pos_tol, ori_tol, rng_seed);
  searchIk(arm_b_pre, current, seeds_b, max_attempts, max_unique, timeout_exact, timeout_nearby,
           nearby_radius, nearby_radius_local, min_distance, pos_tol, ori_tol, rng_seed + 17);
  searchIk(arm_b_handover, current, seeds_b, max_attempts, max_unique, timeout_exact,
           timeout_nearby, nearby_radius, nearby_radius_local, min_distance, pos_tol, ori_tol,
           rng_seed + 31);
  emit("STAGE: IK solve complete");

  printTarget(arm_a);
  printTarget(arm_b_pre);
  printTarget(arm_b_handover);

  const bool all_pass = arm_a.pass && arm_b_pre.pass && arm_b_handover.pass;
  emit("");
  if (all_pass)
    emit("DUAL-4B-1 IK PASS");
  else
  {
    emit("DUAL-4B-1 IK FAIL");
    if (!arm_a.pass)
      emit("  failed target: Arm A Handover");
    if (!arm_b_pre.pass)
      emit("  failed target: Arm B PreHandover");
    if (!arm_b_handover.pass)
      emit("  failed target: Arm B Handover");
  }

  emit("");
  emit("Collision check: NOT PERFORMED");
  emit("Gripper interference check: NOT PERFORMED");
  emit("Grasp feasibility: NOT VERIFIED");
  emit("Real robot commands sent: ZERO");
  emit("Collision and grasp feasibility: NOT CHECKED");

  stop();
  return all_pass ? 0 : 2;
  }
  catch (const std::exception& e)
  {
    emit(std::string("STAGE exception: ") + e.what());
    emit("DUAL-4B-1 aborted with a caught exception. No robot commands were sent.");
    try
    {
      stop();
    }
    catch (const std::exception& stop_error)
    {
      emit(std::string("shutdown error: ") + stop_error.what());
      if (rclcpp::ok())
        rclcpp::shutdown();
    }
    return 1;
  }
  catch (...)
  {
    emit("STAGE exception: unknown non-std exception");
    emit("DUAL-4B-1 aborted with a caught exception. No robot commands were sent.");
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
