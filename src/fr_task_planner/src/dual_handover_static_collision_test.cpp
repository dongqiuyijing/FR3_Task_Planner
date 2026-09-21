#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <yaml-cpp/yaml.h>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/collision_detection/collision_matrix.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/joint_model.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
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
constexpr char kGripperGroupA[] = "arm_a_gripper";
constexpr char kGripperGroupB[] = "arm_b_gripper";
constexpr char kGripperJointA[] = "arm_a_gripper_joint";
constexpr char kGripperJointB[] = "arm_b_gripper_joint";

const std::vector<std::string> kArmAJoints = {
    "arm_a_j1", "arm_a_j2", "arm_a_j3", "arm_a_j4", "arm_a_j5", "arm_a_j6"};
const std::vector<std::string> kArmBJoints = {
    "arm_b_j1", "arm_b_j2", "arm_b_j3", "arm_b_j4", "arm_b_j5", "arm_b_j6"};

const std::vector<std::string> kArmATouchLinks = {
    "arm_a_gripper_base_link", "arm_a_rail_155", "arm_a_slider_l", "arm_a_slider_r",
    "arm_a_finger_l",          "arm_a_finger_r", "arm_a_gripper_gap_link"};
const std::vector<std::string> kArmBFingerLinks = {"arm_b_finger_l", "arm_b_finger_r"};
const std::vector<std::string> kArmAFingerLinks = {"arm_a_finger_l", "arm_a_finger_r"};
const std::vector<std::string> kArmAGripperLinks = {
    "arm_a_gripper_base_link", "arm_a_rail_155", "arm_a_slider_l", "arm_a_slider_r",
    "arm_a_finger_l",          "arm_a_finger_r", "arm_a_gripper_gap_link", "arm_a_gripper_tcp"};
const std::vector<std::string> kArmBGripperLinks = {
    "arm_b_gripper_base_link", "arm_b_rail_155", "arm_b_slider_l", "arm_b_slider_r",
    "arm_b_finger_l",          "arm_b_finger_r", "arm_b_gripper_gap_link", "arm_b_gripper_tcp"};

struct PoseXYZW
{
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
};

struct HandoverGeometry
{
  std::string object_name = "small_part";
  double radius = 0.0075;
  double length = 0.035;
  Eigen::Isometry3d world_object = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d world_tcp_a = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d world_tcp_b = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d world_pre_b = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d tcp_a_object = Eigen::Isometry3d::Identity();
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

struct SceneSnapshot
{
  std::vector<std::string> world_ids;
  std::vector<std::string> attached_ids;
  std::vector<std::string> attached_links;
  std::vector<std::string> world_pose_lines;
  std::vector<std::string> attached_pose_lines;
  size_t acm_entries = 0;
  bool has_octomap = false;
  bool has_table = false;
  bool has_column = false;
  bool acm_a_base_column = false;
  bool acm_b_base_column = false;
  std::string acm_a_base_column_status = "UNKNOWN";
  std::string acm_b_base_column_status = "UNKNOWN";
};

struct CollisionDiag
{
  bool collision = false;
  bool truncated = false;
  int contact_count = 0;
  std::vector<std::string> pairs;
  std::vector<std::string> gripper_interference;
  std::vector<std::string> expected_b_finger;
  std::vector<std::string> expected_a_touch;
  std::vector<std::string> object_illegal;
  std::vector<std::string> robot_env;
  std::vector<std::string> cross_arm;
  std::vector<std::string> arm_a_self;
  std::vector<std::string> arm_b_self;
  std::vector<std::string> finger_self;
  std::vector<std::string> arm_table;
  std::vector<std::string> arm_column;
  std::vector<std::string> upperarm_column;
  bool has_b_finger_part_contact = false;
  bool has_a_finger_part_contact = false;
  bool has_upperarm_column = false;
  bool has_finger_self = false;
  double max_b_finger_depth = 0.0;
  int b_finger_contact_points = 0;
  int a_finger_contact_points = 0;
};

struct ComboResult
{
  size_t a_index = 0;
  size_t b_index = 0;
  std::string state_name;
  std::vector<double> joints_a;
  std::vector<double> joints_b;
  double q_a = 0.0;
  double q_b = 0.0;
  double slider_la = 0.0;
  double slider_ra = 0.0;
  double slider_lb = 0.0;
  double slider_rb = 0.0;
  bool joints_ok = false;
  bool fk_ok = false;
  bool object_pose_ok = false;
  double a_fk_position_error = 0.0;
  double a_fk_orientation_error_deg = 0.0;
  double b_fk_position_error = 0.0;
  double b_fk_orientation_error_deg = 0.0;
  double object_position_error = 0.0;
  double object_orientation_error_deg = 0.0;
  std::string fail_reason;
  CollisionDiag strict;
  CollisionDiag expected;
  bool strict_free = false;
  bool expected_free = false;
};

struct MimicJointInfo
{
  std::string name;
  double factor = 0.0;
  double offset = 0.0;
};

struct GripperModelInfo
{
  std::string active_joint;
  std::string finger_l;
  std::string finger_r;
  std::string slider_l;
  std::string slider_r;
  std::vector<MimicJointInfo> mimics;
  bool present = false;
  double min_q = 0.0;
  double max_q = 0.1;
};

struct OpeningSample
{
  double q = 0.0;
  double slider_l = 0.0;
  double slider_r = 0.0;
  double finger_origin_sep = 0.0;
  bool in_limits = false;
  bool mimic_ok = false;
  bool finger_self_collision = false;
  bool truncated = false;
  std::string mimic_error;
};

struct OpeningChoice
{
  bool found = false;
  bool physical_incomplete = true;
  double q = 0.0;
  double slider_l = 0.0;
  double slider_r = 0.0;
  double finger_origin_sep = 0.0;
  bool both_fingers_contact_object = false;
  bool object_in_grasp_region = false;
  std::string note;
  std::vector<OpeningSample> samples;
};

struct GripperStateInfo
{
  std::string source = "NOMINAL GRIPPER STATE";
  bool actual_a = false;
  bool actual_b = false;
  bool model_has_a = false;
  bool model_has_b = false;
  double actual_a_value = 0.0;
  double actual_b_value = 0.0;
  double q_a_grasp = 0.0;
  double q_b_open = 0.0;
  double q_b_receive = 0.0;
  bool finger_collision_geometry = true;
  bool opening_found = false;
  std::string geometry_note;
};

struct LiveAcmCheck
{
  bool msg_has_entries = false;
  bool complete = false;
  bool a_base_column = false;
  bool b_base_column = false;
  std::string a_status = "UNKNOWN";
  std::string b_status = "UNKNOWN";
  std::vector<std::string> missing;
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

geometry_msgs::msg::Pose toPoseMsg(const Eigen::Isometry3d& T)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = T.translation().x();
  pose.position.y = T.translation().y();
  pose.position.z = T.translation().z();
  const Eigen::Quaterniond q(T.rotation());
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();
  return pose;
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
    error = "DUAL-4B-2 must remain static-check-only; execution.enabled is true";
    return false;
  }

  geo.object_name = cfg["object"]["name"].as<std::string>();
  const Eigen::Vector3d object_center(cfg["object"]["center_world"]["x"].as<double>(),
                                      cfg["object"]["center_world"]["y"].as<double>(),
                                      cfg["object"]["center_world"]["z"].as<double>());
  geo.radius = cfg["object"]["radius_m"].as<double>();
  geo.length = cfg["object"]["length_m"].as<double>();
  geo.arm_b_offset_x = cfg["handover"]["arm_b_grasp_offset_x_m"].as<double>();
  geo.prehandover_distance = cfg["handover"]["prehandover_b_distance_m"].as<double>();

  std::ostringstream diff;
  if (!nearlyEqual(object_center.x(), 0.0, 1e-12) || !nearlyEqual(object_center.y(), 0.4, 1e-12) ||
      !nearlyEqual(object_center.z(), 0.9, 1e-12))
  {
    geo.yaml_matches_dual4a_doc = false;
    diff << "object.center_world differs from DUAL-4A (0, 0.4, 0.9); ";
  }
  if (!nearlyEqual(geo.radius, 0.0075, 1e-12) || !nearlyEqual(geo.length, 0.035, 1e-12))
  {
    geo.yaml_matches_dual4a_doc = false;
    diff << "object size differs from DUAL-4A r=0.0075 l=0.035; ";
  }
  if (!nearlyEqual(geo.arm_b_offset_x, -0.008, 1e-12))
  {
    geo.yaml_matches_dual4a_doc = false;
    diff << "arm_b_grasp_offset_x_m=" << geo.arm_b_offset_x << " (DUAL-4A used -0.008); ";
  }
  if (!nearlyEqual(geo.prehandover_distance, 0.20, 1e-12))
  {
    geo.yaml_matches_dual4a_doc = false;
    diff << "prehandover_b_distance_m=" << geo.prehandover_distance << " (DUAL-4A used 0.20); ";
  }
  geo.yaml_diff = diff.str();

  if (!(geo.arm_b_offset_x > -geo.length / 2.0 && geo.arm_b_offset_x < 0.0))
  {
    error = "Arm B offset must be inside the negative-X half of the object";
    return false;
  }
  if (geo.prehandover_distance <= 0.0)
  {
    error = "PreHandover distance must be positive";
    return false;
  }

  Eigen::Matrix3d R_world_object;
  R_world_object << 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  if (!checkRotation(R_world_object, "R_world_object", error))
    return false;
  geo.world_object = makeTransform(R_world_object, object_center);

  Eigen::Matrix3d R_tcpA_object;
  R_tcpA_object << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0;
  geo.tcp_a_object = makeTransform(R_tcpA_object, Eigen::Vector3d::Zero());
  geo.world_tcp_a = geo.world_object * geo.tcp_a_object.inverse();

  Eigen::Matrix3d R_world_tcpB;
  R_world_tcpB << 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0;
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
    {
      return false;
    }
    setStringParam(node, parameter.get_name(), parameter.as_string());
  }

  const std::vector<std::string> groups = {kGroupA, kGroupB};
  const std::vector<std::string> keys = {"kinematics_solver", "kinematics_solver_search_resolution",
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
    emit(std::string("STAGE overlay: /move_group kinematics params skipped: ") + e.what());
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
  sensor_msgs::msg::JointState::SharedPtr latest;
  auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states",
      rclcpp::QoS(rclcpp::KeepLast(10)).reliable().transient_local(),
      [&latest](const sensor_msgs::msg::JointState::SharedPtr msg) { latest = msg; });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec);
  while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline)
  {
    if (latest)
    {
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

bool describeSolver(const rclcpp::Node::SharedPtr& node,
                    const moveit::core::RobotModelConstPtr& model, const std::string& group_name,
                    const std::string& tcp_name, TargetResult& result)
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
  cand.fk_ok =
      cand.fk_position_error <= pos_tol && cand.fk_orientation_error_deg <= ori_tol_deg;
  cand.seed_l2 = jointL2(cand.joints, jointsOf(seed_state, joint_names));
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
    if (result.attempts >= max_attempts || static_cast<int>(result.unique.size()) >= max_unique)
      break;

    moveit::core::RobotState exact(current);
    applyGroupJoints(exact, joint_names, seed.second);
    consider(exact, seed.first + "_exact", timeout_exact);

    const int nearby_n = (seed.first == "current") ? 8 : 4;
    for (int i = 0; i < nearby_n; ++i)
    {
      if (result.attempts >= max_attempts || static_cast<int>(result.unique.size()) >= max_unique)
        break;
      moveit::core::RobotState nearby(current);
      applyGroupJoints(nearby, joint_names, seed.second);
      const double radius = (i < nearby_n / 2) ? nearby_radius_local : nearby_radius;
      perturbGroup(nearby, group, radius, rng);
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
  if (!result.pass)
    result.fail_reason = "no unique in-bounds FK-valid IK";
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
  }
}

int blocked(const std::string& why)
{
  emit("BUILD PASS");
  emit("RUNTIME BLOCKED: existing dual_bringup unavailable");
  emit("reason: " + why);
  emit("Collision check: NOT PERFORMED");
  emit("Gripper interference check: NOT PERFORMED");
  emit("Grasp feasibility: NOT VERIFIED");
  emit("Global PlanningScene modified: NO");
  emit("Real robot commands sent: ZERO");
  return 1;
}

bool inList(const std::vector<std::string>& names, const std::string& value)
{
  return std::find(names.begin(), names.end(), value) != names.end();
}

bool startsWith(const std::string& value, const std::string& prefix)
{
  return value.rfind(prefix, 0) == 0;
}

bool isEnvObject(const std::string& name)
{
  if (name == kTableName || name == kColumnName)
    return true;
  return name.find("table") != std::string::npos || name.find("column") != std::string::npos;
}

bool linkHasCollisionGeometry(const moveit::core::RobotModelConstPtr& model,
                              const std::string& link_name)
{
  if (!model->hasLinkModel(link_name))
    return false;
  return !model->getLinkModel(link_name)->getShapes().empty();
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
  if (a <= b)
    return a + " <-> " + b;
  return b + " <-> " + a;
}

std::string poseLine(const geometry_msgs::msg::Pose& pose)
{
  return "xyz=" + fmtVec({pose.position.x, pose.position.y, pose.position.z}) +
         " xyzw=" + fmtVec({pose.orientation.x, pose.orientation.y, pose.orientation.z,
                            pose.orientation.w});
}

std::string acmEntryStatus(const collision_detection::AllowedCollisionMatrix& acm,
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

bool acmAlwaysAllowed(const collision_detection::AllowedCollisionMatrix& acm, const std::string& a,
                      const std::string& b)
{
  collision_detection::AllowedCollision::Type type;
  return acm.getEntry(a, b, type) && type == collision_detection::AllowedCollision::ALWAYS;
}

std::string msgAcmStatus(const moveit_msgs::msg::PlanningScene& msg, const std::string& a,
                         const std::string& b)
{
  const auto& names = msg.allowed_collision_matrix.entry_names;
  if (names.empty())
    return "EMPTY MESSAGE ACM";
  auto find_index = [&](const std::string& name) -> int {
    for (size_t i = 0; i < names.size(); ++i)
    {
      if (names[i] == name)
        return static_cast<int>(i);
    }
    return -1;
  };
  const int ia = find_index(a);
  const int ib = find_index(b);
  if (ia < 0 || ib < 0)
    return "ABSENT from message ACM";
  const auto& values = msg.allowed_collision_matrix.entry_values;
  if (ia >= static_cast<int>(values.size()) ||
      ib >= static_cast<int>(values[static_cast<size_t>(ia)].enabled.size()))
  {
    return "INDEX OUT OF RANGE";
  }
  return values[static_cast<size_t>(ia)].enabled[static_cast<size_t>(ib)] ? "ALWAYS allowed" :
                                                                           "checked / not allowed";
}

LiveAcmCheck inspectLiveAcm(const moveit_msgs::msg::PlanningScene& msg,
                            const collision_detection::AllowedCollisionMatrix& scene_acm)
{
  LiveAcmCheck check;
  check.msg_has_entries = !msg.allowed_collision_matrix.entry_names.empty();
  check.a_status = acmEntryStatus(scene_acm, "arm_a_base_link", kColumnName);
  check.b_status = acmEntryStatus(scene_acm, "arm_b_base_link", kColumnName);
  check.a_base_column = acmAlwaysAllowed(scene_acm, "arm_a_base_link", kColumnName);
  check.b_base_column = acmAlwaysAllowed(scene_acm, "arm_b_base_link", kColumnName);
  if (!check.msg_has_entries)
    check.missing.push_back("GetPlanningScene ALLOWED_COLLISION_MATRIX is empty");
  if (!check.a_base_column)
    check.missing.push_back("arm_a_base_link <-> mounting_column");
  if (!check.b_base_column)
    check.missing.push_back("arm_b_base_link <-> mounting_column");
  check.complete = check.msg_has_entries && check.a_base_column && check.b_base_column;
  return check;
}

GripperModelInfo inspectGripperModel(const moveit::core::RobotModelConstPtr& model,
                                     const std::string& active_joint, const std::string& finger_l,
                                     const std::string& finger_r)
{
  GripperModelInfo info;
  info.active_joint = active_joint;
  info.finger_l = finger_l;
  info.finger_r = finger_r;
  info.present = model->hasJointModel(active_joint);
  if (!info.present)
    return info;
  const auto* jm = model->getJointModel(active_joint);
  const auto& bounds = jm->getVariableBounds();
  if (!bounds.empty() && bounds.front().position_bounded_)
  {
    info.min_q = bounds.front().min_position_;
    info.max_q = bounds.front().max_position_;
  }
  for (const auto* mimic : jm->getMimicRequests())
  {
    MimicJointInfo item;
    item.name = mimic->getName();
    item.factor = mimic->getMimicFactor();
    item.offset = mimic->getMimicOffset();
    info.mimics.push_back(item);
    if (item.name.find("slider_l") != std::string::npos)
      info.slider_l = item.name;
    if (item.name.find("slider_r") != std::string::npos)
      info.slider_r = item.name;
  }
  return info;
}

bool setActiveGripperJoint(moveit::core::RobotState& state, const GripperModelInfo& gripper,
                           double q, std::string& error)
{
  if (!gripper.present)
  {
    error = "missing active gripper joint " + gripper.active_joint;
    return false;
  }
  if (q < gripper.min_q - 1e-12 || q > gripper.max_q + 1e-12)
  {
    error = gripper.active_joint + " q=" + fmtScalar(q) + " outside limits [" +
            fmtScalar(gripper.min_q) + ", " + fmtScalar(gripper.max_q) + "]";
    return false;
  }
  const auto* jm = state.getRobotModel()->getJointModel(gripper.active_joint);
  state.setJointPositions(jm, &q);
  state.update();
  const double actual = state.getVariablePosition(gripper.active_joint);
  if (std::abs(actual - q) > 1e-9)
  {
    error = gripper.active_joint + " did not accept requested q";
    return false;
  }
  for (const auto& mimic : gripper.mimics)
  {
    const double expected = mimic.factor * q + mimic.offset;
    const double got = state.getVariablePosition(mimic.name);
    if (std::abs(got - expected) > 1e-9)
    {
      error = "mimic mismatch " + mimic.name + " expected " + fmtScalar(expected) + " got " +
              fmtScalar(got);
      return false;
    }
  }
  return true;
}

double jointOrNan(const moveit::core::RobotState& state, const std::string& name)
{
  if (name.empty() || !state.getRobotModel()->hasJointModel(name))
    return std::numeric_limits<double>::quiet_NaN();
  return state.getVariablePosition(name);
}

double fingerOriginSeparation(const moveit::core::RobotState& state, const GripperModelInfo& gripper)
{
  if (!state.getRobotModel()->hasLinkModel(gripper.finger_l) ||
      !state.getRobotModel()->hasLinkModel(gripper.finger_r))
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return (state.getGlobalLinkTransform(gripper.finger_r).translation() -
          state.getGlobalLinkTransform(gripper.finger_l).translation())
      .norm();
}

void printGripperMimic(const moveit::core::RobotState& state, const GripperModelInfo& a,
                       const GripperModelInfo& b, const std::string& tag)
{
  emit(tag + " (NOMINAL GRIPPER STATE, not measured hardware opening):");
  emit("  " + a.active_joint + ": " + fmtScalar(jointOrNan(state, a.active_joint)));
  emit("  " + (a.slider_l.empty() ? std::string("arm_a slider_l MISSING") : a.slider_l) + ": " +
       fmtScalar(jointOrNan(state, a.slider_l)));
  emit("  " + (a.slider_r.empty() ? std::string("arm_a slider_r MISSING") : a.slider_r) + ": " +
       fmtScalar(jointOrNan(state, a.slider_r)));
  emit("  " + b.active_joint + ": " + fmtScalar(jointOrNan(state, b.active_joint)));
  emit("  " + (b.slider_l.empty() ? std::string("arm_b slider_l MISSING") : b.slider_l) + ": " +
       fmtScalar(jointOrNan(state, b.slider_l)));
  emit("  " + (b.slider_r.empty() ? std::string("arm_b slider_r MISSING") : b.slider_r) + ": " +
       fmtScalar(jointOrNan(state, b.slider_r)));
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
    error = "/get_planning_scene empty response";
    return false;
  }
  out = response->scene;
  return true;
}

SceneSnapshot snapshotFromMsg(const moveit_msgs::msg::PlanningScene& msg)
{
  SceneSnapshot snap;
  for (const auto& object : msg.world.collision_objects)
  {
    snap.world_ids.push_back(object.id);
    geometry_msgs::msg::Pose pose = object.pose;
    if (!object.primitive_poses.empty())
      pose = object.primitive_poses.front();
    snap.world_pose_lines.push_back(object.id + " " + poseLine(pose));
  }
  for (const auto& attached : msg.robot_state.attached_collision_objects)
  {
    snap.attached_ids.push_back(attached.object.id);
    snap.attached_links.push_back(attached.link_name);
    geometry_msgs::msg::Pose pose = attached.object.pose;
    if (!attached.object.primitive_poses.empty())
      pose = attached.object.primitive_poses.front();
    snap.attached_pose_lines.push_back(attached.object.id + "@" + attached.link_name + " " +
                                       poseLine(pose));
  }
  snap.acm_entries = msg.allowed_collision_matrix.entry_names.size();
  snap.has_octomap = !msg.world.octomap.octomap.data.empty();
  snap.has_table = inList(snap.world_ids, kTableName);
  snap.has_column = inList(snap.world_ids, kColumnName);
  snap.acm_a_base_column_status = msgAcmStatus(msg, "arm_a_base_link", kColumnName);
  snap.acm_b_base_column_status = msgAcmStatus(msg, "arm_b_base_link", kColumnName);
  snap.acm_a_base_column = snap.acm_a_base_column_status == "ALWAYS allowed";
  snap.acm_b_base_column = snap.acm_b_base_column_status == "ALWAYS allowed";
  std::sort(snap.world_ids.begin(), snap.world_ids.end());
  std::sort(snap.attached_ids.begin(), snap.attached_ids.end());
  std::sort(snap.world_pose_lines.begin(), snap.world_pose_lines.end());
  std::sort(snap.attached_pose_lines.begin(), snap.attached_pose_lines.end());
  return snap;
}

void printSnapshotDiff(const SceneSnapshot& before, const SceneSnapshot& after)
{
  auto diff = [](const std::vector<std::string>& a, const std::vector<std::string>& b,
                 const std::string& label) {
    std::vector<std::string> added, removed;
    std::set_difference(b.begin(), b.end(), a.begin(), a.end(), std::back_inserter(added));
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(removed));
    emit("  added " + label + ": " + joinNames(added));
    emit("  removed " + label + ": " + joinNames(removed));
  };
  diff(before.world_ids, after.world_ids, "world IDs");
  diff(before.attached_ids, after.attached_ids, "attached IDs");
  diff(before.world_pose_lines, after.world_pose_lines, "world poses");
  diff(before.attached_pose_lines, after.attached_pose_lines, "attached poses");
  if (before.acm_a_base_column_status != after.acm_a_base_column_status)
  {
    emit("  ACM arm_a_base_link <-> mounting_column: " + before.acm_a_base_column_status +
         " -> " + after.acm_a_base_column_status);
  }
  if (before.acm_b_base_column_status != after.acm_b_base_column_status)
  {
    emit("  ACM arm_b_base_link <-> mounting_column: " + before.acm_b_base_column_status +
         " -> " + after.acm_b_base_column_status);
  }
  if (before.acm_entries != after.acm_entries)
  {
    emit("  ACM entry_names: " + std::to_string(before.acm_entries) + " -> " +
         std::to_string(after.acm_entries));
  }
}

bool sceneUnchanged(const SceneSnapshot& a, const SceneSnapshot& b)
{
  return a.world_ids == b.world_ids && a.attached_ids == b.attached_ids &&
         a.attached_links == b.attached_links && a.world_pose_lines == b.world_pose_lines &&
         a.attached_pose_lines == b.attached_pose_lines &&
         a.acm_a_base_column_status == b.acm_a_base_column_status &&
         a.acm_b_base_column_status == b.acm_b_base_column_status && a.acm_entries == b.acm_entries;
}

void classifyPair(const std::string& object_id, const std::string& a, const std::string& b,
                  CollisionDiag& diag, double depth)
{
  const std::string key = pairKey(a, b);
  const bool a_is_obj = (a == object_id);
  const bool b_is_obj = (b == object_id);
  const std::string other = a_is_obj ? b : a;

  if (a_is_obj || b_is_obj)
  {
    if (inList(kArmATouchLinks, other))
    {
      diag.expected_a_touch.push_back(key);
      if (inList(kArmAFingerLinks, other))
      {
        diag.has_a_finger_part_contact = true;
        ++diag.a_finger_contact_points;
      }
      return;
    }
    if (inList(kArmBFingerLinks, other))
    {
      diag.expected_b_finger.push_back(key);
      diag.has_b_finger_part_contact = true;
      diag.max_b_finger_depth = std::max(diag.max_b_finger_depth, depth);
      ++diag.b_finger_contact_points;
      return;
    }
    if (isEnvObject(other))
    {
      diag.robot_env.push_back(key);
      diag.object_illegal.push_back(key);
      return;
    }
    diag.object_illegal.push_back(key);
    return;
  }

  const bool finger_self_a = (a == "arm_a_finger_l" && b == "arm_a_finger_r") ||
                             (a == "arm_a_finger_r" && b == "arm_a_finger_l");
  const bool finger_self_b = (a == "arm_b_finger_l" && b == "arm_b_finger_r") ||
                             (a == "arm_b_finger_r" && b == "arm_b_finger_l");
  if (finger_self_a || finger_self_b)
  {
    diag.finger_self.push_back(key);
    diag.has_finger_self = true;
    if (finger_self_a)
      diag.arm_a_self.push_back(key);
    else
      diag.arm_b_self.push_back(key);
    return;
  }

  const bool a_grip_a = inList(kArmAGripperLinks, a);
  const bool b_grip_a = inList(kArmAGripperLinks, b);
  const bool a_grip_b = inList(kArmBGripperLinks, a);
  const bool b_grip_b = inList(kArmBGripperLinks, b);
  if ((a_grip_a && b_grip_b) || (a_grip_b && b_grip_a))
  {
    diag.gripper_interference.push_back(key);
    diag.cross_arm.push_back(key);
    return;
  }

  const bool a_arm_a = startsWith(a, "arm_a_");
  const bool b_arm_a = startsWith(b, "arm_a_");
  const bool a_arm_b = startsWith(a, "arm_b_");
  const bool b_arm_b = startsWith(b, "arm_b_");
  if ((a_arm_a && b_arm_b) || (a_arm_b && b_arm_a))
  {
    diag.cross_arm.push_back(key);
    return;
  }

  const bool upperarm_col =
      (key == pairKey("arm_a_upperarm_link", kColumnName) ||
       key == pairKey("arm_b_upperarm_link", kColumnName));
  if (upperarm_col)
  {
    diag.upperarm_column.push_back(key);
    diag.has_upperarm_column = true;
    diag.arm_column.push_back(key);
    diag.robot_env.push_back(key);
    return;
  }

  if (a == kTableName || b == kTableName || a.find("table") != std::string::npos ||
      b.find("table") != std::string::npos)
  {
    diag.arm_table.push_back(key);
    diag.robot_env.push_back(key);
    return;
  }
  if (a == kColumnName || b == kColumnName || a.find("column") != std::string::npos ||
      b.find("column") != std::string::npos)
  {
    diag.arm_column.push_back(key);
    diag.robot_env.push_back(key);
    return;
  }

  if (a_arm_a && b_arm_a)
    diag.arm_a_self.push_back(key);
  else if (a_arm_b && b_arm_b)
    diag.arm_b_self.push_back(key);
}

CollisionDiag fillDiag(const collision_detection::CollisionResult& result, int max_contacts,
                       const std::string& object_id)
{
  CollisionDiag diag;
  diag.collision = result.collision;
  diag.contact_count = static_cast<int>(result.contact_count);
  diag.truncated = diag.contact_count >= max_contacts;
  std::set<std::string> unique_pairs;
  for (const auto& item : result.contacts)
  {
    const std::string key = pairKey(item.first.first, item.first.second);
    unique_pairs.insert(key);
    double depth = 0.0;
    for (const auto& contact : item.second)
      depth = std::max(depth, contact.depth);
    classifyPair(object_id, item.first.first, item.first.second, diag, depth);
  }
  diag.pairs.assign(unique_pairs.begin(), unique_pairs.end());
  auto uniq = [](std::vector<std::string>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
  };
  uniq(diag.gripper_interference);
  uniq(diag.expected_b_finger);
  uniq(diag.expected_a_touch);
  uniq(diag.object_illegal);
  uniq(diag.robot_env);
  uniq(diag.cross_arm);
  uniq(diag.arm_a_self);
  uniq(diag.arm_b_self);
  uniq(diag.finger_self);
  uniq(diag.arm_table);
  uniq(diag.arm_column);
  uniq(diag.upperarm_column);
  return diag;
}

CollisionDiag checkState(planning_scene::PlanningScene& scene, moveit::core::RobotState& state,
                         const collision_detection::AllowedCollisionMatrix& acm, int max_contacts,
                         int max_per_pair, const std::string& object_id)
{
  collision_detection::CollisionRequest request;
  collision_detection::CollisionResult result;
  request.contacts = true;
  request.max_contacts = static_cast<std::size_t>(max_contacts);
  request.max_contacts_per_pair = static_cast<std::size_t>(max_per_pair);
  request.group_name.clear();
  scene.checkCollision(request, result, state, acm);
  return fillDiag(result, max_contacts, object_id);
}

bool allVariablesFinite(const moveit::core::RobotState& state, std::string& error)
{
  const double* values = state.getVariablePositions();
  const auto& names = state.getVariableNames();
  for (size_t i = 0; i < names.size(); ++i)
  {
    if (!std::isfinite(values[i]))
    {
      error = "non-finite joint variable: " + names[i];
      return false;
    }
  }
  return true;
}

OpeningSample sampleOpening(planning_scene::PlanningScene& scene, moveit::core::RobotState& state,
                            const GripperModelInfo& gripper, double q, int max_contacts,
                            int max_per_pair)
{
  OpeningSample sample;
  sample.q = q;
  std::string error;
  sample.in_limits = (q + 1e-12 >= gripper.min_q && q - 1e-12 <= gripper.max_q);
  if (!setActiveGripperJoint(state, gripper, q, error))
  {
    sample.mimic_ok = false;
    sample.mimic_error = error;
    return sample;
  }
  sample.mimic_ok = true;
  sample.slider_l = jointOrNan(state, gripper.slider_l);
  sample.slider_r = jointOrNan(state, gripper.slider_r);
  sample.finger_origin_sep = fingerOriginSeparation(state, gripper);
  const auto diag =
      checkState(scene, state, scene.getAllowedCollisionMatrix(), max_contacts, max_per_pair, "");
  sample.finger_self_collision = diag.has_finger_self;
  sample.truncated = diag.truncated;
  return sample;
}

OpeningChoice searchNominalOpening(planning_scene::PlanningScene& scene,
                                   const moveit::core::RobotState& seed,
                                   const GripperModelInfo& gripper, double seed_q, double q_min,
                                   double q_max, double q_step, double object_diameter,
                                   int max_contacts, int max_per_pair)
{
  OpeningChoice choice;
  choice.physical_incomplete = true;
  moveit::core::RobotState state(seed);
  auto consider = [&](double q) {
    const auto sample =
        sampleOpening(scene, state, gripper, q, max_contacts, max_per_pair);
    choice.samples.push_back(sample);
    return sample;
  };

  consider(0.0);
  consider(gripper.max_q);
  consider(seed_q);

  std::vector<double> qs;
  for (double q = q_min; q <= q_max + 1e-12; q += q_step)
    qs.push_back(q);
  for (double q : qs)
  {
    bool already = false;
    for (const auto& sample : choice.samples)
    {
      if (std::abs(sample.q - q) < 1e-12)
      {
        already = true;
        break;
      }
    }
    if (!already)
      consider(q);
  }

  auto usable = [](const OpeningSample& sample) {
    return sample.in_limits && sample.mimic_ok && !sample.finger_self_collision &&
           !sample.truncated;
  };

  const OpeningSample* best = nullptr;
  double best_score = 1e9;
  for (const auto& sample : choice.samples)
  {
    if (!usable(sample))
      continue;
    const double seed_error = std::abs(sample.q - seed_q);
    if (std::isfinite(sample.finger_origin_sep) &&
        sample.finger_origin_sep + 1e-9 < object_diameter)
    {
      continue;
    }
    if (seed_error < best_score)
    {
      best_score = seed_error;
      best = &sample;
    }
  }
  if (!best)
    return choice;

  choice.found = true;
  choice.q = best->q;
  choice.slider_l = best->slider_l;
  choice.slider_r = best->slider_r;
  choice.finger_origin_sep = best->finger_origin_sep;
  choice.note =
      "NOMINAL OPENING FOUND from model search. Finger origin separation is not a certified "
      "inner-surface measurement. PHYSICAL GRASP FEASIBILITY: INCOMPLETE";
  return choice;
}

bool attachLocalObject(planning_scene::PlanningScene& scene, const HandoverGeometry& geo,
                       std::string& error)
{
  if (scene.getCurrentState().hasAttachedBody(geo.object_name))
  {
    moveit_msgs::msg::AttachedCollisionObject detach;
    detach.object.id = geo.object_name;
    detach.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    scene.processAttachedCollisionObjectMsg(detach);
  }
  if (scene.getWorld()->hasObject(geo.object_name))
  {
    moveit_msgs::msg::CollisionObject remove;
    remove.id = geo.object_name;
    remove.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    scene.processCollisionObjectMsg(remove);
  }

  moveit_msgs::msg::AttachedCollisionObject attached;
  attached.link_name = kTcpA;
  attached.touch_links = kArmATouchLinks;
  attached.object.id = geo.object_name;
  attached.object.header.frame_id = kTcpA;
  attached.object.operation = moveit_msgs::msg::CollisionObject::ADD;
  attached.object.pose.orientation.w = 1.0;
  attached.object.primitives.resize(1);
  attached.object.primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
  attached.object.primitives[0].dimensions.resize(2);
  attached.object.primitives[0].dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT] =
      geo.length;
  attached.object.primitives[0].dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS] =
      geo.radius;
  attached.object.primitive_poses.push_back(toPoseMsg(geo.tcp_a_object));
  scene.processAttachedCollisionObjectMsg(attached);

  if (!scene.getCurrentState().hasAttachedBody(geo.object_name))
  {
    error = "local attached object missing after attach";
    return false;
  }
  if (scene.getWorld()->hasObject(geo.object_name))
  {
    error = "object still present in local world after attach";
    return false;
  }
  const auto* body = scene.getCurrentState().getAttachedBody(geo.object_name);
  if (!body || body->getAttachedLinkName() != kTcpA)
  {
    error = "object not attached to arm_a_gripper_tcp";
    return false;
  }
  return true;
}

bool verifyObjectPose(const moveit::core::RobotState& state, const HandoverGeometry& geo,
                      double pos_tol, double ori_tol_deg, double& pos_err, double& ori_err,
                      std::string& error)
{
  if (!state.hasAttachedBody(geo.object_name))
  {
    error = "attached body missing during pose check";
    return false;
  }
  const auto* body = state.getAttachedBody(geo.object_name);
  if (!body || body->getShapePosesInLinkFrame().empty() ||
      body->getGlobalCollisionBodyTransforms().empty())
  {
    error = "attached body has no collision transform";
    return false;
  }

  double rel_pos = 0.0;
  double rel_ori = 0.0;
  poseError(geo.tcp_a_object, body->getShapePosesInLinkFrame().front(), rel_pos, rel_ori);
  if (rel_pos > pos_tol || rel_ori > ori_tol_deg)
  {
    error = "T_tcpA_object jumped relative to attach link";
    pos_err = rel_pos;
    ori_err = rel_ori;
    return false;
  }

  poseError(geo.world_object, body->getGlobalCollisionBodyTransforms().front(), pos_err, ori_err);
  if (pos_err > pos_tol || ori_err > ori_tol_deg)
  {
    error = "attached object world pose inconsistent with DUAL-4A target";
    return false;
  }
  return true;
}

ComboResult evaluateCombo(planning_scene::PlanningScene& scene, const HandoverGeometry& geo,
                          const GripperModelInfo& gripper_a, const GripperModelInfo& gripper_b,
                          const IkCandidate& a_cand, const IkCandidate& b_cand, size_t a_index,
                          size_t b_index, const Eigen::Isometry3d& b_target, double q_a, double q_b,
                          double pos_tol, double ori_tol_deg, int max_contacts, int max_per_pair,
                          bool allow_b_finger_expected)
{
  ComboResult out;
  out.a_index = a_index;
  out.b_index = b_index;
  out.joints_a = a_cand.joints;
  out.joints_b = b_cand.joints;
  out.q_a = q_a;
  out.q_b = q_b;

  moveit::core::RobotState combined(scene.getCurrentState());
  applyGroupJoints(combined, kArmAJoints, a_cand.joints);
  applyGroupJoints(combined, kArmBJoints, b_cand.joints);
  if (!setActiveGripperJoint(combined, gripper_a, q_a, out.fail_reason) ||
      !setActiveGripperJoint(combined, gripper_b, q_b, out.fail_reason))
  {
    return out;
  }
  out.slider_la = jointOrNan(combined, gripper_a.slider_l);
  out.slider_ra = jointOrNan(combined, gripper_a.slider_r);
  out.slider_lb = jointOrNan(combined, gripper_b.slider_l);
  out.slider_rb = jointOrNan(combined, gripper_b.slider_r);

  if (!allVariablesFinite(combined, out.fail_reason))
    return out;

  const auto* group_a = combined.getJointModelGroup(kGroupA);
  const auto* group_b = combined.getJointModelGroup(kGroupB);
  if (!combined.satisfiesBounds(group_a) || !combined.satisfiesBounds(group_b))
  {
    out.fail_reason = "combined state outside joint bounds";
    return out;
  }
  out.joints_ok = true;

  poseError(geo.world_tcp_a, combined.getGlobalLinkTransform(kTcpA), out.a_fk_position_error,
            out.a_fk_orientation_error_deg);
  poseError(b_target, combined.getGlobalLinkTransform(kTcpB), out.b_fk_position_error,
            out.b_fk_orientation_error_deg);
  if (out.a_fk_position_error > pos_tol || out.a_fk_orientation_error_deg > ori_tol_deg ||
      out.b_fk_position_error > pos_tol || out.b_fk_orientation_error_deg > ori_tol_deg)
  {
    out.fail_reason = "combined FK failed";
    return out;
  }
  out.fk_ok = true;

  if (!verifyObjectPose(combined, geo, pos_tol, ori_tol_deg, out.object_position_error,
                        out.object_orientation_error_deg, out.fail_reason))
  {
    return out;
  }
  out.object_pose_ok = true;

  const auto& strict_acm = scene.getAllowedCollisionMatrix();
  out.strict = checkState(scene, combined, strict_acm, max_contacts, max_per_pair, geo.object_name);

  collision_detection::AllowedCollisionMatrix expected_acm = strict_acm;
  if (allow_b_finger_expected)
    expected_acm.setEntry(geo.object_name, kArmBFingerLinks, true);
  out.expected =
      checkState(scene, combined, expected_acm, max_contacts, max_per_pair, geo.object_name);

  out.strict_free = out.object_pose_ok && !out.strict.collision && !out.strict.truncated &&
                    !out.strict.has_upperarm_column && !out.strict.has_finger_self;
  out.expected_free = out.object_pose_ok && !out.expected.collision && !out.expected.truncated &&
                      !out.expected.has_upperarm_column && !out.expected.has_finger_self;
  return out;
}

void printPairList(const std::string& title, const std::vector<std::string>& pairs)
{
  emit("      " + title + ": " + (pairs.empty() ? std::string("(none)") : std::string("")));
  for (const auto& pair : pairs)
    emit("        " + pair);
}

void printDiag(const std::string& title, const CollisionDiag& diag)
{
  emit("    " + title + ": " + std::string(diag.collision ? "COLLISION" : "no collision"));
  emit("      contacts: " + std::to_string(diag.contact_count) +
       (diag.truncated ? " CONTACT ENUMERATION INCOMPLETE" : ""));
  if (diag.pairs.empty())
    emit("      collision pairs: (none)");
  else
  {
    emit("      collision pairs:");
    for (const auto& pair : diag.pairs)
      emit("        " + pair);
  }
  printPairList("Arm A self-collision", diag.arm_a_self);
  printPairList("Arm B self-collision", diag.arm_b_self);
  printPairList("finger self-collision", diag.finger_self);
  printPairList("Arm A <-> Arm B", diag.cross_arm);
  printPairList("gripper A <-> gripper B", diag.gripper_interference);
  printPairList("arm <-> table", diag.arm_table);
  printPairList("arm <-> column", diag.arm_column);
  printPairList("upperarm <-> mounting_column", diag.upperarm_column);
  printPairList("object illegal", diag.object_illegal);
  printPairList("expected A touch", diag.expected_a_touch);
  printPairList("expected B finger-object", diag.expected_b_finger);
  emit(std::string("      upperarm-column: ") + (diag.has_upperarm_column ? "YES" : "NO"));
  emit(std::string("      A finger-object contact observed: ") +
       (diag.has_a_finger_part_contact ? "yes (geometry only)" : "no"));
  emit(std::string("      B finger-object contact observed: ") +
       (diag.has_b_finger_part_contact ? "yes (geometry only, not a grasp proof)" : "no"));
}

void printComboDetail(const ComboResult& item)
{
  emit("  A" + std::to_string(item.a_index) + " + B" + std::to_string(item.b_index) + ":");
  emit("    Arm A joints_rad: " + fmtVec(item.joints_a));
  emit("    Arm B joints_rad: " + fmtVec(item.joints_b));
  emit("    NOMINAL GRIPPER STATE q_A=" + fmtScalar(item.q_a) + " q_B=" + fmtScalar(item.q_b));
  emit("    sliders A l/r: " + fmtVec({item.slider_la, item.slider_ra}) +
       "  B l/r: " + fmtVec({item.slider_lb, item.slider_rb}));
  emit("    FK A: " + fmtScalar(item.a_fk_position_error) + " m / " +
       fmtScalar(item.a_fk_orientation_error_deg) + " deg");
  emit("    FK B: " + fmtScalar(item.b_fk_position_error) + " m / " +
       fmtScalar(item.b_fk_orientation_error_deg) + " deg");
  emit("    joints_ok: " + std::string(item.joints_ok ? "true" : "false") +
       "  object_pose_ok: " + std::string(item.object_pose_ok ? "true" : "false"));
  if (!item.fail_reason.empty() && !item.object_pose_ok)
  {
    emit("    fail reason: " + item.fail_reason);
    return;
  }
  printDiag("STRICT COLLISION RESULT", item.strict);
  printDiag("EXPECTED-CONTACT-AWARE RESULT", item.expected);
}

void printCombos(const std::string& heading, const std::vector<ComboResult>& results)
{
  emit("");
  emit("----------------------------------------");
  emit(heading);
  emit("----------------------------------------");

  int strict_free = 0;
  int expected_free = 0;
  for (const auto& item : results)
  {
    if (item.strict_free)
      ++strict_free;
    if (item.expected_free)
      ++expected_free;
  }
  emit("combinations tested: " + std::to_string(results.size()));
  emit("strict collision-free combinations: " + std::to_string(strict_free));
  emit("expected-contact-aware combinations: " + std::to_string(expected_free));

  emit("");
  emit("strict collision-free combinations:");
  bool any_strict = false;
  for (const auto& item : results)
  {
    if (!item.strict_free)
      continue;
    any_strict = true;
    emit("  A" + std::to_string(item.a_index) + " + B" + std::to_string(item.b_index));
  }
  if (!any_strict)
    emit("  (none)");

  emit("");
  emit("expected-contact-aware combinations:");
  bool any_expected = false;
  for (const auto& item : results)
  {
    if (!item.expected_free)
      continue;
    any_expected = true;
    emit("  A" + std::to_string(item.a_index) + " + B" + std::to_string(item.b_index));
  }
  if (!any_expected)
    emit("  (none)");

  emit("");
  emit("all combinations:");
  for (const auto& item : results)
    printComboDetail(item);

  emit("");
  emit("failed collision pairs:");
  bool any_fail = false;
  for (const auto& item : results)
  {
    if (item.strict_free && item.expected_free)
      continue;
    any_fail = true;
    printComboDetail(item);
  }
  if (!any_fail)
    emit("  (none)");
}

GripperStateInfo inspectGrippers(const moveit::core::RobotModelConstPtr& model,
                                 const std::map<std::string, double>& named)
{
  GripperStateInfo info;
  info.model_has_a = model->hasJointModel(kGripperJointA);
  info.model_has_b = model->hasJointModel(kGripperJointB);
  auto it_a = named.find(kGripperJointA);
  auto it_b = named.find(kGripperJointB);
  info.actual_a = it_a != named.end();
  info.actual_b = it_b != named.end();
  if (info.actual_a)
    info.actual_a_value = it_a->second;
  if (info.actual_b)
    info.actual_b_value = it_b->second;

  std::vector<std::string> missing_shapes;
  for (const auto& link : {kArmAFingerLinks[0], kArmAFingerLinks[1], kArmBFingerLinks[0],
                           kArmBFingerLinks[1], std::string("arm_a_gripper_base_link"),
                           std::string("arm_b_gripper_base_link")})
  {
    if (!linkHasCollisionGeometry(model, link))
      missing_shapes.push_back(link);
  }
  info.finger_collision_geometry = missing_shapes.empty();
  if (!info.finger_collision_geometry)
  {
    info.geometry_note = "MODEL_LIMITATION missing collision geometry: " + joinNames(missing_shapes);
  }
  else
  {
    info.geometry_note =
        "finger/base collision meshes present in URDF (STL); hardware size not independently verified";
  }
  return info;
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("dual_handover_static_collision_test", options);

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
    emit("========== DUAL-4B-2B STATIC COLLISION ==========");
    emit("STATIC CHECK ONLY. NO EXECUTION. NO GRIPPER COMMANDS.");
    emit("NO /apply_planning_scene. Local PlanningScene discarded on exit.");
    emit("NOMINAL GRIPPER STATE only. Not measured physical gripper opening.");

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
    const int max_contacts = getInt(node, "max_contacts", 1000);
    const int max_per_pair = getInt(node, "max_contacts_per_pair", 20);
    const double opening_seed = getDouble(node, "nominal_opening_seed", 0.083);
    const double opening_min = getDouble(node, "opening_search_min", 0.070);
    const double opening_max = getDouble(node, "opening_search_max", 0.099);
    const double opening_step = getDouble(node, "opening_search_step", 0.001);

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

    std::string kin_error;
    if (!loadKinematicsYaml(node, kinematics_yaml, kin_error))
      emit("Kinematics YAML note: " + kin_error);

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
    if (!model)
    {
      emit("RobotModel is null");
      stop();
      return 1;
    }

    emit("");
    emit("RobotModel:");
    emit(std::string("  ") + model->getName());
    emit("Planning frame:");
    emit(std::string("  ") + model->getModelFrame());
    emit("Object:");
    emit("  " + geo.object_name);
    emit("Object radius:");
    emit("  " + fmtScalar(geo.radius) + " m");
    emit("Object length:");
    emit("  " + fmtScalar(geo.length) + " m");
    emit("Arm B X offset:");
    emit("  " + fmtScalar(geo.arm_b_offset_x) + " m");
    emit("Scene source:");
    emit("  /get_planning_scene");

    if (model->getName() != kExpectedModel || model->getModelFrame() != kModelFrame)
    {
      emit("Wrong robot model or planning frame.");
      stop();
      return 1;
    }
    if (!model->hasJointModelGroup(kGroupA) || !model->hasJointModelGroup(kGroupB) ||
        !model->hasJointModelGroup("dual_arms") || !model->hasLinkModel(kTcpA) ||
        !model->hasLinkModel(kTcpB))
    {
      emit("Missing dual-arm groups or TCP frames.");
      stop();
      return 1;
    }

    sensor_msgs::msg::JointState js;
    bool joint_states_stale = false;
    if (!waitForJointStates(node, joint_timeout, joint_max_age, js, error))
    {
      emit("Fresh /joint_states unavailable: " + error);
      emit("Retrying last latched /joint_states regardless of stamp age.");
      emit("This still requires live /move_group; no substitute scene is created.");
      if (!waitForJointStates(node, 3.0, 1.0e9, js, error))
      {
        stop();
        return blocked(error);
      }
      joint_states_stale = true;
      const rclcpp::Time stamp(js.header.stamp);
      const double age = (stamp.nanoseconds() == 0) ? -1.0 : (node->now() - stamp).seconds();
      emit("Using latched /joint_states. stamp age: " +
           (age < 0.0 ? std::string("unset") : fmtScalar(age) + " s"));
      emit(std::string("joint_states_stale: ") + (joint_states_stale ? "YES" : "NO"));
      emit("These joints are a latched ROS control snapshot, not a newly sampled robot pose.");
    }
    std::map<std::string, double> named;
    if (!extractNamedPositions(js, named, error) || !requireArmJoints(named, error))
    {
      emit("Current state read failed: " + error);
      stop();
      return 1;
    }

    moveit_msgs::msg::PlanningScene scene_msg;
    if (!fetchPlanningSceneMsg(node, scene_msg, error))
    {
      stop();
      return blocked(error);
    }
    const SceneSnapshot before_global = snapshotFromMsg(scene_msg);

    auto fetched = std::make_shared<planning_scene::PlanningScene>(model);
    const collision_detection::AllowedCollisionMatrix srdf_acm =
        fetched->getAllowedCollisionMatrix();
    fetched->setPlanningSceneMsg(scene_msg);
    const auto live_acm_check = inspectLiveAcm(scene_msg, fetched->getAllowedCollisionMatrix());
    emit("");
    emit("----------------------------------------");
    emit("LIVE PLANNINGSCENE ACM");
    emit("----------------------------------------");
    emit("GetPlanningScene ACM entry_names: " +
         std::to_string(scene_msg.allowed_collision_matrix.entry_names.size()));
    emit("SRDF/model ACM size (comparison only, not used to replace live ACM): " +
         std::to_string(srdf_acm.getSize()));
    emit("Scene ACM size after setPlanningSceneMsg: " +
         std::to_string(fetched->getAllowedCollisionMatrix().getSize()));
    emit("arm_a_base_link <-> mounting_column:");
    emit("  live scene ACM: " + live_acm_check.a_status);
    emit("  SRDF-only ACM: " + acmEntryStatus(srdf_acm, "arm_a_base_link", kColumnName));
    emit("  message ACM: " + msgAcmStatus(scene_msg, "arm_a_base_link", kColumnName));
    emit("arm_b_base_link <-> mounting_column:");
    emit("  live scene ACM: " + live_acm_check.b_status);
    emit("  SRDF-only ACM: " + acmEntryStatus(srdf_acm, "arm_b_base_link", kColumnName));
    emit("  message ACM: " + msgAcmStatus(scene_msg, "arm_b_base_link", kColumnName));
    emit("arm_a_upperarm_link <-> mounting_column live ACM: " +
         acmEntryStatus(fetched->getAllowedCollisionMatrix(), "arm_a_upperarm_link", kColumnName));
    emit("arm_b_upperarm_link <-> mounting_column live ACM: " +
         acmEntryStatus(fetched->getAllowedCollisionMatrix(), "arm_b_upperarm_link", kColumnName));
    emit("arm_a_finger_l <-> arm_a_finger_r live ACM: " +
         acmEntryStatus(fetched->getAllowedCollisionMatrix(), "arm_a_finger_l", "arm_a_finger_r"));
    emit("arm_b_finger_l <-> arm_b_finger_r live ACM: " +
         acmEntryStatus(fetched->getAllowedCollisionMatrix(), "arm_b_finger_l", "arm_b_finger_r"));
    if (!live_acm_check.complete)
    {
      emit("LIVE ACM INCOMPLETE");
      emit("Expected installation entries missing:");
      for (const auto& item : live_acm_check.missing)
        emit("  " + item);
      emit("DUAL-4B-2B:");
      emit("  INCOMPLETE");
      emit("Continuing read-only diagnostics. This run cannot be an official PASS.");
    }
    else
    {
      emit("Live ACM complete: YES");
      emit("Base-column expected contacts preserved: YES");
    }

    auto& fetched_state = fetched->getCurrentStateNonConst();
    applyNamedJoints(fetched_state, named);
    fetched_state.update();

    emit("");
    emit("World objects:");
    emit("  " + joinNames(before_global.world_ids));
    emit("Attached objects before local test:");
    if (before_global.attached_ids.empty())
      emit("  (none)");
    else
    {
      for (size_t i = 0; i < before_global.attached_ids.size(); ++i)
      {
        emit("  " + before_global.attached_ids[i] + " @ " +
             (i < before_global.attached_links.size() ? before_global.attached_links[i] : "?"));
      }
    }
    emit(std::string("Octomap present in scene: ") + (before_global.has_octomap ? "yes" : "no"));
    emit(std::string("Table present: ") + (before_global.has_table ? "yes" : "NO"));
    emit(std::string("Mounting column present: ") + (before_global.has_column ? "yes" : "NO"));
    const bool env_complete = before_global.has_table && before_global.has_column;
    if (!env_complete)
      emit("Environment check: INCOMPLETE (missing table and/or mounting_column)");

    auto local = planning_scene::PlanningScene::clone(fetched);
    const auto gripper_model_a =
        inspectGripperModel(model, kGripperJointA, kArmAFingerLinks[0], kArmAFingerLinks[1]);
    const auto gripper_model_b =
        inspectGripperModel(model, kGripperJointB, kArmBFingerLinks[0], kArmBFingerLinks[1]);
    GripperStateInfo gripper = inspectGrippers(model, named);
    gripper.q_b_open = 0.0;

    emit("");
    emit("----------------------------------------");
    emit("NOMINAL GRIPPER OPENING PRE-CHECK");
    emit("----------------------------------------");
    emit("These values are NOMINAL GRIPPER STATE, not MEASURED PHYSICAL GRIPPER OPENING.");
    emit("RobotModel gripper groups: " + std::string(kGripperGroupA) + ", " +
         std::string(kGripperGroupB));
    emit("RobotModel mimic joints for " + gripper_model_a.active_joint + ":");
    for (const auto& mimic : gripper_model_a.mimics)
    {
      emit("  " + mimic.name + " = " + fmtScalar(mimic.factor) + " * q + " +
           fmtScalar(mimic.offset));
    }
    emit("RobotModel mimic joints for " + gripper_model_b.active_joint + ":");
    for (const auto& mimic : gripper_model_b.mimics)
    {
      emit("  " + mimic.name + " = " + fmtScalar(mimic.factor) + " * q + " +
           fmtScalar(mimic.offset));
    }
    OpeningChoice opening_a =
        searchNominalOpening(*local, local->getCurrentState(), gripper_model_a, opening_seed,
                             opening_min, opening_max, opening_step, 2.0 * geo.radius,
                             max_contacts, max_per_pair);
    OpeningChoice opening_b =
        searchNominalOpening(*local, local->getCurrentState(), gripper_model_b, opening_seed,
                             opening_min, opening_max, opening_step, 2.0 * geo.radius,
                             max_contacts, max_per_pair);
    auto printOpeningSearch = [](const std::string& label, const OpeningChoice& choice) {
      emit(label + ":");
      emit("  candidate log: q | sliders l/r | finger origin sep | finger-finger | mimic");
      for (const auto& sample : choice.samples)
      {
        emit("  q=" + fmtScalar(sample.q) + " sliders=[" + fmtScalar(sample.slider_l) + ", " +
             fmtScalar(sample.slider_r) + "] origin_sep=" + fmtScalar(sample.finger_origin_sep) +
             " finger-finger=" + std::string(sample.finger_self_collision ? "YES" : "NO") +
             " mimic=" + (sample.mimic_ok ? "OK" : sample.mimic_error) +
             (sample.truncated ? " CONTACT ENUMERATION INCOMPLETE" : ""));
      }
      emit(std::string("  NOMINAL OPENING FOUND: ") + (choice.found ? "YES" : "NO"));
      if (choice.found)
        emit("  selected q: " + fmtScalar(choice.q));
      emit("  PHYSICAL GRASP FEASIBILITY:");
      emit("    INCOMPLETE");
    };
    printOpeningSearch("Arm A isolated opening search", opening_a);
    printOpeningSearch("Arm B isolated opening search", opening_b);
    gripper.opening_found = opening_a.found && opening_b.found;
    gripper.q_a_grasp = opening_a.found ? opening_a.q : opening_seed;
    gripper.q_b_receive = opening_b.found ? opening_b.q : opening_seed;

    std::string gripper_set_error;
    if (!setActiveGripperJoint(local->getCurrentStateNonConst(), gripper_model_a, gripper.q_a_grasp,
                               gripper_set_error) ||
        !setActiveGripperJoint(local->getCurrentStateNonConst(), gripper_model_b, gripper.q_b_open,
                               gripper_set_error))
    {
      emit("Cannot set staged NOMINAL GRIPPER STATE: " + gripper_set_error);
    }
    printGripperMimic(local->getCurrentState(), gripper_model_a, gripper_model_b,
                      "After staged set (A grasp / B open)");
    emit("Local object attachment:");
    emit(std::string("  ") + kTcpA);

    if (!attachLocalObject(*local, geo, error))
    {
      emit("Local attach failed: " + error);
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

    bool solvers_ok = true;
    solvers_ok &= describeSolver(node, model, kGroupA, kTcpA, arm_a);
    solvers_ok &= describeSolver(node, model, kGroupB, kTcpB, arm_b_pre);
    arm_b_handover.solver_name = arm_b_pre.solver_name;
    arm_b_handover.solver_tip = arm_b_pre.solver_tip;
    arm_b_handover.solver_base = arm_b_pre.solver_base;
    if (!solvers_ok)
    {
      emit("IK solver missing or cannot solve for the required TCP.");
      stop();
      return 1;
    }

    moveit::core::RobotState current(model);
    current.setToDefaultValues();
    applyNamedJoints(current, named);
    current.update();

    std::vector<double> step14_c;
    std::string seed_error;
    const bool have_step14 = loadStep14ArmASeed(step14_yaml, step14_c, seed_error);
    if (!have_step14)
      emit("STEP14 C-face seed skipped: " + seed_error);

    std::vector<std::pair<std::string, std::vector<double>>> seeds_a;
    seeds_a.push_back({"current", jointsOf(current, kArmAJoints)});
    if (have_step14)
      seeds_a.push_back({"step14_c_face", step14_c});
    std::vector<std::pair<std::string, std::vector<double>>> seeds_b;
    seeds_b.push_back({"current", jointsOf(current, kArmBJoints)});

    searchIk(arm_a, current, seeds_a, max_attempts, max_unique, timeout_exact, timeout_nearby,
             nearby_radius, nearby_radius_local, min_distance, pos_tol, ori_tol, rng_seed);
    searchIk(arm_b_pre, current, seeds_b, max_attempts, max_unique, timeout_exact, timeout_nearby,
             nearby_radius, nearby_radius_local, min_distance, pos_tol, ori_tol, rng_seed + 17);
    searchIk(arm_b_handover, current, seeds_b, max_attempts, max_unique, timeout_exact,
             timeout_nearby, nearby_radius, nearby_radius_local, min_distance, pos_tol, ori_tol,
             rng_seed + 31);

    printTarget(arm_a);
    printTarget(arm_b_pre);
    printTarget(arm_b_handover);

    bool object_pose_pass = false;
    if (!arm_a.unique.empty())
    {
      moveit::core::RobotState probe(local->getCurrentState());
      applyGroupJoints(probe, kArmAJoints, arm_a.unique.front().joints);
      probe.update();
      double pos_err = 0.0;
      double ori_err = 0.0;
      object_pose_pass =
          verifyObjectPose(probe, geo, pos_tol, ori_tol, pos_err, ori_err, error);
      emit("");
      emit("Object pose consistency:");
      emit(object_pose_pass ? "  PASS" : "  FAIL");
      std::ostringstream line;
      line.setf(std::ios::fixed);
      line << std::setprecision(6) << "  position error: " << pos_err
           << " m  orientation error: " << ori_err << " deg";
      emit(line.str());
      if (!object_pose_pass)
        emit("  " + error);
    }
    else
    {
      emit("");
      emit("Object pose consistency:");
      emit("  FAIL");
      emit("  no Arm A IK to place the attached object at the handover pose");
    }

    emit("Gripper joint state source:");
    emit("  NOMINAL GRIPPER STATE (model), not MEASURED PHYSICAL GRIPPER OPENING");
    emit("  /joint_states " + std::string(kGripperJointA) + ": " +
         (gripper.actual_a ? fmtVec({gripper.actual_a_value}) : std::string("missing")) +
         " (ROS control value, not claimed as true stroke)");
    emit("  /joint_states " + std::string(kGripperJointB) + ": " +
         (gripper.actual_b ? fmtVec({gripper.actual_b_value}) : std::string("missing")) +
         " (ROS control value, not claimed as true stroke)");
    emit("Arm A nominal grasp q: " + fmtScalar(gripper.q_a_grasp));
    emit("Arm B open q: " + fmtScalar(gripper.q_b_open));
    emit("Arm B nominal receiving q: " + fmtScalar(gripper.q_b_receive));
    emit("Arm A touch links (existing grasp model, prefixed):");
    emit("  " + joinNames(kArmATouchLinks));
    emit("Arm B expected-contact finger links (STATE 3 local ACM only):");
    emit("  " + joinNames(kArmBFingerLinks));
    emit("Gripper collision geometry:");
    emit("  " + gripper.geometry_note);

    if (!arm_a.unique.empty())
    {
      moveit::core::RobotState grasp_probe(local->getCurrentState());
      applyGroupJoints(grasp_probe, kArmAJoints, arm_a.unique.front().joints);
      std::string grasp_error;
      setActiveGripperJoint(grasp_probe, gripper_model_a, gripper.q_a_grasp, grasp_error);
      setActiveGripperJoint(grasp_probe, gripper_model_b, gripper.q_b_open, grasp_error);
      const auto grasp_diag = checkState(*local, grasp_probe, local->getAllowedCollisionMatrix(),
                                         max_contacts, max_per_pair, geo.object_name);
      opening_a.both_fingers_contact_object =
          grasp_diag.has_a_finger_part_contact &&
          inList(grasp_diag.expected_a_touch, pairKey(geo.object_name, "arm_a_finger_l")) &&
          inList(grasp_diag.expected_a_touch, pairKey(geo.object_name, "arm_a_finger_r"));
      opening_a.object_in_grasp_region = object_pose_pass;
      emit("");
      emit("Arm A finger-object evidence at first A IK, NOMINAL grasp q:");
      emit("  both fingers listed vs object: " +
           std::string(opening_a.both_fingers_contact_object ? "yes" : "no"));
      emit("  object pose consistent: " + std::string(object_pose_pass ? "yes" : "no"));
      emit("  finger-finger self-collision: " +
           std::string(grasp_diag.has_finger_self ? "YES" : "NO"));
      if (!opening_a.both_fingers_contact_object)
      {
        emit("  No certified inner-surface grasp. PHYSICAL GRASP FEASIBILITY: INCOMPLETE");
      }
    }

    auto eval = [&](const IkCandidate& a_cand, const IkCandidate& b_cand, size_t a_index,
                    size_t b_index, const Eigen::Isometry3d& b_target, double q_a, double q_b,
                    bool allow_b_expected) {
      return evaluateCombo(*local, geo, gripper_model_a, gripper_model_b, a_cand, b_cand, a_index,
                           b_index, b_target, q_a, q_b, pos_tol, ori_tol, max_contacts,
                           max_per_pair, allow_b_expected);
    };

    std::vector<ComboResult> state1;
    std::vector<ComboResult> state2;
    std::vector<ComboResult> state3;
    for (size_t i = 0; i < arm_a.unique.size(); ++i)
    {
      for (size_t j = 0; j < arm_b_pre.unique.size(); ++j)
      {
        auto item = eval(arm_a.unique[i], arm_b_pre.unique[j], i + 1, j + 1, geo.world_pre_b,
                         gripper.q_a_grasp, gripper.q_b_open, false);
        item.state_name = "STATE 1";
        state1.push_back(std::move(item));
      }
      for (size_t k = 0; k < arm_b_handover.unique.size(); ++k)
      {
        auto item2 = eval(arm_a.unique[i], arm_b_handover.unique[k], i + 1, k + 1, geo.world_tcp_b,
                          gripper.q_a_grasp, gripper.q_b_open, false);
        item2.state_name = "STATE 2";
        state2.push_back(item2);
        auto item3 = eval(arm_a.unique[i], arm_b_handover.unique[k], i + 1, k + 1, geo.world_tcp_b,
                          gripper.q_a_grasp, gripper.q_b_receive, true);
        item3.state_name = "STATE 3";
        state3.push_back(std::move(item3));
      }
    }

    emit("");
    emit("A IK candidates: " + std::to_string(arm_a.unique.size()));
    emit("B Pre IK candidates: " + std::to_string(arm_b_pre.unique.size()));
    emit("B Handover IK candidates: " + std::to_string(arm_b_handover.unique.size()));
    printCombos("STATE 1: A handover + B pre/open", state1);
    printCombos("STATE 2: A handover + B handover/open", state2);
    printCombos("STATE 3: A handover + B handover/receiving", state3);

    emit("");
    emit("----------------------------------------");
    emit("COMMON ARM A CANDIDATES");
    emit("----------------------------------------");

    std::vector<std::string> triplets;
    bool any_common = false;
    std::set<std::string> all_gripper_interference;
    std::set<std::string> all_strict_pairs;
    bool any_truncated = false;
    bool any_b_finger_part = false;
    auto accumulate = [&](const std::vector<ComboResult>& results) {
      for (const auto& item : results)
      {
        any_truncated = any_truncated || item.strict.truncated || item.expected.truncated;
        any_b_finger_part = any_b_finger_part || item.strict.has_b_finger_part_contact ||
                            item.expected.has_b_finger_part_contact;
        for (const auto& pair : item.strict.gripper_interference)
          all_gripper_interference.insert(pair);
        for (const auto& pair : item.strict.pairs)
          all_strict_pairs.insert(pair);
      }
    };
    accumulate(state1);
    accumulate(state2);
    accumulate(state3);

    std::string block_reason = "no IK candidates";
    if (arm_a.unique.empty())
      block_reason = "STATE search blocked: no Arm A Handover IK";
    else if (arm_b_pre.unique.empty())
      block_reason = "STATE 1 blocked: no Arm B PreHandover IK";
    else if (arm_b_handover.unique.empty())
      block_reason = "STATE 2/3 blocked: no Arm B Handover IK";

    for (size_t i = 0; i < arm_a.unique.size(); ++i)
    {
      std::vector<size_t> pre_ok;
      std::vector<size_t> han_ok;
      for (const auto& item : state1)
      {
        if (item.a_index == i + 1 && item.strict_free)
          pre_ok.push_back(item.b_index);
      }
      for (const auto& item2 : state2)
      {
        if (item2.a_index != i + 1 || !item2.strict_free)
          continue;
        bool state3_ok = false;
        for (const auto& item3 : state3)
        {
          if (item3.a_index == i + 1 && item3.b_index == item2.b_index && item3.expected_free)
          {
            state3_ok = true;
            break;
          }
        }
        if (state3_ok)
          han_ok.push_back(item2.b_index);
      }
      emit("");
      emit("A candidate: " + std::to_string(i + 1));
      emit("  Compatible B Pre candidates (STATE 1 strict): " +
           (pre_ok.empty() ? std::string("(none)") : joinNames([&]() {
             std::vector<std::string> names;
             for (size_t idx : pre_ok)
               names.push_back("B" + std::to_string(idx));
             return names;
           }())));
      emit("  Compatible B Handover candidates (STATE 2 strict AND STATE 3 expected-aware): " +
           (han_ok.empty() ? std::string("(none)") : joinNames([&]() {
             std::vector<std::string> names;
             for (size_t idx : han_ok)
               names.push_back("B" + std::to_string(idx));
             return names;
           }())));
      if (pre_ok.empty() && han_ok.empty())
        block_reason = "A" + std::to_string(i + 1) + " blocked in STATE 1 and STATE 2/3";
      else if (pre_ok.empty())
        block_reason = "A" + std::to_string(i + 1) + " blocked in STATE 1";
      else if (han_ok.empty())
        block_reason = "A" + std::to_string(i + 1) + " blocked in STATE 2/3";
      if (!pre_ok.empty() && !han_ok.empty())
      {
        any_common = true;
        for (size_t j : pre_ok)
        {
          for (size_t k : han_ok)
          {
            const std::string triple = "(A" + std::to_string(i + 1) + ", B_pre" +
                                       std::to_string(j) + ", B_handover" + std::to_string(k) + ")";
            triplets.push_back(triple);
          }
        }
      }
    }
    emit("");
    emit("Common A candidate / compatible B Pre / compatible B Handover:");
    emit("STATE 1/2/3 compatible triplets:");
    if (triplets.empty())
    {
      emit("  (none)");
      emit("  block reason: " + block_reason);
    }
    else
    {
      for (const auto& triple : triplets)
        emit("  " + triple);
    }

    emit("");
    emit("----------------------------------------");
    emit("GRIPPER INTERFERENCE");
    emit("----------------------------------------");
    emit("Modeled robot/gripper geometry: checked");
    emit("Physical gripper opening: NOT VERIFIED");
    emit("Physical grasp force: NOT VERIFIED");
    emit("Grasp contact geometry: INCOMPLETE");
    if (all_gripper_interference.empty())
      emit("Modeled Arm A gripper <-> Arm B gripper contacts: none in tested combinations");
    else
    {
      emit("Modeled Arm A gripper <-> Arm B gripper contacts:");
      for (const auto& pair : all_gripper_interference)
        emit("  " + pair);
    }
    emit(std::string("Arm B finger <-> object contacts observed: ") +
         (any_b_finger_part ? "yes (geometry only, not a grasp proof)" : "no"));
    if (any_truncated)
    {
      emit("CONTACT ENUMERATION INCOMPLETE");
      emit("At least one check hit max_contacts; collision-free claims from truncated lists are invalid.");
    }

    emit("");
    emit("Local diagnostic ACM additions (STATE 3 expected-contact-aware only, not written globally):");
    emit("  " + geo.object_name + " <-> arm_b_finger_l");
    emit("  " + geo.object_name + " <-> arm_b_finger_r");
    emit("Arm A touch_links remain those of the existing prefixed grasp model.");
    emit("No arm_a/arm_b blanket allowance. No upperarm-column allowance. No finger-finger allowance.");
    emit("small_part remains attached to arm_a_gripper_tcp through STATE 3. No attachment transfer.");

    moveit_msgs::msg::PlanningScene after_msg;
    std::string after_error;
    const bool after_ok = fetchPlanningSceneMsg(node, after_msg, after_error);
    SceneSnapshot after_global = after_ok ? snapshotFromMsg(after_msg) : SceneSnapshot();
    const bool global_same = after_ok && sceneUnchanged(before_global, after_global);

    emit("");
    emit("----------------------------------------");
    emit("GLOBAL PLANNINGSCENE PRE/POST");
    emit("----------------------------------------");
    emit("This test issued global scene write: NO");
    emit("Before world IDs: " + joinNames(before_global.world_ids));
    emit("Before attached IDs: " + joinNames(before_global.attached_ids));
    for (const auto& line : before_global.world_pose_lines)
      emit("  before pose " + line);
    emit("Before ACM a_base-column: " + before_global.acm_a_base_column_status);
    emit("Before ACM b_base-column: " + before_global.acm_b_base_column_status);
    if (!after_ok)
      emit("post-check GetPlanningScene failed: " + after_error);
    else
    {
      emit("After world IDs: " + joinNames(after_global.world_ids));
      emit("After attached IDs: " + joinNames(after_global.attached_ids));
      for (const auto& line : after_global.world_pose_lines)
        emit("  after pose " + line);
      emit("After ACM a_base-column: " + after_global.acm_a_base_column_status);
      emit("After ACM b_base-column: " + after_global.acm_b_base_column_status);
      emit(std::string("Global scene changed: ") + (global_same ? "NO" : "YES"));
      if (!global_same)
      {
        printSnapshotDiff(before_global, after_global);
        emit("Cause: UNDETERMINED");
        emit("This test issued global scene write: NO");
      }
    }

    emit("");
    emit("----------------------------------------");
    emit("FINAL");
    emit("----------------------------------------");

    int s1_strict = 0, s2_strict = 0, s3_expected = 0;
    for (const auto& item : state1)
      if (item.strict_free)
        ++s1_strict;
    for (const auto& item : state2)
      if (item.strict_free)
        ++s2_strict;
    for (const auto& item : state3)
      if (item.expected_free)
        ++s3_expected;

    const bool data_complete = env_complete && live_acm_check.complete && object_pose_pass &&
                               gripper.opening_found && gripper.finger_collision_geometry &&
                               arm_a.pass && arm_b_pre.pass && arm_b_handover.pass && !any_truncated;
    std::string verdict = "INCOMPLETE";
    if (!live_acm_check.complete || !env_complete || !gripper.opening_found || any_truncated ||
        !gripper.finger_collision_geometry)
    {
      verdict = "INCOMPLETE";
    }
    else if (!object_pose_pass && !arm_a.unique.empty())
      verdict = "FAIL";
    else if (data_complete && any_common)
      verdict = "PASS";
    else if (data_complete && !any_common)
      verdict = "FAIL";

    emit("STATE 1 strict-free: " + std::to_string(s1_strict) + " / " +
         std::to_string(state1.size()));
    emit("STATE 2 strict-free: " + std::to_string(s2_strict) + " / " +
         std::to_string(state2.size()));
    emit("STATE 3 expected-contact-aware: " + std::to_string(s3_expected) + " / " +
         std::to_string(state3.size()));
    emit("Live ACM complete: " + std::string(live_acm_check.complete ? "YES" : "NO"));
    emit("Base-column expected contacts preserved: " +
         std::string((live_acm_check.a_base_column && live_acm_check.b_base_column) ? "YES" : "NO"));
    if (verdict == "PASS")
      emit("DUAL-4B-2B STATIC MODEL PASS");
    else if (verdict == "FAIL")
      emit("DUAL-4B-2B STATIC MODEL FAIL");
    else
      emit("DUAL-4B-2B INCOMPLETE");
    emit("Physical gripper opening:");
    emit("  NOT VERIFIED");
    emit("Physical grasp force:");
    emit("  NOT VERIFIED");
    emit("Continuous robot trajectory:");
    emit("  NOT CHECKED");
    emit("Handover geometry changed: NO");
    emit("Arm B X offset changed: NO");
    emit("URDF/SRDF changed: NO");
    emit("Robot mounting changed: NO");
    emit("World geometry changed: NO");
    emit("Global ACM changed: NO");
    emit("Global PlanningScene write issued: NO");
    emit("Real robot motion commands: ZERO");
    emit("Real gripper commands: ZERO");

    if (!all_strict_pairs.empty())
    {
      emit("");
      emit("Union of strict collision pairs across all tested states:");
      for (const auto& pair : all_strict_pairs)
        emit("  " + pair);
    }

    stop();
    if (verdict == "PASS")
      return 0;
    return verdict == "INCOMPLETE" ? 3 : 2;
  }
  catch (const std::exception& e)
  {
    emit(std::string("STAGE exception: ") + e.what());
    emit("DUAL-4B-2 aborted with a caught exception. No robot commands were sent.");
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
