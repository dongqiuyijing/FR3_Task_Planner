// DUAL-6: Arm B H→I1→I2→I3 time-first optimization. Reuses DUAL-5-R geometry,
// IK, graph, collision, and Attachment validation. PLAN ONLY / OFFLINE TOTG.
// No execute, no gripper commands, no scene apply, no joint-limit edits.
#include <algorithm>
#include <array>
#include <cctype>
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
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>
#include <rclcpp/parameter_client.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <yaml-cpp/yaml.h>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/collision_detection/collision_matrix.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/joint_model.h>
#include <moveit/robot_model/revolute_joint_model.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/attached_body.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <geometric_shapes/shapes.h>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/motion_plan_request.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/planning_scene_components.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

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
  Eigen::Isometry3d tcp_b_object = Eigen::Isometry3d::Identity();
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

struct PipelineInfo
{
  bool reachable = false;
  bool pilz_loaded = false;
  std::vector<std::string> pipelines;
  std::string note;
};

struct PathSample
{
  int index = 0;
  double s = 0.0;
  Eigen::Vector3d tcp = Eigen::Vector3d::Zero();
  std::vector<double> joints_a;
  std::vector<double> joints_b;
  double q_a = 0.0;
  double q_b = 0.0;
  double line_error = 0.0;
  double ori_error_deg = 0.0;
  double joint_step = 0.0;
  double max_abs_joint_step = 0.0;
  double object_pos_error = 0.0;
  double object_ori_error_deg = 0.0;
  CollisionDiag diag;
  std::string fail_reason;
};

struct PathResult
{
  size_t a_index = 0;
  size_t pre_index = 0;
  size_t han_index = 0;
  bool pass = false;
  bool ik_discontinuity = false;
  bool collision_fail = false;
  bool end_mismatch = false;
  bool truncated = false;
  bool b_finger_object_during_approach = false;
  bool arm_a_unchanged = true;
  bool x_monotonic = true;
  bool yz_constant = true;
  bool orientation_constant = true;
  bool cartesian_ok = false;
  std::string fail_stage;
  std::string fail_reason;
  std::vector<PathSample> samples;
  double max_line_error = 0.0;
  double max_ori_error = 0.0;
  double max_adjacent_joint_change = 0.0;
  double arm_b_path_length = 0.0;
  double max_cartesian_gap = 0.0;
  double max_joint_gap = 0.0;
  Eigen::Vector3d start_tcp = Eigen::Vector3d::Zero();
  Eigen::Vector3d end_tcp = Eigen::Vector3d::Zero();
  std::vector<double> end_joints_b;
  int matched_han_index = -1;
  double end_to_k_l2 = std::numeric_limits<double>::infinity();
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
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_BOOL)
    return parameter.as_bool() ? 1 : 0;
  return fallback;
}

bool getBool(const rclcpp::Node::SharedPtr& node, const std::string& name, bool fallback)
{
  if (!node->has_parameter(name))
    return fallback;
  const auto parameter = node->get_parameter(name);
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_BOOL)
    return parameter.as_bool();
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
    return parameter.as_int() != 0;
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
    return parameter.as_double() != 0.0;
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

void setBoolParam(const rclcpp::Node::SharedPtr& node, const std::string& name, bool value)
{
  if (!node->has_parameter(name))
    node->declare_parameter<bool>(name, value);
  else
    node->set_parameter(rclcpp::Parameter(name, value));
}

bool overlayJointLimitsFromMoveGroup(const rclcpp::Node::SharedPtr& node)
{
  auto client = std::make_shared<rclcpp::SyncParametersClient>(node, "/move_group");
  if (!client->wait_for_service(2s))
    return false;
  std::vector<std::string> names;
  const std::vector<std::string> joints = [] {
    std::vector<std::string> all = kArmAJoints;
    all.insert(all.end(), kArmBJoints.begin(), kArmBJoints.end());
    return all;
  }();
  names.push_back("robot_description_planning.default_velocity_scaling_factor");
  names.push_back("robot_description_planning.default_acceleration_scaling_factor");
  for (const auto& joint : joints)
  {
    const std::string prefix = "robot_description_planning.joint_limits." + joint + ".";
    names.push_back(prefix + "has_velocity_limits");
    names.push_back(prefix + "max_velocity");
    names.push_back(prefix + "has_acceleration_limits");
    names.push_back(prefix + "max_acceleration");
  }
  try
  {
    const auto values = client->get_parameters(names);
    for (const auto& parameter : values)
    {
      if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET)
        continue;
      if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_BOOL)
        setBoolParam(node, parameter.get_name(), parameter.as_bool());
      else if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
        setDoubleParam(node, parameter.get_name(), parameter.as_double());
      else if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
        setDoubleParam(node, parameter.get_name(), static_cast<double>(parameter.as_int()));
    }
  }
  catch (const std::exception& e)
  {
    emit(std::string("STAGE overlay: /move_group joint_limits skipped: ") + e.what());
    return false;
  }
  return true;
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
    error = "DUAL-4C must remain plan-only; execution.enabled is true";
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
  geo.tcp_b_object = geo.world_tcp_b.inverse() * geo.world_object;

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
  emit("DUAL-6 BLOCKED");
  emit("Existing /move_group unavailable.");
  emit("reason: " + why);
  emit("Arm B time-first optimization: NOT PERFORMED");
  emit("Global PlanningScene modified: NO");
  emit("Real robot commands sent: ZERO");
  return 4;
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

int countNamedAttachments(const moveit::core::RobotState& state, const std::string& id)
{
  std::vector<const moveit::core::AttachedBody*> bodies;
  state.getAttachedBodies(bodies);
  int n = 0;
  for (const auto* body : bodies)
  {
    if (body && body->getName() == id)
      ++n;
  }
  return n;
}

bool getAttachedWorldPose(const moveit::core::RobotState& state, const std::string& id,
                          Eigen::Isometry3d& world, std::string& error)
{
  if (!state.hasAttachedBody(id))
  {
    error = "attached body missing: " + id;
    return false;
  }
  const auto* body = state.getAttachedBody(id);
  if (!body || body->getGlobalCollisionBodyTransforms().empty())
  {
    error = "attached body has no world collision transform: " + id;
    return false;
  }
  world = body->getGlobalCollisionBodyTransforms().front();
  return true;
}

bool attachObjectToLink(planning_scene::PlanningScene& scene, const HandoverGeometry& geo,
                        const std::string& link, const Eigen::Isometry3d& pose_in_link,
                        const std::vector<std::string>& touch_links, std::string& error)
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
  attached.link_name = link;
  attached.touch_links = touch_links;
  attached.object.id = geo.object_name;
  attached.object.header.frame_id = link;
  attached.object.operation = moveit_msgs::msg::CollisionObject::ADD;
  attached.object.pose.orientation.w = 1.0;
  attached.object.primitives.resize(1);
  attached.object.primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
  attached.object.primitives[0].dimensions.resize(2);
  attached.object.primitives[0].dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT] =
      geo.length;
  attached.object.primitives[0].dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS] =
      geo.radius;
  attached.object.primitive_poses.push_back(toPoseMsg(pose_in_link));
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
  if (countNamedAttachments(scene.getCurrentState(), geo.object_name) != 1)
  {
    error = "duplicate attached object after attach: count=" +
            std::to_string(countNamedAttachments(scene.getCurrentState(), geo.object_name));
    return false;
  }
  const auto* body = scene.getCurrentState().getAttachedBody(geo.object_name);
  if (!body || body->getAttachedLinkName() != link)
  {
    error = "object not attached to " + link;
    return false;
  }
  return true;
}

bool attachLocalObject(planning_scene::PlanningScene& scene, const HandoverGeometry& geo,
                       std::string& error)
{
  return attachObjectToLink(scene, geo, kTcpA, geo.tcp_a_object, kArmATouchLinks, error);
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

Eigen::Isometry3d cartesianPoseB(const HandoverGeometry& geo, double s)
{
  Eigen::Isometry3d pose = geo.world_pre_b;
  pose.translation() =
      geo.world_pre_b.translation() +
      s * (geo.world_tcp_b.translation() - geo.world_pre_b.translation());
  pose.linear() = geo.world_tcp_b.linear();
  return pose;
}

double maxAbsJointDelta(const std::vector<double>& a, const std::vector<double>& b)
{
  const size_t n = std::min(a.size(), b.size());
  double m = 0.0;
  for (size_t i = 0; i < n; ++i)
    m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

double lineDeviation(const Eigen::Vector3d& p, const Eigen::Vector3d& a, const Eigen::Vector3d& b)
{
  const Eigen::Vector3d d = b - a;
  const double len2 = d.squaredNorm();
  const double t = len2 > 1e-18 ? (p - a).dot(d) / len2 : 0.0;
  return (p - (a + t * d)).norm();
}

std::string collisionCategory(const CollisionDiag& diag)
{
  if (diag.has_upperarm_column)
    return "upperarm <-> mounting_column";
  if (diag.has_finger_self)
    return "finger self-collision";
  if (diag.has_b_finger_part_contact)
    return "Arm B finger <-> object (illegal during open approach)";
  if (!diag.gripper_interference.empty())
    return "gripper A <-> gripper B";
  if (!diag.cross_arm.empty())
    return "Arm A <-> Arm B";
  if (!diag.arm_column.empty())
    return "arm <-> column";
  if (!diag.arm_table.empty())
    return "arm <-> table";
  if (diag.has_a_finger_part_contact)
    return "Arm A finger <-> object";
  if (!diag.object_illegal.empty())
    return "object illegal contact";
  if (!diag.arm_a_self.empty())
    return "Arm A self-collision";
  if (!diag.arm_b_self.empty())
    return "Arm B self-collision";
  if (!diag.pairs.empty())
    return "other collision";
  return "none";
}

PipelineInfo inspectMoveGroupPipelines(const rclcpp::Node::SharedPtr& node)
{
  PipelineInfo info;
  auto probe = rclcpp::Node::make_shared(std::string(node->get_name()) + "_pipeline_probe");
  auto client = std::make_shared<rclcpp::SyncParametersClient>(probe, "/move_group");
  if (!client->wait_for_service(3s))
  {
    info.note = "/move_group parameter service unreachable";
    return info;
  }
  info.reachable = true;
  try
  {
    if (client->has_parameter("planning_pipelines"))
    {
      const auto parameters = client->get_parameters({"planning_pipelines"});
      if (!parameters.empty())
      {
        const auto& parameter = parameters.front();
        if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_STRING_ARRAY)
          info.pipelines = parameter.as_string_array();
        else if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
          info.pipelines.push_back(parameter.as_string());
      }
    }
    const auto listed = client->list_parameters({}, 2);
    for (const auto& name : listed.names)
    {
      const auto lower = [&]() {
        std::string copy = name;
        for (char& ch : copy)
          ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return copy;
      }();
      if (lower.find("pilz") != std::string::npos)
        info.pilz_loaded = true;
    }
    for (const auto& pipeline : info.pipelines)
    {
      std::string lower = pipeline;
      for (char& ch : lower)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
      if (lower.find("pilz") != std::string::npos)
        info.pilz_loaded = true;
    }
  }
  catch (const std::exception& e)
  {
    info.note = e.what();
  }
  if (info.pipelines.empty() && info.note.empty())
    info.note = "planning_pipelines parameter missing or empty";
  return info;
}

PathSample solveCartesianSample(planning_scene::PlanningScene& scene, const HandoverGeometry& geo,
                                const GripperModelInfo& gripper_a, const GripperModelInfo& gripper_b,
                                const std::vector<double>& joints_a, const std::vector<double>& seed_b,
                                double s, bool use_ik, double q_a, double q_b, double ik_timeout,
                                double pos_tol, double ori_tol_deg, int max_contacts, int max_per_pair)
{
  PathSample out;
  out.s = s;
  const Eigen::Isometry3d target = cartesianPoseB(geo, s);
  const Eigen::Vector3d p0 = geo.world_pre_b.translation();
  const Eigen::Vector3d p1 = geo.world_tcp_b.translation();

  moveit::core::RobotState state(scene.getCurrentState());
  applyGroupJoints(state, kArmAJoints, joints_a);
  applyGroupJoints(state, kArmBJoints, seed_b);
  if (!setActiveGripperJoint(state, gripper_a, q_a, out.fail_reason) ||
      !setActiveGripperJoint(state, gripper_b, q_b, out.fail_reason))
  {
    return out;
  }

  const auto* group_a = state.getJointModelGroup(kGroupA);
  const auto* group_b = state.getJointModelGroup(kGroupB);
  if (!group_a || !group_b)
  {
    out.fail_reason = "missing arm_a or arm_b JointModelGroup";
    return out;
  }

  if (use_ik)
  {
    if (!state.setFromIK(group_b, target, kTcpB, ik_timeout))
    {
      out.fail_reason = "Arm B IK failed at s=" + fmtScalar(s);
      return out;
    }
    applyGroupJoints(state, kArmAJoints, joints_a);
    if (!setActiveGripperJoint(state, gripper_a, q_a, out.fail_reason) ||
        !setActiveGripperJoint(state, gripper_b, q_b, out.fail_reason))
    {
      return out;
    }
  }
  state.update();

  out.joints_a = jointsOf(state, kArmAJoints);
  out.joints_b = jointsOf(state, kArmBJoints);
  out.q_a = jointOrNan(state, kGripperJointA);
  out.q_b = jointOrNan(state, kGripperJointB);
  const Eigen::Isometry3d tcp_b = state.getGlobalLinkTransform(kTcpB);
  out.tcp = tcp_b.translation();
  out.line_error = lineDeviation(out.tcp, p0, p1);
  double pos_err = 0.0;
  poseError(target, tcp_b, pos_err, out.ori_error_deg);

  if (jointL2(out.joints_a, joints_a) > 1e-9)
  {
    out.fail_reason = "Arm A joints changed during Arm B approach";
    return out;
  }
  if (!nearlyEqual(out.q_a, q_a, 1e-6) || !nearlyEqual(out.q_b, q_b, 1e-6))
  {
    out.fail_reason = "gripper nominal opening changed: q_A=" + fmtScalar(out.q_a) +
                      " q_B=" + fmtScalar(out.q_b);
    return out;
  }
  if (!allVariablesFinite(state, out.fail_reason))
    return out;
  if (!state.satisfiesBounds(group_a) || !state.satisfiesBounds(group_b))
  {
    out.fail_reason = "combined state outside joint bounds at s=" + fmtScalar(s);
    return out;
  }
  if (pos_err > pos_tol || out.ori_error_deg > ori_tol_deg)
  {
    out.fail_reason = "Arm B FK missed Cartesian line target at s=" + fmtScalar(s) +
                      " pos=" + fmtScalar(pos_err) + " m ori=" + fmtScalar(out.ori_error_deg) +
                      " deg";
    return out;
  }
  if (!verifyObjectPose(state, geo, pos_tol, ori_tol_deg, out.object_pos_error,
                        out.object_ori_error_deg, out.fail_reason))
  {
    return out;
  }

  out.diag = checkState(scene, state, scene.getAllowedCollisionMatrix(), max_contacts, max_per_pair,
                        geo.object_name);
  if (out.diag.truncated)
  {
    out.fail_reason = "CONTACT ENUMERATION INCOMPLETE at s=" + fmtScalar(s);
    return out;
  }
  if (out.diag.collision || out.diag.has_upperarm_column || out.diag.has_finger_self)
  {
    out.fail_reason = "full-robot collision at s=" + fmtScalar(s) + " category=" +
                      collisionCategory(out.diag);
    if (!out.diag.pairs.empty())
      out.fail_reason += " pair=" + out.diag.pairs.front();
    return out;
  }
  return out;
}

PathResult validateCartesianPath(planning_scene::PlanningScene& scene, const HandoverGeometry& geo,
                                 const GripperModelInfo& gripper_a, const GripperModelInfo& gripper_b,
                                 const IkCandidate& a_cand, const IkCandidate& b_pre,
                                 const IkCandidate& b_han, size_t a_index, size_t pre_index,
                                 size_t han_index, double q_a, double q_b, double cartesian_step,
                                 double densify_min, double max_joint_jump, double ik_timeout,
                                 double pos_tol, double ori_tol_deg, double min_end_l2,
                                 int max_contacts, int max_per_pair)
{
  PathResult out;
  out.a_index = a_index;
  out.pre_index = pre_index;
  out.han_index = han_index;

  const Eigen::Vector3d p0 = geo.world_pre_b.translation();
  const Eigen::Vector3d p1 = geo.world_tcp_b.translation();
  const double distance = (p1 - p0).norm();
  const int n = std::max(2, 1 + static_cast<int>(std::llround(distance / cartesian_step)));
  std::vector<double> s_grid;
  s_grid.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    s_grid.push_back(static_cast<double>(i) / static_cast<double>(n - 1));
  s_grid.back() = 1.0;

  std::vector<double> seed_b = b_pre.joints;
  size_t i = 0;
  while (i < s_grid.size())
  {
    if (out.samples.size() > 500)
    {
      out.fail_stage = "densify";
      out.fail_reason = "too many densified Cartesian samples";
      out.ik_discontinuity = true;
      return out;
    }
    const double s = s_grid[i];
    const bool is_start = out.samples.empty() && s <= 1e-12;
    PathSample sample =
        solveCartesianSample(scene, geo, gripper_a, gripper_b, a_cand.joints, seed_b, s, !is_start,
                             q_a, q_b, ik_timeout, pos_tol, ori_tol_deg, max_contacts, max_per_pair);
    sample.index = static_cast<int>(out.samples.size());
    if (!sample.fail_reason.empty())
    {
      out.fail_reason = sample.fail_reason;
      out.truncated = sample.fail_reason.find("INCOMPLETE") != std::string::npos;
      out.collision_fail = sample.fail_reason.find("collision") != std::string::npos ||
                           sample.diag.collision || sample.diag.has_b_finger_part_contact;
      out.b_finger_object_during_approach = sample.diag.has_b_finger_part_contact;
      out.fail_stage = is_start ? "start sample" : "sample " + std::to_string(sample.index);
      out.samples.push_back(sample);
      return out;
    }

    if (!out.samples.empty())
    {
      const PathSample& prev = out.samples.back();
      const double jump = maxAbsJointDelta(prev.joints_b, sample.joints_b);
      const double cartesian_gap = (sample.tcp - prev.tcp).norm();
      const double s_gap = s - prev.s;
      if (jump > max_joint_jump && s_gap * distance > densify_min + 1e-12)
      {
        s_grid.insert(s_grid.begin() + static_cast<std::ptrdiff_t>(i), 0.5 * (prev.s + s));
        continue;
      }
      if (jump > max_joint_jump)
      {
        out.ik_discontinuity = true;
        out.fail_stage = "IK branch continuity";
        out.fail_reason = "Arm B IK branch discontinuity at s=" + fmtScalar(s) +
                          " max_|dq|=" + fmtScalar(jump) +
                          " rad cartesian_gap=" + fmtScalar(cartesian_gap) + " m";
        sample.max_abs_joint_step = jump;
        sample.joint_step = jointL2(prev.joints_b, sample.joints_b);
        out.samples.push_back(sample);
        return out;
      }
      sample.max_abs_joint_step = jump;
      sample.joint_step = jointL2(prev.joints_b, sample.joints_b);
      if (sample.tcp.x() + 1e-6 < prev.tcp.x())
        out.x_monotonic = false;
      out.max_cartesian_gap = std::max(out.max_cartesian_gap, cartesian_gap);
      out.max_joint_gap = std::max(out.max_joint_gap, sample.joint_step);
      out.max_adjacent_joint_change = std::max(out.max_adjacent_joint_change, jump);
      out.arm_b_path_length += sample.joint_step;
    }

    if (std::abs(sample.tcp.y() - p0.y()) > pos_tol || std::abs(sample.tcp.z() - p0.z()) > pos_tol)
      out.yz_constant = false;
    if (sample.ori_error_deg > ori_tol_deg)
      out.orientation_constant = false;
    if (jointL2(sample.joints_a, a_cand.joints) > 1e-9)
      out.arm_a_unchanged = false;
    out.max_line_error = std::max(out.max_line_error, sample.line_error);
    out.max_ori_error = std::max(out.max_ori_error, sample.ori_error_deg);
    out.samples.push_back(sample);
    seed_b = sample.joints_b;
    ++i;
  }

  if (out.samples.empty())
  {
    out.fail_stage = "sampling";
    out.fail_reason = "no Cartesian samples generated";
    return out;
  }
  if (!out.x_monotonic)
  {
    out.fail_stage = "straight-line constraint";
    out.fail_reason = "Arm B TCP X did not increase monotonically";
    return out;
  }
  if (!out.yz_constant)
  {
    out.fail_stage = "straight-line constraint";
    out.fail_reason = "Arm B TCP Y or Z left the requested line";
    return out;
  }
  if (!out.orientation_constant)
  {
    out.fail_stage = "straight-line constraint";
    out.fail_reason = "Arm B TCP orientation was not constant";
    return out;
  }
  if (!out.arm_a_unchanged)
  {
    out.fail_stage = "Arm A lock";
    out.fail_reason = "Arm A joints changed along the path";
    return out;
  }

  out.start_tcp = out.samples.front().tcp;
  out.end_tcp = out.samples.back().tcp;
  out.end_joints_b = out.samples.back().joints_b;
  out.cartesian_ok = true;
  out.end_to_k_l2 = jointL2(out.end_joints_b, b_han.joints);
  if (out.end_to_k_l2 > min_end_l2)
  {
    out.end_mismatch = true;
    out.fail_stage = "endpoint IK branch";
    out.fail_reason = "final Arm B joints do not match selected B_handover" +
                      std::to_string(han_index) + " L2=" + fmtScalar(out.end_to_k_l2);
    return out;
  }
  out.matched_han_index = static_cast<int>(han_index);
  out.pass = true;
  return out;
}

void printPathFailure(const PathResult& path)
{
  emit("  A" + std::to_string(path.a_index) + " + B_pre" + std::to_string(path.pre_index) +
       " + B_handover" + std::to_string(path.han_index) + ": FAIL");
  emit("    stage: " + path.fail_stage);
  emit("    reason: " + path.fail_reason);
  if (path.samples.empty())
    return;
  const PathSample& sample = path.samples.back();
  emit("    sample: " + std::to_string(sample.index));
  emit("    s: " + fmtScalar(sample.s));
  emit("    TCP xyz: " + fmtXyz(sample.tcp));
  emit("    Arm B joints_rad: " + fmtVec(sample.joints_b));
  emit("    collision category: " + collisionCategory(sample.diag));
  if (!sample.diag.pairs.empty())
  {
    emit("    collision pairs:");
    for (const auto& pair : sample.diag.pairs)
      emit("      " + pair);
  }
}

PathResult withHandoverMatch(PathResult tracked, const IkCandidate& b_han, size_t han_index,
                             double min_end_l2)
{
  tracked.han_index = han_index;
  if (!tracked.cartesian_ok)
    return tracked;
  tracked.end_to_k_l2 = jointL2(tracked.end_joints_b, b_han.joints);
  if (tracked.end_to_k_l2 > min_end_l2)
  {
    tracked.pass = false;
    tracked.end_mismatch = true;
    tracked.matched_han_index = -1;
    tracked.fail_stage = "endpoint IK branch";
    tracked.fail_reason = "final Arm B joints do not match selected B_handover" +
                          std::to_string(han_index) + " L2=" + fmtScalar(tracked.end_to_k_l2);
    return tracked;
  }
  tracked.pass = true;
  tracked.end_mismatch = false;
  tracked.matched_han_index = static_cast<int>(han_index);
  tracked.fail_stage.clear();
  tracked.fail_reason.clear();
  return tracked;
}

[[maybe_unused]] void printPairList(const std::string& title, const std::vector<std::string>& pairs)
{
  emit("      " + title + ": " + (pairs.empty() ? std::string("(none)") : std::string("")));
  for (const auto& pair : pairs)
    emit("        " + pair);
}

[[maybe_unused]] void printDiag(const std::string& title, const CollisionDiag& diag)
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

[[maybe_unused]] void printComboDetail(const ComboResult& item)
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

[[maybe_unused]] void printCombos(const std::string& heading, const std::vector<ComboResult>& results)
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

collision_detection::AllowedCollisionMatrix makePhaseAcm(
    const collision_detection::AllowedCollisionMatrix& live, const std::string& object_id,
    bool allow_b_fingers)
{
  collision_detection::AllowedCollisionMatrix acm = live;
  acm.setEntry("arm_a_finger_l", "arm_a_finger_r", false);
  acm.setEntry("arm_b_finger_l", "arm_b_finger_r", false);
  if (allow_b_fingers)
    acm.setEntry(object_id, kArmBFingerLinks, true);
  return acm;
}

bool hasIllegalCollision(const CollisionDiag& diag)
{
  if (diag.truncated || diag.has_finger_self || diag.has_upperarm_column)
    return true;
  if (!diag.gripper_interference.empty() || !diag.cross_arm.empty() || !diag.robot_env.empty() ||
      !diag.object_illegal.empty() || !diag.arm_a_self.empty() || !diag.arm_b_self.empty())
    return true;
  return diag.collision;
}

void fingerContactFlags(const CollisionDiag& diag, const std::string& object_id, bool& a_l,
                        bool& a_r, bool& b_l, bool& b_r)
{
  const auto hit = [&](const std::vector<std::string>& pairs, const std::string& link) {
    const std::string key = pairKey(object_id, link);
    return inList(pairs, key) || inList(diag.pairs, key);
  };
  a_l = hit(diag.expected_a_touch, "arm_a_finger_l");
  a_r = hit(diag.expected_a_touch, "arm_a_finger_r");
  b_l = hit(diag.expected_b_finger, "arm_b_finger_l");
  b_r = hit(diag.expected_b_finger, "arm_b_finger_r");
}

std::vector<double> gripperGrid(double start, double end, double step)
{
  std::vector<double> qs;
  const double span = std::abs(end - start);
  if (span < 1e-15)
  {
    qs.push_back(start);
    return qs;
  }
  const double use_step = std::max(1e-9, std::min(step, span));
  const int n = std::max(1, static_cast<int>(std::llround(span / use_step)));
  qs.reserve(static_cast<size_t>(n + 1));
  for (int i = 0; i <= n; ++i)
  {
    const double t = static_cast<double>(i) / static_cast<double>(n);
    qs.push_back(start + t * (end - start));
  }
  qs.front() = start;
  qs.back() = end;
  return qs;
}

bool applyFixedArmsAndGrippers(moveit::core::RobotState& state, const GripperModelInfo& gripper_a,
                               const GripperModelInfo& gripper_b, const std::vector<double>& joints_a,
                               const std::vector<double>& joints_b, double q_a, double q_b,
                               std::string& error)
{
  applyGroupJoints(state, kArmAJoints, joints_a);
  applyGroupJoints(state, kArmBJoints, joints_b);
  if (!setActiveGripperJoint(state, gripper_a, q_a, error) ||
      !setActiveGripperJoint(state, gripper_b, q_b, error))
  {
    return false;
  }
  applyGroupJoints(state, kArmAJoints, joints_a);
  applyGroupJoints(state, kArmBJoints, joints_b);
  state.update();
  return true;
}

struct GripperSample
{
  int index = 0;
  double q_a = 0.0;
  double q_b = 0.0;
  double slider_la = 0.0;
  double slider_ra = 0.0;
  double slider_lb = 0.0;
  double slider_rb = 0.0;
  bool joints_ok = false;
  bool mimic_ok = false;
  bool tcp_ok = false;
  bool object_ok = false;
  bool illegal = false;
  bool truncated = false;
  bool a_l = false;
  bool a_r = false;
  bool b_l = false;
  bool b_r = false;
  bool finger_self = false;
  bool inter_gripper = false;
  bool robot_world = false;
  double object_pos_error = 0.0;
  double object_ori_error_deg = 0.0;
  CollisionDiag expected;
  CollisionDiag contact;
  std::string fail_reason;
};

struct GripperPhaseResult
{
  std::string name;
  double q_a_start = 0.0;
  double q_a_end = 0.0;
  double q_b_start = 0.0;
  double q_b_end = 0.0;
  int samples_checked = 0;
  double max_adjacent_q = 0.0;
  bool illegal = false;
  bool truncated = false;
  bool object_pose_ok = true;
  bool six_joints_fixed = true;
  bool tcp_fixed = true;
  int first_b_contact_index = -1;
  int first_b_bilateral_index = -1;
  int first_illegal_index = -1;
  std::string first_illegal_pair;
  std::string first_illegal_category;
  bool end_b_l = false;
  bool end_b_r = false;
  bool end_a_l = false;
  bool end_a_r = false;
  std::vector<GripperSample> samples;
};

GripperSample evaluateGripperSample(planning_scene::PlanningScene& scene,
                                    planning_scene::PlanningScene& contact_scene,
                                    const HandoverGeometry& geo, const GripperModelInfo& gripper_a,
                                    const GripperModelInfo& gripper_b,
                                    const std::vector<double>& joints_a,
                                    const std::vector<double>& joints_b, double q_a, double q_b,
                                    bool allow_b_fingers, bool verify_relative_a, double pos_tol,
                                    double ori_tol_deg, int max_contacts, int max_per_pair)
{
  GripperSample out;
  out.q_a = q_a;
  out.q_b = q_b;

  moveit::core::RobotState state(scene.getCurrentState());
  if (!applyFixedArmsAndGrippers(state, gripper_a, gripper_b, joints_a, joints_b, q_a, q_b,
                                 out.fail_reason))
  {
    out.illegal = true;
    return out;
  }
  out.mimic_ok = true;
  out.slider_la = jointOrNan(state, gripper_a.slider_l);
  out.slider_ra = jointOrNan(state, gripper_a.slider_r);
  out.slider_lb = jointOrNan(state, gripper_b.slider_l);
  out.slider_rb = jointOrNan(state, gripper_b.slider_r);

  if (!allVariablesFinite(state, out.fail_reason))
  {
    out.illegal = true;
    return out;
  }
  const auto* group_a = state.getJointModelGroup(kGroupA);
  const auto* group_b = state.getJointModelGroup(kGroupB);
  if (!state.satisfiesBounds(group_a) || !state.satisfiesBounds(group_b))
  {
    out.fail_reason = "combined state outside joint bounds";
    out.illegal = true;
    return out;
  }
  if (jointL2(jointsOf(state, kArmAJoints), joints_a) > 1e-9 ||
      jointL2(jointsOf(state, kArmBJoints), joints_b) > 1e-9)
  {
    out.fail_reason = "six-axis joints changed while sampling gripper";
    out.illegal = true;
    return out;
  }
  out.joints_ok = true;

  double a_pos = 0.0, a_ori = 0.0, b_pos = 0.0, b_ori = 0.0;
  poseError(geo.world_tcp_a, state.getGlobalLinkTransform(kTcpA), a_pos, a_ori);
  poseError(geo.world_tcp_b, state.getGlobalLinkTransform(kTcpB), b_pos, b_ori);
  if (a_pos > pos_tol || a_ori > ori_tol_deg || b_pos > pos_tol || b_ori > ori_tol_deg)
  {
    out.fail_reason = "TCP moved during gripper sampling";
    out.illegal = true;
    return out;
  }
  out.tcp_ok = true;

  if (verify_relative_a)
  {
    if (!verifyObjectPose(state, geo, pos_tol, ori_tol_deg, out.object_pos_error,
                          out.object_ori_error_deg, out.fail_reason))
    {
      out.illegal = true;
      return out;
    }
  }
  else
  {
    Eigen::Isometry3d world = Eigen::Isometry3d::Identity();
    if (!getAttachedWorldPose(state, geo.object_name, world, out.fail_reason))
    {
      out.illegal = true;
      return out;
    }
    poseError(geo.world_object, world, out.object_pos_error, out.object_ori_error_deg);
    if (out.object_pos_error > pos_tol || out.object_ori_error_deg > ori_tol_deg)
    {
      out.fail_reason = "object world pose jumped";
      out.illegal = true;
      return out;
    }
  }
  out.object_ok = true;

  const auto expected_acm =
      makePhaseAcm(scene.getAllowedCollisionMatrix(), geo.object_name, allow_b_fingers);
  out.expected = checkState(scene, state, expected_acm, max_contacts, max_per_pair, geo.object_name);
  out.truncated = out.expected.truncated;
  out.illegal = hasIllegalCollision(out.expected);
  out.finger_self = out.expected.has_finger_self;
  out.inter_gripper = !out.expected.gripper_interference.empty();
  out.robot_world = !out.expected.robot_env.empty();
  if (out.illegal && out.fail_reason.empty())
  {
    out.fail_reason = "illegal collision category=" + collisionCategory(out.expected);
    if (!out.expected.pairs.empty())
      out.fail_reason += " pair=" + out.expected.pairs.front();
  }

  auto& probe_state = contact_scene.getCurrentStateNonConst();
  std::string probe_error;
  if (applyFixedArmsAndGrippers(probe_state, gripper_a, gripper_b, joints_a, joints_b, q_a, q_b,
                                probe_error))
  {
    const auto contact_acm =
        makePhaseAcm(contact_scene.getAllowedCollisionMatrix(), geo.object_name, false);
    out.contact =
        checkState(contact_scene, probe_state, contact_acm, max_contacts, max_per_pair, geo.object_name);
    out.truncated = out.truncated || out.contact.truncated;
    fingerContactFlags(out.contact, geo.object_name, out.a_l, out.a_r, out.b_l, out.b_r);
    out.finger_self = out.finger_self || out.contact.has_finger_self;
  }
  else if (out.fail_reason.empty())
  {
    out.fail_reason = probe_error;
    out.illegal = true;
  }
  return out;
}

GripperPhaseResult runGripperPhase(planning_scene::PlanningScene& scene,
                                   planning_scene::PlanningScene& contact_scene,
                                   const HandoverGeometry& geo, const GripperModelInfo& gripper_a,
                                   const GripperModelInfo& gripper_b,
                                   const std::vector<double>& joints_a,
                                   const std::vector<double>& joints_b, double q_a_start,
                                   double q_a_end, double q_b_start, double q_b_end, double step,
                                   bool vary_b, bool allow_b_fingers, bool verify_relative_a,
                                   double pos_tol, double ori_tol_deg, int max_contacts,
                                   int max_per_pair, const std::string& name)
{
  GripperPhaseResult phase;
  phase.name = name;
  phase.q_a_start = q_a_start;
  phase.q_a_end = q_a_end;
  phase.q_b_start = q_b_start;
  phase.q_b_end = q_b_end;
  const auto qs = gripperGrid(vary_b ? q_b_start : q_a_start, vary_b ? q_b_end : q_a_end, step);
  for (size_t i = 0; i < qs.size(); ++i)
  {
    const double q_a = vary_b ? q_a_start : qs[i];
    const double q_b = vary_b ? qs[i] : q_b_start;
    GripperSample sample =
        evaluateGripperSample(scene, contact_scene, geo, gripper_a, gripper_b, joints_a, joints_b,
                              q_a, q_b, allow_b_fingers, verify_relative_a, pos_tol, ori_tol_deg,
                              max_contacts, max_per_pair);
    sample.index = static_cast<int>(i);
    if (i > 0)
    {
      const double dq = std::abs((vary_b ? sample.q_b : sample.q_a) -
                                 (vary_b ? phase.samples.back().q_b : phase.samples.back().q_a));
      phase.max_adjacent_q = std::max(phase.max_adjacent_q, dq);
    }
    if (!sample.joints_ok)
      phase.six_joints_fixed = false;
    if (!sample.tcp_ok)
      phase.tcp_fixed = false;
    if (!sample.object_ok)
      phase.object_pose_ok = false;
    if (sample.truncated)
      phase.truncated = true;
    if (phase.first_b_contact_index < 0 && (sample.b_l || sample.b_r))
      phase.first_b_contact_index = sample.index;
    if (phase.first_b_bilateral_index < 0 && sample.b_l && sample.b_r)
      phase.first_b_bilateral_index = sample.index;
    if (sample.illegal && phase.first_illegal_index < 0)
    {
      phase.first_illegal_index = sample.index;
      phase.first_illegal_category = collisionCategory(sample.expected);
      phase.first_illegal_pair =
          sample.expected.pairs.empty() ? sample.fail_reason : sample.expected.pairs.front();
      phase.illegal = true;
    }
    phase.samples.push_back(std::move(sample));
  }
  phase.samples_checked = static_cast<int>(phase.samples.size());
  if (!phase.samples.empty())
  {
    phase.end_a_l = phase.samples.back().a_l;
    phase.end_a_r = phase.samples.back().a_r;
    phase.end_b_l = phase.samples.back().b_l;
    phase.end_b_r = phase.samples.back().b_r;
  }
  return phase;
}

[[maybe_unused]] void printGripperSample(const GripperSample& sample, const std::string& phase)
{
  emit("  [" + phase + " i=" + std::to_string(sample.index) + "] q_A=" + fmtScalar(sample.q_a) +
       " q_B=" + fmtScalar(sample.q_b));
  emit("    sliders A l/r: " + fmtVec({sample.slider_la, sample.slider_ra}) +
       "  B l/r: " + fmtVec({sample.slider_lb, sample.slider_rb}));
  emit("    finger-finger: " + std::string(sample.finger_self ? "YES" : "NO"));
  emit("    A finger-object: L=" + std::string(sample.a_l ? "YES" : "NO") +
       " R=" + std::string(sample.a_r ? "YES" : "NO"));
  emit("    B finger-object: L=" + std::string(sample.b_l ? "YES" : "NO") +
       " R=" + std::string(sample.b_r ? "YES" : "NO"));
  emit("    inter-gripper: " + std::string(sample.inter_gripper ? "YES" : "NO"));
  emit("    robot-world: " + std::string(sample.robot_world ? "YES" : "NO"));
  emit("    illegal: " + std::string(sample.illegal ? "YES" : "NO"));
  if (sample.truncated)
    emit("    CONTACT ENUMERATION INCOMPLETE");
  if (!sample.expected.pairs.empty())
  {
    emit("    expected-mode pairs:");
    for (const auto& pair : sample.expected.pairs)
      emit("      " + pair);
  }
  if (!sample.fail_reason.empty() && sample.illegal)
    emit("    reason: " + sample.fail_reason);
}

[[maybe_unused]] void printPhase(const GripperPhaseResult& phase)
{
  emit("");
  emit("----------------------------------------");
  emit(phase.name);
  emit("----------------------------------------");
  emit("q_A:");
  emit("  " + fmtScalar(phase.q_a_start) +
       (std::abs(phase.q_a_end - phase.q_a_start) > 1e-12 ? " -> " + fmtScalar(phase.q_a_end) : ""));
  emit("q_B:");
  emit("  " + fmtScalar(phase.q_b_start) +
       (std::abs(phase.q_b_end - phase.q_b_start) > 1e-12 ? " -> " + fmtScalar(phase.q_b_end) : ""));
  emit("Samples checked:");
  emit("  " + std::to_string(phase.samples_checked));
  emit("Maximum adjacent q change:");
  emit("  " + fmtScalar(phase.max_adjacent_q));
  emit("Illegal collision:");
  emit(std::string("  ") + (phase.illegal ? "YES" : "NO"));
  emit("Object world pose unchanged:");
  emit(std::string("  ") + (phase.object_pose_ok ? "YES" : "NO"));
  emit("Six-axis joints fixed:");
  emit(std::string("  ") + (phase.six_joints_fixed ? "YES" : "NO"));
  emit("TCP unchanged:");
  emit(std::string("  ") + (phase.tcp_fixed ? "YES" : "NO"));
  emit("First expected finger-object contact:");
  if (phase.first_b_contact_index < 0)
    emit("  none observed");
  else
  {
    const auto& s = phase.samples[static_cast<size_t>(phase.first_b_contact_index)];
    emit("  sample " + std::to_string(phase.first_b_contact_index) + " q_B=" + fmtScalar(s.q_b) +
         " B_L=" + std::string(s.b_l ? "YES" : "NO") + " B_R=" + std::string(s.b_r ? "YES" : "NO"));
  }
  emit("First illegal collision, if any:");
  if (phase.first_illegal_index < 0)
    emit("  none");
  else
  {
    const auto& s = phase.samples[static_cast<size_t>(phase.first_illegal_index)];
    emit("  phase: " + phase.name);
    emit("  sample index: " + std::to_string(phase.first_illegal_index));
    emit("  q_A: " + fmtScalar(s.q_a));
    emit("  q_B: " + fmtScalar(s.q_b));
    emit("  link pair: " + phase.first_illegal_pair);
    emit("  collision category: " + phase.first_illegal_category);
  }
  emit("Finger-finger collision:");
  emit(std::string("  ") + ([&]() {
         for (const auto& s : phase.samples)
           if (s.finger_self)
             return "YES";
         return "NO";
       }()));
  emit("Inter-gripper collision:");
  emit(std::string("  ") + ([&]() {
         for (const auto& s : phase.samples)
           if (s.inter_gripper)
             return "YES";
         return "NO";
       }()));
  emit("Robot-world collision:");
  emit(std::string("  ") + ([&]() {
         for (const auto& s : phase.samples)
           if (s.robot_world)
             return "YES";
         return "NO";
       }()));
  emit("B finger-object contact:");
  emit("  L=" + std::string(phase.end_b_l ? "YES" : "NO") +
       " R=" + std::string(phase.end_b_r ? "YES" : "NO") +
       (phase.end_b_l && phase.end_b_r ? "  bilateral at end" : "  not bilateral at end"));
  emit("A finger-object contact at end:");
  emit("  L=" + std::string(phase.end_a_l ? "YES" : "NO") +
       " R=" + std::string(phase.end_a_r ? "YES" : "NO"));
  if (phase.truncated)
    emit("CONTACT ENUMERATION INCOMPLETE");
  for (const auto& sample : phase.samples)
    printGripperSample(sample, phase.name);
}

planning_scene::PlanningScenePtr makeEmptyTouchProbe(const planning_scene::PlanningScenePtr& src,
                                                     const HandoverGeometry& geo,
                                                     const std::string& link,
                                                     const Eigen::Isometry3d& pose_in_link,
                                                     std::string& error)
{
  auto probe = planning_scene::PlanningScene::clone(src);
  if (!attachObjectToLink(*probe, geo, link, pose_in_link, {}, error))
    return nullptr;
  return probe;
}

struct TransferCheck
{
  bool attempted = false;
  bool ok = false;
  bool duplicate = false;
  bool global_write = false;
  double world_pos_error = 0.0;
  double world_ori_error_deg = 0.0;
  std::string before_link;
  std::string after_link;
  std::string fail_reason;
};

TransferCheck transferAttachmentLocal(planning_scene::PlanningScene& scene,
                                      const HandoverGeometry& geo, const GripperModelInfo& gripper_a,
                                      const GripperModelInfo& gripper_b,
                                      const std::vector<double>& joints_a,
                                      const std::vector<double>& joints_b, double q_a, double q_b,
                                      double pos_tol, double ori_tol_deg)
{
  TransferCheck out;
  out.attempted = true;
  out.before_link = kTcpA;
  auto& state = scene.getCurrentStateNonConst();
  if (!applyFixedArmsAndGrippers(state, gripper_a, gripper_b, joints_a, joints_b, q_a, q_b,
                                 out.fail_reason))
  {
    return out;
  }
  if (!state.hasAttachedBody(geo.object_name))
  {
    out.fail_reason = "object not attached to Arm A before transfer";
    return out;
  }
  const auto* before_body = state.getAttachedBody(geo.object_name);
  if (!before_body || before_body->getAttachedLinkName() != kTcpA)
  {
    out.fail_reason = "pre-transfer attach link is not arm_a_gripper_tcp";
    return out;
  }

  Eigen::Isometry3d world_before = Eigen::Isometry3d::Identity();
  if (!getAttachedWorldPose(state, geo.object_name, world_before, out.fail_reason))
    return out;

  const Eigen::Isometry3d T_world_tcpB = state.getGlobalLinkTransform(kTcpB);
  const Eigen::Isometry3d T_tcpB_object = T_world_tcpB.inverse() * world_before;

  if (!attachObjectToLink(scene, geo, kTcpB, T_tcpB_object, kArmBFingerLinks, out.fail_reason))
    return out;
  scene.getCurrentStateNonConst().update();

  out.after_link = kTcpB;
  const int attached_n = countNamedAttachments(scene.getCurrentState(), geo.object_name);
  out.duplicate = attached_n != 1 || scene.getWorld()->hasObject(geo.object_name);
  if (out.duplicate)
  {
    out.fail_reason = "duplicate object after transfer attached=" + std::to_string(attached_n) +
                      " world=" + std::string(scene.getWorld()->hasObject(geo.object_name) ? "yes" : "no");
    return out;
  }
  const auto* after_body = scene.getCurrentState().getAttachedBody(geo.object_name);
  if (!after_body || after_body->getAttachedLinkName() != kTcpB)
  {
    out.fail_reason = "post-transfer attach link is not arm_b_gripper_tcp";
    return out;
  }

  Eigen::Isometry3d world_after = Eigen::Isometry3d::Identity();
  if (!getAttachedWorldPose(scene.getCurrentState(), geo.object_name, world_after, out.fail_reason))
    return out;
  poseError(world_before, world_after, out.world_pos_error, out.world_ori_error_deg);
  if (out.world_pos_error > pos_tol || out.world_ori_error_deg > ori_tol_deg)
  {
    out.fail_reason = "object world pose jumped during attachment transfer";
    return out;
  }
  double vs_geo_pos = 0.0;
  double vs_geo_ori = 0.0;
  poseError(geo.world_object, world_after, vs_geo_pos, vs_geo_ori);
  if (vs_geo_pos > pos_tol || vs_geo_ori > ori_tol_deg)
  {
    out.fail_reason = "post-transfer world pose disagrees with DUAL-4A object pose";
    out.world_pos_error = std::max(out.world_pos_error, vs_geo_pos);
    out.world_ori_error_deg = std::max(out.world_ori_error_deg, vs_geo_ori);
    return out;
  }
  out.ok = true;
  return out;
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
using MoveGroup = moveit::planning_interface::MoveGroupInterface;

std::string moveitErrorName(int val)
{
  switch (val)
  {
    case moveit_msgs::msg::MoveItErrorCodes::SUCCESS:
      return "SUCCESS";
    case moveit_msgs::msg::MoveItErrorCodes::FAILURE:
      return "FAILURE";
    case moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED:
      return "PLANNING_FAILED";
    case moveit_msgs::msg::MoveItErrorCodes::INVALID_MOTION_PLAN:
      return "INVALID_MOTION_PLAN";
    case moveit_msgs::msg::MoveItErrorCodes::MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE:
      return "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE";
    case moveit_msgs::msg::MoveItErrorCodes::CONTROL_FAILED:
      return "CONTROL_FAILED";
    case moveit_msgs::msg::MoveItErrorCodes::UNABLE_TO_AQUIRE_SENSOR_DATA:
      return "UNABLE_TO_AQUIRE_SENSOR_DATA";
    case moveit_msgs::msg::MoveItErrorCodes::TIMED_OUT:
      return "TIMED_OUT";
    case moveit_msgs::msg::MoveItErrorCodes::PREEMPTED:
      return "PREEMPTED";
    case moveit_msgs::msg::MoveItErrorCodes::START_STATE_IN_COLLISION:
      return "START_STATE_IN_COLLISION";
    case moveit_msgs::msg::MoveItErrorCodes::START_STATE_VIOLATES_PATH_CONSTRAINTS:
      return "START_STATE_VIOLATES_PATH_CONSTRAINTS";
    case moveit_msgs::msg::MoveItErrorCodes::GOAL_IN_COLLISION:
      return "GOAL_IN_COLLISION";
    case moveit_msgs::msg::MoveItErrorCodes::GOAL_VIOLATES_PATH_CONSTRAINTS:
      return "GOAL_VIOLATES_PATH_CONSTRAINTS";
    case moveit_msgs::msg::MoveItErrorCodes::GOAL_CONSTRAINTS_VIOLATED:
      return "GOAL_CONSTRAINTS_VIOLATED";
    case moveit_msgs::msg::MoveItErrorCodes::INVALID_GROUP_NAME:
      return "INVALID_GROUP_NAME";
    case moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS:
      return "INVALID_GOAL_CONSTRAINTS";
    case moveit_msgs::msg::MoveItErrorCodes::INVALID_ROBOT_STATE:
      return "INVALID_ROBOT_STATE";
    case moveit_msgs::msg::MoveItErrorCodes::INVALID_LINK_NAME:
      return "INVALID_LINK_NAME";
    case moveit_msgs::msg::MoveItErrorCodes::INVALID_OBJECT_NAME:
      return "INVALID_OBJECT_NAME";
    case moveit_msgs::msg::MoveItErrorCodes::FRAME_TRANSFORM_FAILURE:
      return "FRAME_TRANSFORM_FAILURE";
    case moveit_msgs::msg::MoveItErrorCodes::COLLISION_CHECKING_UNAVAILABLE:
      return "COLLISION_CHECKING_UNAVAILABLE";
    case moveit_msgs::msg::MoveItErrorCodes::ROBOT_STATE_STALE:
      return "ROBOT_STATE_STALE";
    case moveit_msgs::msg::MoveItErrorCodes::SENSOR_INFO_STALE:
      return "SENSOR_INFO_STALE";
    case moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION:
      return "NO_IK_SOLUTION";
    default:
      return "UNKNOWN(" + std::to_string(val) + ")";
  }
}

std::string collisionCategory4e(const CollisionDiag& diag)
{
  if (!diag.expected_a_touch.empty() || diag.has_a_finger_part_contact)
    return "Arm A <-> small_part";
  for (const auto& pair : diag.object_illegal)
  {
    if (pair.find("arm_a_") != std::string::npos)
      return "Arm A <-> small_part";
  }
  if (!diag.cross_arm.empty() || !diag.gripper_interference.empty())
    return "Arm A <-> Arm B";
  if (diag.has_upperarm_column || !diag.arm_column.empty())
    return "Arm A <-> mounting_column";
  if (!diag.arm_table.empty())
    return "Arm A <-> table";
  if (!diag.arm_a_self.empty())
    return "Arm A self-collision";
  if (!diag.arm_b_self.empty())
    return "Arm B self-collision";
  if (diag.has_finger_self)
    return "finger self-collision";
  if (!diag.object_illegal.empty())
    return "object illegal contact";
  if (!diag.robot_env.empty())
    return "robot <-> environment";
  if (!diag.pairs.empty())
    return "other";
  return "none";
}

bool hasIllegalCollisionReturnHome(const CollisionDiag& diag)
{
  if (hasIllegalCollision(diag))
    return true;
  if (!diag.expected_a_touch.empty() || diag.has_a_finger_part_contact)
    return true;
  return false;
}

bool loadYamlHomeJoints(const std::string& path, std::vector<double>& joints, std::string& error)
{
  YAML::Node yaml;
  try
  {
    yaml = YAML::LoadFile(path);
  }
  catch (const std::exception& e)
  {
    error = std::string("Home YAML unreadable: ") + e.what() + " (" + path + ")";
    return false;
  }
  YAML::Node values = yaml["home_joints_rad"];
  if (!values || !values.IsSequence() || values.size() != 6)
  {
    error = "Home YAML missing home_joints_rad[6]: " + path;
    return false;
  }
  joints.clear();
  for (int i = 0; i < 6; ++i)
  {
    const double q = values[i].as<double>();
    if (!std::isfinite(q))
    {
      error = "Home YAML has a non-finite joint: " + path;
      return false;
    }
    joints.push_back(q);
  }
  return true;
}

bool loadExistingHome(const std::string& step14_path, const std::string& step15_path,
                      std::vector<double>& joints, std::string& source, std::string& error)
{
  std::vector<double> home14;
  std::vector<double> home15;
  if (!loadYamlHomeJoints(step14_path, home14, error))
    return false;
  std::string step15_error;
  const bool have15 = loadYamlHomeJoints(step15_path, home15, step15_error);
  if (have15 && jointL2(home14, home15) > 1e-9)
  {
    error = "HOME DEFINITION AMBIGUOUS: step14 and step15 home_joints_rad differ";
    source.clear();
    return false;
  }
  joints = home14;
  source = step14_path + " (home_joints_rad)";
  source += "; identical to STEP12c/STEP15 winner Home used by dual_arm_a_prefix_plan";
  return true;
}

double jointPathLength(const std::vector<std::vector<double>>& waypoints)
{
  double length = 0.0;
  for (size_t i = 1; i < waypoints.size(); ++i)
    length += jointL2(waypoints[i - 1], waypoints[i]);
  return length;
}

double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b)
{
  const size_t n = std::min(a.size(), b.size());
  double m = 0.0;
  for (size_t i = 0; i < n; ++i)
    m = std::max(m, std::abs(a[i] - b[i]));
  if (a.size() != b.size())
    return std::numeric_limits<double>::infinity();
  return m;
}

std::map<std::string, double> jointMapFrom(const std::vector<std::string>& names,
                                           const std::vector<double>& values)
{
  std::map<std::string, double> out;
  const size_t n = std::min(names.size(), values.size());
  for (size_t i = 0; i < n; ++i)
    out[names[i]] = values[i];
  return out;
}

bool extractTrajectoryJoints(const trajectory_msgs::msg::JointTrajectory& traj, size_t index,
                             const std::vector<std::string>& wanted, std::vector<double>& out,
                             std::string& error)
{
  if (index >= traj.points.size())
  {
    error = "trajectory index out of range";
    return false;
  }
  const auto& point = traj.points[index];
  out.assign(wanted.size(), std::numeric_limits<double>::quiet_NaN());
  for (size_t w = 0; w < wanted.size(); ++w)
  {
    bool found = false;
    for (size_t n = 0; n < traj.joint_names.size() && n < point.positions.size(); ++n)
    {
      if (traj.joint_names[n] == wanted[w])
      {
        out[w] = point.positions[n];
        found = true;
        break;
      }
    }
    if (!found)
    {
      error = "trajectory missing joint " + wanted[w];
      return false;
    }
  }
  return true;
}

struct Dual4dEndCandidate
{
  size_t a_index = 0;
  size_t pre_index = 0;
  size_t han_index = 0;
  std::vector<double> joints_a;
  std::vector<double> joints_b;
  std::string label;
  bool transfer_ok = false;
  std::string fail_stage;
  std::string fail_reason;
};

struct StateCheck
{
  bool ok = false;
  bool attached_to_b = false;
  bool bounds_ok = false;
  bool arm_b_fixed = true;
  bool gripper_ok = true;
  bool object_pose_ok = true;
  bool collision_ok = true;
  double object_pos_error = 0.0;
  double object_ori_error_deg = 0.0;
  double q_a = 0.0;
  double q_b = 0.0;
  CollisionDiag diag;
  std::string fail_reason;
  std::string collision_category;
  std::string collision_pair;
};

struct PlanAttempt
{
  int index = 0;
  std::string planner;
  double planning_time_sec = 0.0;
  bool plan_success = false;
  bool validated = false;
  int error_code = 0;
  std::string error_name;
  size_t waypoint_count = 0;
  size_t validation_samples = 0;
  double joint_path_length = 0.0;
  double home_error = std::numeric_limits<double>::infinity();
  bool arm_b_moved = false;
  bool object_pose_changed = false;
  bool illegal_collision = false;
  std::string fail_reason;
  std::string collision_category;
  std::string collision_pair;
  int fail_waypoint = -1;
  int fail_interp = -1;
  std::vector<double> fail_joints_a;
  trajectory_msgs::msg::JointTrajectory trajectory;
  bool planning_skipped = false;
};

struct RequestAttachmentDiag
{
  bool local_checked = false;
  int local_count = 0;
  std::vector<std::string> local_ids;
  std::vector<std::string> local_links;
  std::string local_small_part_link;
  std::string local_geometry;
  std::vector<std::string> local_touch_links;
  bool local_has_small_part = false;

  bool request_constructed = false;
  int request_count = 0;
  std::string object_id;
  std::string attached_link;
  std::string object_frame;
  std::string geometry;
  std::vector<std::string> touch_links;
  bool is_diff = false;
  std::string group_name;
  std::string planner_id;
  std::vector<double> start_a;
  std::vector<double> start_b;
  bool start_a_complete = false;
  bool start_b_complete = false;
  bool arm_b_hold = false;
  bool geometry_ok = false;
  bool touch_ok = false;
  bool pass = false;
  std::string fail_reason;
  std::vector<std::string> acm_lines;
};

struct ReturnHomeResult
{
  Dual4dEndCandidate candidate;
  bool start_valid = false;
  bool goal_valid = false;
  bool home_state_invalid = false;
  std::string start_reason;
  std::string goal_reason;
  int attempts = 0;
  int successful_attempts = 0;
  PlanAttempt selected;
  bool retained = false;
  std::vector<PlanAttempt> attempt_log;
  RequestAttachmentDiag request_attachment;
};

planning_scene::PlanningScenePtr makeTransferredScene(
    const planning_scene::PlanningScenePtr& local, const HandoverGeometry& geo,
    const GripperModelInfo& gripper_a, const GripperModelInfo& gripper_b,
    const std::vector<double>& joints_a, const std::vector<double>& joints_b, double q_a,
    double q_b, double pos_tol, double ori_tol, std::string& error)
{
  auto work = planning_scene::PlanningScene::clone(local);
  TransferCheck xfer = transferAttachmentLocal(*work, geo, gripper_a, gripper_b, joints_a, joints_b,
                                               q_a, q_b, pos_tol, ori_tol);
  if (!xfer.ok)
  {
    error = xfer.fail_reason;
    return nullptr;
  }
  return work;
}

StateCheck evaluateReturnHomeState(planning_scene::PlanningScene& scene, const HandoverGeometry& geo,
                                   const GripperModelInfo& gripper_a, const GripperModelInfo& gripper_b,
                                   const std::vector<double>& joints_a,
                                   const std::vector<double>& joints_b_fixed, double q_a_open,
                                   double q_b_hold, double pos_tol, double ori_tol, int max_contacts,
                                   int max_per_pair)
{
  StateCheck out;
  moveit::core::RobotState& state = scene.getCurrentStateNonConst();
  if (!applyFixedArmsAndGrippers(state, gripper_a, gripper_b, joints_a, joints_b_fixed, q_a_open,
                                 q_b_hold, out.fail_reason))
  {
    out.collision_ok = false;
    return out;
  }
  out.q_a = jointOrNan(state, kGripperJointA);
  out.q_b = jointOrNan(state, kGripperJointB);
  if (!allVariablesFinite(state, out.fail_reason))
    return out;

  const auto* group_a = state.getJointModelGroup(kGroupA);
  const auto* group_b = state.getJointModelGroup(kGroupB);
  out.bounds_ok = group_a && group_b && state.satisfiesBounds(group_a) && state.satisfiesBounds(group_b);
  if (!out.bounds_ok)
  {
    out.fail_reason = "combined state outside joint bounds";
    return out;
  }

  if (jointL2(jointsOf(state, kArmBJoints), joints_b_fixed) > 1e-9)
  {
    out.arm_b_fixed = false;
    out.fail_reason = "Arm B six-axis joints changed";
    return out;
  }
  if (!nearlyEqual(out.q_a, q_a_open, 1e-6) || !nearlyEqual(out.q_b, q_b_hold, 1e-6))
  {
    out.gripper_ok = false;
    out.fail_reason = "gripper opening changed q_A=" + fmtScalar(out.q_a) +
                      " q_B=" + fmtScalar(out.q_b);
    return out;
  }

  if (!state.hasAttachedBody(geo.object_name))
  {
    out.fail_reason = "small_part not attached";
    return out;
  }
  const auto* body = state.getAttachedBody(geo.object_name);
  if (!body || body->getAttachedLinkName() != kTcpB)
  {
    out.fail_reason = "small_part not attached to arm_b_gripper_tcp";
    return out;
  }
  if (countNamedAttachments(state, geo.object_name) != 1 || scene.getWorld()->hasObject(geo.object_name))
  {
    out.fail_reason = "duplicate or world copy of small_part";
    return out;
  }
  out.attached_to_b = true;

  Eigen::Isometry3d world = Eigen::Isometry3d::Identity();
  if (!getAttachedWorldPose(state, geo.object_name, world, out.fail_reason))
    return out;
  poseError(geo.world_object, world, out.object_pos_error, out.object_ori_error_deg);
  if (out.object_pos_error > pos_tol || out.object_ori_error_deg > ori_tol)
  {
    out.object_pose_ok = false;
    out.fail_reason = "object world pose jumped pos=" + fmtScalar(out.object_pos_error) +
                      " m ori=" + fmtScalar(out.object_ori_error_deg) + " deg";
    return out;
  }

  out.diag = checkState(scene, state, scene.getAllowedCollisionMatrix(), max_contacts, max_per_pair,
                        geo.object_name);
  out.collision_category = collisionCategory4e(out.diag);
  out.collision_pair = out.diag.pairs.empty() ? std::string() : out.diag.pairs.front();
  if (hasIllegalCollisionReturnHome(out.diag))
  {
    out.collision_ok = false;
    out.fail_reason = "illegal collision category=" + out.collision_category;
    if (!out.collision_pair.empty())
      out.fail_reason += " pair=" + out.collision_pair;
    return out;
  }
  out.ok = true;
  return out;
}

bool validateInterpolatedPath(planning_scene::PlanningScene& scene, const HandoverGeometry& geo,
                              const GripperModelInfo& gripper_a, const GripperModelInfo& gripper_b,
                              const std::vector<std::vector<double>>& waypoints_a,
                              const std::vector<double>& joints_b_fixed, double q_a_open,
                              double q_b_hold, double max_joint_step, double pos_tol, double ori_tol,
                              int max_contacts, int max_per_pair, PlanAttempt& attempt)
{
  if (waypoints_a.empty())
  {
    attempt.fail_reason = "empty Arm A waypoint list";
    return false;
  }
  size_t samples = 0;
  auto check_one = [&](int waypoint, int interp, const std::vector<double>& q_a) {
    StateCheck st =
        evaluateReturnHomeState(scene, geo, gripper_a, gripper_b, q_a, joints_b_fixed, q_a_open,
                                q_b_hold, pos_tol, ori_tol, max_contacts, max_per_pair);
    ++samples;
    if (!st.arm_b_fixed)
      attempt.arm_b_moved = true;
    if (!st.object_pose_ok)
      attempt.object_pose_changed = true;
    if (!st.ok)
    {
      attempt.illegal_collision = !st.collision_ok;
      attempt.fail_reason = st.fail_reason;
      attempt.collision_category = st.collision_category;
      attempt.collision_pair = st.collision_pair;
      attempt.fail_waypoint = waypoint;
      attempt.fail_interp = interp;
      attempt.fail_joints_a = q_a;
      return false;
    }
    return true;
  };

  for (size_t i = 0; i < waypoints_a.size(); ++i)
  {
    if (!check_one(static_cast<int>(i), 0, waypoints_a[i]))
    {
      attempt.validation_samples = samples;
      return false;
    }
    if (i + 1 >= waypoints_a.size())
      continue;
    const double jump = maxAbsJointDelta(waypoints_a[i], waypoints_a[i + 1]);
    const int n = std::max(1, static_cast<int>(std::ceil(jump / std::max(1e-9, max_joint_step))));
    for (int s = 1; s < n; ++s)
    {
      const double t = static_cast<double>(s) / static_cast<double>(n);
      std::vector<double> q(waypoints_a[i].size(), 0.0);
      for (size_t j = 0; j < q.size(); ++j)
        q[j] = waypoints_a[i][j] + t * (waypoints_a[i + 1][j] - waypoints_a[i][j]);
      if (!check_one(static_cast<int>(i), s, q))
      {
        attempt.validation_samples = samples;
        return false;
      }
    }
  }
  attempt.validation_samples = samples;
  return true;
}

std::string describeShape(const shapes::ShapeConstPtr& shape)
{
  if (!shape)
    return "null shape";
  switch (shape->type)
  {
    case shapes::CYLINDER:
    {
      const auto* cyl = dynamic_cast<const shapes::Cylinder*>(shape.get());
      if (!cyl)
        return "CYLINDER (cast failed)";
      return "CYLINDER radius=" + fmtScalar(cyl->radius) + " m length=" + fmtScalar(cyl->length) +
             " m";
    }
    case shapes::SPHERE:
    {
      const auto* sph = dynamic_cast<const shapes::Sphere*>(shape.get());
      if (!sph)
        return "SPHERE (cast failed)";
      return "SPHERE radius=" + fmtScalar(sph->radius) + " m";
    }
    case shapes::BOX:
    {
      const auto* box = dynamic_cast<const shapes::Box*>(shape.get());
      if (!box)
        return "BOX (cast failed)";
      return "BOX x=" + fmtScalar(box->size[0]) + " y=" + fmtScalar(box->size[1]) +
             " z=" + fmtScalar(box->size[2]) + " m";
    }
    case shapes::MESH:
    {
      const auto* mesh = dynamic_cast<const shapes::Mesh*>(shape.get());
      if (!mesh)
        return "MESH (cast failed)";
      return "MESH triangles=" + std::to_string(mesh->triangle_count) +
             " vertices=" + std::to_string(mesh->vertex_count);
    }
    default:
      return "shape type=" + std::to_string(static_cast<int>(shape->type));
  }
}

std::string describePrimitive(const shape_msgs::msg::SolidPrimitive& prim)
{
  if (prim.type == shape_msgs::msg::SolidPrimitive::CYLINDER && prim.dimensions.size() >= 2)
  {
    const double height = prim.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT];
    const double radius = prim.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS];
    return "CYLINDER height=dimensions[" +
           std::to_string(shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT) +
           "]=" + fmtScalar(height) + " m radius=dimensions[" +
           std::to_string(shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS) +
           "]=" + fmtScalar(radius) + " m";
  }
  if (prim.type == shape_msgs::msg::SolidPrimitive::BOX && prim.dimensions.size() >= 3)
  {
    return "BOX x=" + fmtScalar(prim.dimensions[0]) + " y=" + fmtScalar(prim.dimensions[1]) +
           " z=" + fmtScalar(prim.dimensions[2]) + " m";
  }
  if (prim.type == shape_msgs::msg::SolidPrimitive::SPHERE && !prim.dimensions.empty())
    return "SPHERE radius=" + fmtScalar(prim.dimensions[0]) + " m";
  std::ostringstream oss;
  oss << "type=" << static_cast<int>(prim.type) << " dims=[";
  for (size_t i = 0; i < prim.dimensions.size(); ++i)
  {
    if (i)
      oss << ", ";
    oss << fmtScalar(prim.dimensions[i]);
  }
  oss << "]";
  return oss.str();
}

std::string describeAttachedGeometryMsg(const moveit_msgs::msg::AttachedCollisionObject& attached)
{
  const auto& obj = attached.object;
  std::ostringstream oss;
  oss << "header.frame_id=" << (obj.header.frame_id.empty() ? std::string("(empty)") : obj.header.frame_id);
  oss << " object.pose " << poseLine(obj.pose);
  oss << " primitives=" << obj.primitives.size();
  oss << " meshes=" << obj.meshes.size();
  oss << " planes=" << obj.planes.size();
  for (size_t i = 0; i < obj.primitives.size(); ++i)
  {
    oss << " | prim[" << i << "] " << describePrimitive(obj.primitives[i]);
    if (i < obj.primitive_poses.size())
      oss << " pose " << poseLine(obj.primitive_poses[i]);
  }
  for (size_t i = 0; i < obj.meshes.size(); ++i)
  {
    oss << " | mesh[" << i << "] triangles=" << obj.meshes[i].triangles.size()
        << " vertices=" << obj.meshes[i].vertices.size();
    if (i < obj.mesh_poses.size())
      oss << " pose " << poseLine(obj.mesh_poses[i]);
  }
  return oss.str();
}

bool cylinderMatchesExpected(const shape_msgs::msg::SolidPrimitive& prim, double radius,
                             double length)
{
  if (prim.type != shape_msgs::msg::SolidPrimitive::CYLINDER || prim.dimensions.size() < 2)
    return false;
  const double height = prim.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT];
  const double r = prim.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS];
  return nearlyEqual(r, radius, 1e-9) && nearlyEqual(height, length, 1e-9);
}

std::vector<double> jointsFromJointState(const sensor_msgs::msg::JointState& js,
                                         const std::vector<std::string>& names, bool& all_found)
{
  all_found = true;
  std::vector<double> values(names.size(), std::numeric_limits<double>::quiet_NaN());
  for (size_t i = 0; i < names.size(); ++i)
  {
    bool found = false;
    for (size_t j = 0; j < js.name.size() && j < js.position.size(); ++j)
    {
      if (js.name[j] == names[i])
      {
        values[i] = js.position[j];
        found = true;
        break;
      }
    }
    if (!found)
      all_found = false;
  }
  return values;
}

std::vector<std::string> acmPairsFor(const collision_detection::AllowedCollisionMatrix& acm,
                                     const std::string& object_id)
{
  std::vector<std::string> names;
  acm.getAllEntryNames(names);
  std::vector<std::string> out;
  if (!acm.hasEntry(object_id))
  {
    out.push_back(object_id + " ABSENT from ACM names");
    return out;
  }
  for (const auto& other : names)
  {
    if (other == object_id)
      continue;
    if (!acm.hasEntry(object_id, other))
      continue;
    out.push_back(object_id + " <-> " + other + ": " + acmEntryStatus(acm, object_id, other));
  }
  if (out.empty())
    out.push_back(object_id + " present in ACM names but has no pairwise entries");
  return out;
}

void fillLocalAttachedDiag(const moveit::core::RobotState& start_state, const HandoverGeometry& geo,
                           RequestAttachmentDiag& diag)
{
  std::vector<const moveit::core::AttachedBody*> bodies;
  start_state.getAttachedBodies(bodies);
  diag.local_checked = true;
  diag.local_count = static_cast<int>(bodies.size());
  for (const auto* body : bodies)
  {
    if (!body)
      continue;
    diag.local_ids.push_back(body->getName());
    diag.local_links.push_back(body->getAttachedLinkName());
    if (body->getName() != geo.object_name)
      continue;
    diag.local_has_small_part = true;
    diag.local_small_part_link = body->getAttachedLinkName();
    diag.local_touch_links.assign(body->getTouchLinks().begin(), body->getTouchLinks().end());
    std::ostringstream geo_line;
    geo_line << "shapes=" << body->getShapes().size();
    for (size_t i = 0; i < body->getShapes().size(); ++i)
      geo_line << " | [" << i << "] " << describeShape(body->getShapes()[i]);
    if (!body->getShapePosesInLinkFrame().empty())
      geo_line << " | pose_in_link " << poseLine(toPoseMsg(body->getShapePosesInLinkFrame().front()));
    diag.local_geometry = geo_line.str();
  }
}

void fillRequestAttachmentDiag(const moveit_msgs::msg::MotionPlanRequest& request,
                               const HandoverGeometry& geo, const std::vector<double>& joints_b_hold,
                               RequestAttachmentDiag& diag)
{
  diag.request_constructed = true;
  diag.group_name = request.group_name;
  diag.planner_id = request.planner_id;
  diag.is_diff = request.start_state.is_diff;
  diag.request_count = static_cast<int>(request.start_state.attached_collision_objects.size());
  diag.start_a = jointsFromJointState(request.start_state.joint_state, kArmAJoints, diag.start_a_complete);
  diag.start_b = jointsFromJointState(request.start_state.joint_state, kArmBJoints, diag.start_b_complete);
  diag.arm_b_hold = diag.start_b_complete && maxAbsDiff(diag.start_b, joints_b_hold) <= 1e-6;

  const moveit_msgs::msg::AttachedCollisionObject* part = nullptr;
  for (const auto& attached : request.start_state.attached_collision_objects)
  {
    if (attached.object.id == geo.object_name)
    {
      part = &attached;
      break;
    }
  }
  if (!part)
  {
    diag.fail_reason = "small_part missing from MotionPlanRequest.start_state";
    diag.pass = false;
    return;
  }

  diag.object_id = part->object.id;
  diag.attached_link = part->link_name;
  diag.object_frame = part->object.header.frame_id;
  diag.touch_links = part->touch_links;
  diag.geometry = describeAttachedGeometryMsg(*part);

  if (diag.attached_link != kTcpB)
  {
    diag.fail_reason = "small_part attached link is " + diag.attached_link +
                       " not arm_b_gripper_tcp";
    diag.pass = false;
    return;
  }

  std::set<std::string> got(diag.touch_links.begin(), diag.touch_links.end());
  const std::set<std::string> expect = {kArmBFingerLinks[0], kArmBFingerLinks[1]};
  diag.touch_ok = (got == expect);
  if (!diag.touch_ok)
  {
    for (const auto& link : diag.touch_links)
    {
      if (link.rfind("arm_a_", 0) == 0)
      {
        diag.fail_reason = "touch_links contain arm_a_* link: " + link;
        diag.pass = false;
        return;
      }
    }
    diag.fail_reason = "touch_links are not exactly arm_b_finger_l, arm_b_finger_r";
    diag.pass = false;
    return;
  }

  if (!part->object.primitives.empty())
  {
    for (const auto& prim : part->object.primitives)
    {
      if (cylinderMatchesExpected(prim, geo.radius, geo.length))
      {
        diag.geometry_ok = true;
        break;
      }
    }
    if (!diag.geometry_ok)
    {
      diag.fail_reason =
          "MotionPlanRequest small_part primitives do not include expected cylinder "
          "radius=0.0075 m length=0.035 m";
      diag.pass = false;
      return;
    }
  }
  else if (!part->object.meshes.empty())
  {
    diag.geometry_ok = true;
  }
  else
  {
    diag.fail_reason = "small_part has neither primitive nor mesh geometry in the request";
    diag.pass = false;
    return;
  }

  if (diag.request_count != 1)
  {
    diag.fail_reason = "unexpected attached object count=" + std::to_string(diag.request_count);
    diag.pass = false;
    return;
  }
  diag.pass = true;
}

void fillLocalAcmNotes(const planning_scene::PlanningScene& pre_transfer,
                       const planning_scene::PlanningScene& transferred, const HandoverGeometry& geo,
                       RequestAttachmentDiag& diag)
{
  const auto& pre = pre_transfer.getAllowedCollisionMatrix();
  const auto& post = transferred.getAllowedCollisionMatrix();
  diag.acm_lines.push_back(
      "MotionPlanRequest has no ACM field; setStartState sends RobotState only.");
  diag.acm_lines.push_back("Live /move_group ACM is not replaced by this test.");
  diag.acm_lines.push_back("arm_a_base_link <-> mounting_column local transferred ACM: " +
                           acmEntryStatus(post, "arm_a_base_link", kColumnName));
  diag.acm_lines.push_back("arm_b_base_link <-> mounting_column local transferred ACM: " +
                           acmEntryStatus(post, "arm_b_base_link", kColumnName));
  diag.acm_lines.push_back(
      "arm_a_finger_l <-> arm_a_finger_r local transferred ACM: " +
      acmEntryStatus(post, "arm_a_finger_l", "arm_a_finger_r") +
      " (makePhaseAcm NEVER is a DUAL-4C/4D collision-copy only, not written here)");
  diag.acm_lines.push_back("Pre-transfer local ACM entries for " + geo.object_name + ":");
  for (const auto& line : acmPairsFor(pre, geo.object_name))
    diag.acm_lines.push_back("  " + line);
  diag.acm_lines.push_back("Post-transfer local ACM entries for " + geo.object_name + ":");
  for (const auto& line : acmPairsFor(post, geo.object_name))
    diag.acm_lines.push_back("  " + line);
  diag.acm_lines.push_back(
      "Those local ACM rows are not serialized by setStartState(); /move_group can recreate "
      "object<->touch_link allowances only from "
      "request.start_state.attached_collision_objects[].touch_links.");
}

void emitRequestAttachmentCheck(const RequestAttachmentDiag& diag, const std::string& global_attached,
                                const std::string& live_acm_a, const std::string& live_acm_b)
{
  emit("");
  emit("========== DUAL-4E REQUEST ATTACHMENT CHECK ==========");
  emit("");
  emit("Local RobotState attached objects:");
  emit("  " + std::to_string(diag.local_count));
  if (diag.local_ids.empty())
  {
    emit("  (none)");
  }
  else
  {
    for (size_t i = 0; i < diag.local_ids.size(); ++i)
    {
      emit("  " + diag.local_ids[i] + " @ " +
           (i < diag.local_links.size() ? diag.local_links[i] : std::string("?")));
    }
  }
  if (diag.local_has_small_part)
  {
    emit("  local small_part link: " + diag.local_small_part_link);
    emit("  local geometry: " + diag.local_geometry);
    emit("  local touch_links: " + joinNames(diag.local_touch_links));
  }
  emit("");
  emit("MotionPlanRequest attached objects:");
  emit("  " + (diag.request_constructed ? std::to_string(diag.request_count) : std::string("NOT CONSTRUCTED")));
  emit("");
  emit("Object ID:");
  emit("  " + (diag.object_id.empty() ? std::string("(none)") : diag.object_id));
  emit("");
  emit("Attached link:");
  emit("  " + (diag.attached_link.empty() ? std::string("(none)") : diag.attached_link));
  emit("");
  emit("Object frame:");
  emit("  " + (diag.object_frame.empty() ? std::string("(empty)") : diag.object_frame));
  emit("");
  emit("Object geometry:");
  emit("  " + (diag.geometry.empty() ? std::string("(none)") : diag.geometry));
  emit("");
  emit("Touch links:");
  emit("  " + joinNames(diag.touch_links));
  emit("");
  emit("start_state.is_diff:");
  emit("  " + std::string(diag.is_diff ? "true" : "false"));
  emit("");
  emit("MotionPlanRequest group:");
  emit("  " + (diag.group_name.empty() ? std::string("(empty)") : diag.group_name) +
       " planner_id=" + (diag.planner_id.empty() ? std::string("(empty)") : diag.planner_id));
  emit("");
  emit("MotionPlanRequest start Arm A:");
  emit("  " + (diag.start_a_complete ? fmtVec(diag.start_a) : std::string("MISSING JOINTS")));
  emit("");
  emit("MotionPlanRequest start Arm B:");
  emit("  " + (diag.start_b_complete ? fmtVec(diag.start_b) : std::string("MISSING JOINTS")));
  emit("");
  emit("Global /move_group attached objects:");
  emit("  " + global_attached);
  emit("");
  emit("Live /move_group ACM base-column:");
  emit("  arm_a_base_link <-> mounting_column: " + live_acm_a);
  emit("  arm_b_base_link <-> mounting_column: " + live_acm_b);
  emit("");
  emit("Local-only ACM differences:");
  if (diag.acm_lines.empty())
  {
    emit("  (not collected)");
  }
  else
  {
    for (const auto& line : diag.acm_lines)
      emit("  " + line);
  }
  emit("");
  emit("Constructed request:");
  emit("  " + std::string(diag.request_constructed ? "VERIFIED" : "NOT CONSTRUCTED"));
  emit("Actual wire-level request:");
  emit("  NOT DIRECTLY CAPTURED");
  emit("");
  emit("----------------------------------------");
  emit("FINAL");
  emit("----------------------------------------");
  emit("");
  emit("REQUEST ATTACHMENT:");
  std::string verdict = "INCOMPLETE";
  if (diag.request_constructed)
    verdict = diag.pass ? "PASS" : "FAIL";
  emit("  " + verdict);
  if (!diag.fail_reason.empty() && verdict != "PASS")
    emit("  Reason: " + diag.fail_reason);
  emit("");
  emit("Planning request contains small_part:");
  emit("  " + std::string((diag.request_constructed && diag.object_id == "small_part" &&
                           diag.attached_link == kTcpB) ?
                              "YES" :
                              "NO"));
  emit("");
  emit("Planning request contains Arm B hold state:");
  emit("  " + std::string(diag.arm_b_hold ? "YES" : "NO"));
  emit("");
  emit("Global PlanningScene modified:");
  emit("  NO");
  emit("");
  emit("Existing DUAL-4E planning logic changed:");
  emit("  NO");
  emit("");
  emit("Real robot commands:");
  emit("  ZERO");
  emit("");
  emit("Real gripper commands:");
  emit("  ZERO");
}

ReturnHomeResult planArmAReturnHome(MoveGroup& arm_a, const planning_scene::PlanningScenePtr& local,
                                    const HandoverGeometry& geo, const GripperModelInfo& gripper_a,
                                    const GripperModelInfo& gripper_b,
                                    const Dual4dEndCandidate& cand, const std::vector<double>& home,
                                    double q_a_open, double q_b_hold, int max_attempts,
                                    double planning_time, double max_joint_step, double pos_tol,
                                    double ori_tol, double home_tol, int max_contacts, int max_per_pair)
{
  ReturnHomeResult out;
  out.candidate = cand;
  std::string error;
  auto scene = makeTransferredScene(local, geo, gripper_a, gripper_b, cand.joints_a, cand.joints_b,
                                    q_a_open, q_b_hold, pos_tol, ori_tol, error);
  if (!scene)
  {
    out.start_reason = "cannot rebuild DUAL-4D end scene: " + error;
    return out;
  }

  StateCheck start = evaluateReturnHomeState(*scene, geo, gripper_a, gripper_b, cand.joints_a,
                                             cand.joints_b, q_a_open, q_b_hold, pos_tol, ori_tol,
                                             max_contacts, max_per_pair);
  out.start_valid = start.ok;
  out.start_reason = start.fail_reason;
  StateCheck goal = evaluateReturnHomeState(*scene, geo, gripper_a, gripper_b, home, cand.joints_b,
                                            q_a_open, q_b_hold, pos_tol, ori_tol, max_contacts,
                                            max_per_pair);
  out.goal_valid = goal.ok;
  out.goal_reason = goal.fail_reason;
  out.home_state_invalid = !goal.ok;
  if (!out.start_valid || !out.goal_valid)
    return out;

  moveit::core::RobotState start_state(scene->getCurrentState());
  applyFixedArmsAndGrippers(start_state, gripper_a, gripper_b, cand.joints_a, cand.joints_b,
                            q_a_open, q_b_hold, error);
  start_state.update();
  fillLocalAttachedDiag(start_state, geo, out.request_attachment);
  fillLocalAcmNotes(*local, *scene, geo, out.request_attachment);

  const auto home_map = jointMapFrom(kArmAJoints, home);
  arm_a.setPlanningTime(planning_time);
  arm_a.setNumPlanningAttempts(1);
  arm_a.setPlanningPipelineId("ompl");
  const std::string planner_id = arm_a.getPlannerId();
  const std::string pipeline_id =
      arm_a.getPlanningPipelineId().empty() ? std::string("ompl") : arm_a.getPlanningPipelineId();
  const std::string planner_label =
      pipeline_id + " / " + (planner_id.empty() ? std::string("default") : planner_id);

  for (int attempt = 1; attempt <= max_attempts; ++attempt)
  {
    PlanAttempt rec;
    rec.index = attempt;
    rec.planner = planner_label;
    out.attempts = attempt;

    arm_a.setStartState(start_state);
    if (!arm_a.setJointValueTarget(home_map))
    {
      rec.error_name = "INVALID_GOAL_CONSTRAINTS";
      rec.fail_reason = "setJointValueTarget(Home) failed";
      out.attempt_log.push_back(rec);
      continue;
    }

    moveit_msgs::msg::MotionPlanRequest request;
    arm_a.constructMotionPlanRequest(request);
    fillRequestAttachmentDiag(request, geo, cand.joints_b, out.request_attachment);
    emit("");
    emit("REQUEST ATTACHMENT CHECK:");
    emit("  " + std::string(out.request_attachment.pass ? "PASS" : "FAIL"));
    emit("  constructed attached objects: " +
         std::to_string(out.request_attachment.request_count));
    emit("  object ID: " +
         (out.request_attachment.object_id.empty() ? std::string("(none)") :
                                                     out.request_attachment.object_id));
    emit("  attached link: " +
         (out.request_attachment.attached_link.empty() ? std::string("(none)") :
                                                         out.request_attachment.attached_link));
    emit("  object frame: " +
         (out.request_attachment.object_frame.empty() ? std::string("(empty)") :
                                                        out.request_attachment.object_frame));
    emit("  geometry: " + (out.request_attachment.geometry.empty() ?
                               std::string("(none)") :
                               out.request_attachment.geometry));
    emit("  touch_links: " + joinNames(out.request_attachment.touch_links));
    emit("  start_state.is_diff: " + std::string(out.request_attachment.is_diff ? "true" : "false"));
    emit("  group: " + out.request_attachment.group_name +
         " planner_id=" + (out.request_attachment.planner_id.empty() ?
                               std::string("(empty)") :
                               out.request_attachment.planner_id));
    emit("  start Arm A: " +
         (out.request_attachment.start_a_complete ? fmtVec(out.request_attachment.start_a) :
                                                    std::string("MISSING JOINTS")));
    emit("  start Arm B: " +
         (out.request_attachment.start_b_complete ? fmtVec(out.request_attachment.start_b) :
                                                    std::string("MISSING JOINTS")));
    emit("  Constructed request: VERIFIED");
    emit("  Actual wire-level request: NOT DIRECTLY CAPTURED");
    if (!out.request_attachment.pass)
    {
      rec.planning_skipped = true;
      rec.error_name = "REQUEST_ATTACHMENT_CHECK_FAIL";
      rec.fail_reason = out.request_attachment.fail_reason.empty() ?
                            std::string("small_part missing from MotionPlanRequest.start_state") :
                            out.request_attachment.fail_reason;
      emit("  Reason:");
      emit("    " + rec.fail_reason);
      emit("  Planning:");
      emit("    SKIPPED");
      out.attempt_log.push_back(std::move(rec));
      break;
    }

    MoveGroup::Plan plan;
    const auto t0 = std::chrono::steady_clock::now();
    const auto code = arm_a.plan(plan);
    rec.planning_time_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    rec.error_code = code.val;
    rec.error_name = moveitErrorName(code.val);
    rec.plan_success = (code == moveit::core::MoveItErrorCode::SUCCESS);
    rec.trajectory = plan.trajectory_.joint_trajectory;
    rec.waypoint_count = rec.trajectory.points.size();
    if (!rec.plan_success)
    {
      rec.fail_reason = "planner error " + rec.error_name;
      out.attempt_log.push_back(std::move(rec));
      continue;
    }
    if (rec.waypoint_count < 2)
    {
      rec.fail_reason = "planned trajectory has too few waypoints";
      out.attempt_log.push_back(std::move(rec));
      continue;
    }

    bool unexpected_b = false;
    for (const auto& name : rec.trajectory.joint_names)
    {
      if (name.rfind("arm_b_", 0) == 0)
      {
        std::vector<double> qb;
        std::string extract_error;
        if (!extractTrajectoryJoints(rec.trajectory, 0, kArmBJoints, qb, extract_error))
        {
          unexpected_b = true;
          rec.fail_reason = "Arm B joints present but unreadable: " + extract_error;
          break;
        }
        for (size_t p = 0; p < rec.trajectory.points.size(); ++p)
        {
          std::vector<double> qbp;
          if (!extractTrajectoryJoints(rec.trajectory, p, kArmBJoints, qbp, extract_error) ||
              maxAbsDiff(qbp, cand.joints_b) > 1e-9)
          {
            unexpected_b = true;
            rec.arm_b_moved = true;
            rec.fail_reason = "Arm B joints changed in planned trajectory";
            break;
          }
        }
      }
    }
    if (unexpected_b)
    {
      out.attempt_log.push_back(std::move(rec));
      continue;
    }

    std::vector<std::vector<double>> waypoints_a;
    waypoints_a.reserve(rec.waypoint_count);
    bool extract_ok = true;
    for (size_t p = 0; p < rec.waypoint_count; ++p)
    {
      std::vector<double> qa;
      std::string extract_error;
      if (!extractTrajectoryJoints(rec.trajectory, p, kArmAJoints, qa, extract_error))
      {
        rec.fail_reason = extract_error;
        extract_ok = false;
        break;
      }
      waypoints_a.push_back(std::move(qa));
    }
    if (!extract_ok)
    {
      out.attempt_log.push_back(std::move(rec));
      continue;
    }
    rec.joint_path_length = jointPathLength(waypoints_a);
    rec.home_error = maxAbsDiff(waypoints_a.back(), home);
    if (rec.home_error > home_tol)
    {
      rec.fail_reason = "final Home error " + fmtScalar(rec.home_error, 8) + " rad > " +
                        fmtScalar(home_tol, 8);
      out.attempt_log.push_back(std::move(rec));
      continue;
    }

    auto check_scene = planning_scene::PlanningScene::clone(scene);
    if (!validateInterpolatedPath(*check_scene, geo, gripper_a, gripper_b, waypoints_a,
                                  cand.joints_b, q_a_open, q_b_hold, max_joint_step, pos_tol,
                                  ori_tol, max_contacts, max_per_pair, rec))
    {
      out.attempt_log.push_back(std::move(rec));
      continue;
    }
    rec.validated = true;
    ++out.successful_attempts;
    if (!out.retained || rec.joint_path_length < out.selected.joint_path_length)
      out.selected = rec;
    out.retained = true;
    out.attempt_log.push_back(std::move(rec));
    break;
  }
  return out;
}

bool runDual4dTransfer(const planning_scene::PlanningScenePtr& local, const HandoverGeometry& geo,
                       const GripperModelInfo& gripper_a, const GripperModelInfo& gripper_b,
                       const std::vector<double>& joints_a, const std::vector<double>& joints_b,
                       double q_a_grasp, double q_a_open, double q_b_open, double q_b_recv,
                       double gripper_step, double pos_tol, double ori_tol, int max_contacts,
                       int max_per_pair, Dual4dEndCandidate& out)
{
  auto work = planning_scene::PlanningScene::clone(local);
  std::string probe_error;
  auto probe_a = makeEmptyTouchProbe(work, geo, kTcpA, geo.tcp_a_object, probe_error);
  if (!probe_a)
  {
    out.fail_stage = "contact probe";
    out.fail_reason = probe_error;
    return false;
  }

  auto phaseFailed = [](const GripperPhaseResult& phase) {
    return phase.illegal || phase.truncated || !phase.object_pose_ok || !phase.six_joints_fixed ||
           !phase.tcp_fixed || phase.samples_checked < 2;
  };

  GripperPhaseResult p1 = runGripperPhase(*work, *probe_a, geo, gripper_a, gripper_b, joints_a,
                                          joints_b, q_a_grasp, q_a_grasp, q_b_open, q_b_recv,
                                          gripper_step, true, true, true, pos_tol, ori_tol,
                                          max_contacts, max_per_pair, "PHASE 1: B CLOSING");
  if (phaseFailed(p1))
  {
    out.fail_stage = p1.name;
    out.fail_reason = p1.first_illegal_pair.empty() ? "phase 1 failed" : p1.first_illegal_pair;
    return false;
  }
  GripperPhaseResult p2 = runGripperPhase(*work, *probe_a, geo, gripper_a, gripper_b, joints_a,
                                          joints_b, q_a_grasp, q_a_open, q_b_recv, q_b_recv,
                                          gripper_step, false, true, true, pos_tol, ori_tol,
                                          max_contacts, max_per_pair, "PHASE 2: A OPENING");
  if (phaseFailed(p2))
  {
    out.fail_stage = p2.name;
    out.fail_reason = p2.first_illegal_pair.empty() ? "phase 2 failed" : p2.first_illegal_pair;
    return false;
  }
  TransferCheck xfer = transferAttachmentLocal(*work, geo, gripper_a, gripper_b, joints_a, joints_b,
                                               q_a_open, q_b_recv, pos_tol, ori_tol);
  if (!xfer.ok)
  {
    out.fail_stage = "PHASE 3: LOCAL ATTACHMENT TRANSFER";
    out.fail_reason = xfer.fail_reason;
    return false;
  }
  const auto* body_b = work->getCurrentState().getAttachedBody(geo.object_name);
  Eigen::Isometry3d pose_in_b = Eigen::Isometry3d::Identity();
  if (body_b && !body_b->getShapePosesInLinkFrame().empty())
    pose_in_b = body_b->getShapePosesInLinkFrame().front();
  auto probe_b = makeEmptyTouchProbe(work, geo, kTcpB, pose_in_b, probe_error);
  if (!probe_b)
  {
    out.fail_stage = "PHASE 4: POST-TRANSFER COLLISION";
    out.fail_reason = probe_error;
    return false;
  }
  GripperPhaseResult p4 = runGripperPhase(*work, *probe_b, geo, gripper_a, gripper_b, joints_a,
                                          joints_b, q_a_open, q_a_open, q_b_recv, q_b_recv,
                                          gripper_step, false, false, false, pos_tol, ori_tol,
                                          max_contacts, max_per_pair, "PHASE 4: POST-TRANSFER COLLISION");
  const bool p4_fail = p4.illegal || p4.truncated || !p4.object_pose_ok || !p4.six_joints_fixed ||
                       !p4.tcp_fixed || p4.samples_checked < 1;
  if (p4_fail)
  {
    out.fail_stage = p4.name;
    out.fail_reason = p4.first_illegal_pair.empty() ? "phase 4 failed" : p4.first_illegal_pair;
    return false;
  }
  out.transfer_ok = true;
  return true;
}

std::string fmtIso(const Eigen::Isometry3d& T)
{
  return "xyz=" + fmtXyz(T.translation()) + " xyzw=" + fmtXyzw(Eigen::Quaterniond(T.rotation()));
}

double angleDeg(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
{
  const double na = a.norm();
  const double nb = b.norm();
  if (na < 1e-12 || nb < 1e-12)
    return 180.0;
  const double c = std::max(-1.0, std::min(1.0, a.dot(b) / (na * nb)));
  return std::acos(c) * 180.0 / M_PI;
}

bool yamlVec3(const YAML::Node& node, Eigen::Vector3d& v)
{
  if (!node || !node.IsSequence() || node.size() != 3)
    return false;
  v = Eigen::Vector3d(node[0].as<double>(), node[1].as<double>(), node[2].as<double>());
  return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

bool yamlQuatXyzw(const YAML::Node& node, Eigen::Quaterniond& q)
{
  if (!node || !node.IsSequence() || node.size() != 4)
    return false;
  q = Eigen::Quaterniond(node[3].as<double>(), node[0].as<double>(), node[1].as<double>(),
                         node[2].as<double>());
  if (!std::isfinite(q.x()) || !std::isfinite(q.y()) || !std::isfinite(q.z()) || !std::isfinite(q.w()) ||
      q.norm() < 1e-12)
    return false;
  q.normalize();
  return true;
}

Eigen::Isometry3d makeIso(const Eigen::Vector3d& p, const Eigen::Quaterniond& q)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = p;
  T.linear() = q.toRotationMatrix();
  return T;
}

struct InspectionTarget
{
  int index = 0;
  std::string name;
  std::string winner_key;
  std::string source;
  std::string view_source;
  std::string physical_face;
  double roll_deg = 0.0;
  Eigen::Isometry3d object_target = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d tcp_b_target = Eigen::Isometry3d::Identity();
  Eigen::Vector3d center_in_object = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal_in_object = Eigen::Vector3d::Zero();
  Eigen::Vector3d up_in_object = Eigen::Vector3d::Zero();
  Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
  Eigen::Vector3d d1 = Eigen::Vector3d::Zero();
};

struct InspectionIk
{
  int id = 0;
  std::string status = "IK_INVALID";
  IkCandidate ik;
  double object_pos_error = 0.0;
  double object_ori_error_deg = 0.0;
  double face_center_error = 0.0;
  double face_normal_error_deg = 0.0;
  CollisionDiag diag;
  std::string fail_reason;
};

struct HeldStateCheck
{
  bool ok = false;
  bool attached_to_b = false;
  bool bounds_ok = false;
  bool arm_a_fixed = true;
  bool gripper_ok = true;
  bool relative_attach_ok = true;
  bool object_target_ok = true;
  bool face_ok = true;
  bool collision_ok = true;
  double q_a = 0.0;
  double q_b = 0.0;
  double relative_pos_error = 0.0;
  double relative_ori_error_deg = 0.0;
  double object_pos_error = 0.0;
  double object_ori_error_deg = 0.0;
  double face_center_error = 0.0;
  double face_normal_error_deg = 0.0;
  CollisionDiag diag;
  std::string fail_reason;
  std::string collision_category;
  std::string collision_pair;
};

struct TimedPath
{
  bool ok = false;
  bool collision_validated = false;
  bool time_parameterized = false;
  bool within_velocity_limits = false;
  bool within_acceleration_limits = false;
  bool gratuitous_j6 = false;
  std::string fail_reason;
  PlanAttempt attempt;
  std::vector<std::vector<double>> waypoints;
  std::vector<double> times;
  std::vector<std::array<double, 6>> velocities;
  std::vector<std::array<double, 6>> accelerations;
  double duration = 0.0;
  double path_length = 0.0;
  std::array<double, 6> travel{};
  std::array<int, 6> reversals{};
  std::array<double, 6> net{};
  std::array<double, 6> peak_vel{};
  std::array<double, 6> peak_acc{};
};

struct SegmentPlan
{
  bool ok = false;
  PlanAttempt selected;
  RequestAttachmentDiag request_attachment;
  std::vector<double> start_b;
  std::vector<double> goal_b;
  std::vector<double> end_b;
  int attempts = 0;
  int successful_attempts = 0;
  int time_parameterized_attempts = 0;
  std::string fail_reason;
  std::vector<TimedPath> timed_paths;
};

struct ArmBJointLimit
{
  std::string name;
  bool velocity_bounded = false;
  double max_velocity = 0.0;
  bool acceleration_bounded = false;
  double max_acceleration = 0.0;
  bool position_bounded = false;
  double min_position = 0.0;
  double max_position = 0.0;
  bool continuous = false;
};

struct ArmBLimits
{
  std::vector<ArmBJointLimit> joints;
  bool velocity_complete = false;
  bool acceleration_complete = false;
  bool complete = false;
  std::string source;
  std::string missing;
};

struct ChainMetrics
{
  double t_total = 0.0;
  double t1 = 0.0;
  double t2 = 0.0;
  double t3 = 0.0;
  double path1 = 0.0;
  double path2 = 0.0;
  double path3 = 0.0;
  double path_total = 0.0;
  std::array<double, 6> travel{};
  std::array<int, 6> reversals{};
  std::array<double, 6> net{};
  std::array<double, 6> peak_vel{};
  std::array<double, 6> peak_acc{};
  double j6_share = 0.0;
  bool gratuitous_j6 = false;
  bool within_velocity_limits = false;
  bool within_acceleration_limits = false;
};

struct CompleteChain
{
  Dual4dEndCandidate handover;
  InspectionIk i1;
  InspectionIk i2;
  InspectionIk i3;
  SegmentPlan h_to_i1;
  SegmentPlan i1_to_i2;
  SegmentPlan i2_to_i3;
  TimedPath p1;
  TimedPath p2;
  TimedPath p3;
  ChainMetrics metrics;
  double total_length = 0.0;
  bool time_ok = false;
  bool seam_ok = true;
  int path_choice1 = 0;
  int path_choice2 = 0;
  int path_choice3 = 0;
};

struct CollisionObs
{
  bool part_a = false;
  bool part_table = false;
  bool part_column = false;
  bool a_b = false;
};

void accumulateCollisionObs(CollisionObs& obs, const CollisionDiag& diag)
{
  if (!diag.expected_a_touch.empty() || diag.has_a_finger_part_contact)
    obs.part_a = true;
  if (!diag.cross_arm.empty() || !diag.gripper_interference.empty())
    obs.a_b = true;
  auto mark = [&](const std::string& pair) {
    if (pair.find("arm_a_") != std::string::npos && pair.find("small_part") != std::string::npos)
      obs.part_a = true;
    if (pair.find("table") != std::string::npos && pair.find("small_part") != std::string::npos)
      obs.part_table = true;
    if (pair.find("mounting_column") != std::string::npos &&
        pair.find("small_part") != std::string::npos)
      obs.part_column = true;
  };
  for (const auto& pair : diag.object_illegal)
    mark(pair);
  for (const auto& pair : diag.pairs)
    mark(pair);
}

double wrapPi(double d)
{
  while (d > M_PI)
    d -= 2.0 * M_PI;
  while (d < -M_PI)
    d += 2.0 * M_PI;
  return d;
}

double jointDelta(double from, double to, bool continuous)
{
  const double raw = to - from;
  return continuous ? wrapPi(raw) : raw;
}

ArmBLimits readArmBLimits(const moveit::core::RobotModel& model)
{
  ArmBLimits out;
  out.source = "RobotModel VariableBounds (URDF + robot_description_planning/joint_limits overlay)";
  out.velocity_complete = true;
  out.acceleration_complete = true;
  std::ostringstream missing;
  for (const auto& name : kArmBJoints)
  {
    ArmBJointLimit row;
    row.name = name;
    const auto* joint = model.getJointModel(name);
    if (!joint || joint->getVariableBounds().empty())
    {
      out.velocity_complete = false;
      out.acceleration_complete = false;
      missing << name << " missing from RobotModel; ";
      out.joints.push_back(row);
      continue;
    }
    const auto& bounds = joint->getVariableBounds().front();
    row.velocity_bounded = bounds.velocity_bounded_;
    row.max_velocity = std::min(std::fabs(bounds.max_velocity_), std::fabs(bounds.min_velocity_));
    if (!(row.max_velocity > 0.0) && bounds.velocity_bounded_)
      row.max_velocity = std::fabs(bounds.max_velocity_);
    row.acceleration_bounded = bounds.acceleration_bounded_;
    row.max_acceleration =
        std::min(std::fabs(bounds.max_acceleration_), std::fabs(bounds.min_acceleration_));
    if (!(row.max_acceleration > 0.0) && bounds.acceleration_bounded_)
      row.max_acceleration = std::fabs(bounds.max_acceleration_);
    row.position_bounded = bounds.position_bounded_;
    row.min_position = bounds.min_position_;
    row.max_position = bounds.max_position_;
    if (joint->getType() == moveit::core::JointModel::REVOLUTE)
    {
      const auto* rev = dynamic_cast<const moveit::core::RevoluteJointModel*>(joint);
      row.continuous = rev && rev->isContinuous();
    }
    if (!row.velocity_bounded || !(row.max_velocity > 0.0))
    {
      out.velocity_complete = false;
      missing << name << " velocity limit missing; ";
    }
    if (!row.acceleration_bounded || !(row.max_acceleration > 0.0))
    {
      out.acceleration_complete = false;
      missing << name << " acceleration limit missing; ";
    }
    out.joints.push_back(row);
  }
  out.missing = missing.str();
  out.complete = out.velocity_complete && out.acceleration_complete && out.joints.size() == 6;
  return out;
}

void printArmBLimits(const ArmBLimits& limits)
{
  emit("");
  emit("----------------------------------------");
  emit("ARM B JOINT LIMITS (RobotModel)");
  emit("----------------------------------------");
  emit("Source:");
  emit("  " + limits.source);
  emit("Invented velocity/acceleration:");
  emit("  NO");
  for (const auto& j : limits.joints)
  {
    emit("");
    emit(j.name + ":");
    emit("  velocity bounded: " + std::string(j.velocity_bounded ? "YES" : "NO"));
    emit("  max velocity: " + fmtScalar(j.max_velocity) + " rad/s");
    emit("  acceleration bounded: " + std::string(j.acceleration_bounded ? "YES" : "NO"));
    emit("  max acceleration: " + fmtScalar(j.max_acceleration) + " rad/s^2");
    emit("  continuous: " + std::string(j.continuous ? "YES" : "NO"));
    if (j.position_bounded)
      emit("  position: [" + fmtScalar(j.min_position) + ", " + fmtScalar(j.max_position) + "]");
  }
  emit("");
  emit("Velocity limits complete:");
  emit(std::string("  ") + (limits.velocity_complete ? "YES" : "NO"));
  emit("Acceleration limits complete:");
  emit(std::string("  ") + (limits.acceleration_complete ? "YES" : "NO"));
}

bool jointIsContinuous(const ArmBLimits& limits, size_t index)
{
  return index < limits.joints.size() && limits.joints[index].continuous;
}

void accumulateJointMetrics(const std::vector<std::vector<double>>& waypoints, const ArmBLimits& limits,
                            double direction_eps, std::array<double, 6>& travel,
                            std::array<int, 6>& reversals, std::array<double, 6>& net)
{
  travel.fill(0.0);
  reversals.fill(0);
  net.fill(0.0);
  if (waypoints.size() < 2)
    return;
  std::array<int, 6> last_sign{};
  last_sign.fill(0);
  for (size_t i = 1; i < waypoints.size(); ++i)
  {
    if (waypoints[i].size() < 6 || waypoints[i - 1].size() < 6)
      continue;
    for (size_t j = 0; j < 6; ++j)
    {
      const double d = jointDelta(waypoints[i - 1][j], waypoints[i][j], jointIsContinuous(limits, j));
      travel[j] += std::fabs(d);
      net[j] += d;
      int sign = 0;
      if (d > direction_eps)
        sign = 1;
      else if (d < -direction_eps)
        sign = -1;
      if (sign != 0)
      {
        if (last_sign[j] != 0 && sign != last_sign[j])
          ++reversals[j];
        last_sign[j] = sign;
      }
    }
  }
}

bool detectGratuitousJ6(double j6_travel, double j6_net, int j6_reversals, bool reject)
{
  if (!reject)
    return false;
  const double extra = j6_travel - std::fabs(j6_net);
  const bool extra_full_turn = extra >= (2.0 * M_PI - 0.05);
  const bool empty_spin = extra_full_turn && j6_reversals >= 1;
  return extra_full_turn || empty_spin;
}

double j6Share(const std::array<double, 6>& travel)
{
  const double denom = travel[1] + travel[2] + travel[3] + travel[4] + travel[5];
  if (denom <= 1e-12)
    return 0.0;
  return travel[5] / denom;
}

bool extractArmBFromState(const moveit::core::RobotState& state, std::array<double, 6>& q,
                          std::array<double, 6>& v, std::array<double, 6>& a)
{
  for (size_t j = 0; j < 6; ++j)
  {
    const int idx = state.getRobotModel()->getVariableIndex(kArmBJoints[j]);
    q[j] = state.getVariablePosition(idx);
    v[j] = state.getVariableVelocity(idx);
    a[j] = state.getVariableAcceleration(idx);
  }
  return true;
}

bool timeParameterizeArmB(const moveit::core::RobotModelConstPtr& model,
                          const moveit::core::RobotState& ref_state,
                          const trajectory_msgs::msg::JointTrajectory& traj, const ArmBLimits& limits,
                          double direction_eps, double vel_slack, double acc_slack, TimedPath& out)
{
  out.waypoints.clear();
  out.times.clear();
  out.velocities.clear();
  out.accelerations.clear();
  if (!limits.complete)
  {
    out.fail_reason = "TIME OPTIMIZATION LIMITS INCOMPLETE";
    return false;
  }
  robot_trajectory::RobotTrajectory rt(model, kGroupB);
  moveit_msgs::msg::RobotTrajectory msg;
  msg.joint_trajectory = traj;
  rt.setRobotTrajectoryMsg(ref_state, msg);
  if (rt.getWayPointCount() < 2)
  {
    out.fail_reason = "RobotTrajectory has too few waypoints";
    return false;
  }

  std::unordered_map<std::string, double> vel_map;
  std::unordered_map<std::string, double> acc_map;
  for (const auto& j : limits.joints)
  {
    vel_map[j.name] = j.max_velocity;
    acc_map[j.name] = j.max_acceleration;
  }
  trajectory_processing::TimeOptimalTrajectoryGeneration totg(0.01, 0.01, 0.001);
  if (!totg.computeTimeStamps(rt, vel_map, acc_map))
  {
    out.fail_reason = "TimeOptimalTrajectoryGeneration failed";
    return false;
  }
  out.duration = rt.getDuration();
  if (!(out.duration > 0.0) || !std::isfinite(out.duration))
  {
    out.fail_reason = "non-positive TOTG duration";
    return false;
  }

  out.peak_vel.fill(0.0);
  out.peak_acc.fill(0.0);
  out.within_velocity_limits = true;
  out.within_acceleration_limits = true;
  for (std::size_t i = 0; i < rt.getWayPointCount(); ++i)
  {
    const auto& st = rt.getWayPoint(i);
    std::array<double, 6> q{};
    std::array<double, 6> v{};
    std::array<double, 6> a{};
    extractArmBFromState(st, q, v, a);
    out.waypoints.emplace_back(q.begin(), q.end());
    out.times.push_back(rt.getWayPointDurationFromStart(i));
    out.velocities.push_back(v);
    out.accelerations.push_back(a);
    for (size_t j = 0; j < 6; ++j)
    {
      out.peak_vel[j] = std::max(out.peak_vel[j], std::fabs(v[j]));
      out.peak_acc[j] = std::max(out.peak_acc[j], std::fabs(a[j]));
    }
  }
  for (size_t j = 0; j < 6 && j < limits.joints.size(); ++j)
  {
    if (out.peak_vel[j] > limits.joints[j].max_velocity * (1.0 + vel_slack) + 1e-6)
      out.within_velocity_limits = false;
    if (out.peak_acc[j] > limits.joints[j].max_acceleration * (1.0 + acc_slack) + 1e-6)
      out.within_acceleration_limits = false;
  }
  if (!out.within_velocity_limits || !out.within_acceleration_limits)
  {
    out.fail_reason = "time-parameterized trajectory violates RobotModel joint limits";
    return false;
  }
  out.path_length = jointPathLength(out.waypoints);
  accumulateJointMetrics(out.waypoints, limits, direction_eps, out.travel, out.reversals, out.net);
  out.gratuitous_j6 = detectGratuitousJ6(out.travel[5], out.net[5], out.reversals[5], true);
  out.time_parameterized = true;
  out.ok = true;
  return true;
}

ChainMetrics combineChainMetrics(const TimedPath& a, const TimedPath& b, const TimedPath& c,
                                 const ArmBLimits& limits, double direction_eps, bool reject_j6)
{
  ChainMetrics m;
  m.t1 = a.duration;
  m.t2 = b.duration;
  m.t3 = c.duration;
  m.t_total = m.t1 + m.t2 + m.t3;
  m.path1 = a.path_length;
  m.path2 = b.path_length;
  m.path3 = c.path_length;
  m.path_total = m.path1 + m.path2 + m.path3;
  std::vector<std::vector<double>> all = a.waypoints;
  auto append_skip_first = [&](const TimedPath& p) {
    size_t start = 0;
    if (!all.empty() && !p.waypoints.empty() && maxAbsDiff(all.back(), p.waypoints.front()) <= 1e-4)
      start = 1;
    for (size_t i = start; i < p.waypoints.size(); ++i)
      all.push_back(p.waypoints[i]);
  };
  append_skip_first(b);
  append_skip_first(c);
  accumulateJointMetrics(all, limits, direction_eps, m.travel, m.reversals, m.net);
  m.j6_share = j6Share(m.travel);
  m.gratuitous_j6 =
      detectGratuitousJ6(m.travel[5], m.net[5], m.reversals[5], reject_j6) || a.gratuitous_j6 ||
      b.gratuitous_j6 || c.gratuitous_j6;
  for (size_t j = 0; j < 6; ++j)
  {
    m.peak_vel[j] = std::max({a.peak_vel[j], b.peak_vel[j], c.peak_vel[j]});
    m.peak_acc[j] = std::max({a.peak_acc[j], b.peak_acc[j], c.peak_acc[j]});
  }
  m.within_velocity_limits =
      a.within_velocity_limits && b.within_velocity_limits && c.within_velocity_limits;
  m.within_acceleration_limits =
      a.within_acceleration_limits && b.within_acceleration_limits && c.within_acceleration_limits;
  return m;
}

bool sameObjectPose(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b, double pos_tol,
                    double ori_tol_deg)
{
  double pos = 0.0;
  double ori = 0.0;
  poseError(a, b, pos, ori);
  return pos <= pos_tol && ori <= ori_tol_deg;
}

bool loadInspectionTargets(const std::string& winner_path, const std::string& sequence_path,
                           const Eigen::Isometry3d& tcp_b_object, std::vector<InspectionTarget>& out,
                           InspectionTarget& previous_i3, std::string& order_source,
                           std::string& error)
{
  out.clear();
  previous_i3 = InspectionTarget();
  YAML::Node winner;
  YAML::Node seq;
  try
  {
    winner = YAML::LoadFile(winner_path);
  }
  catch (const std::exception& e)
  {
    error = std::string("STEP12C winner YAML unreadable: ") + e.what();
    return false;
  }
  try
  {
    seq = YAML::LoadFile(sequence_path);
  }
  catch (const std::exception& e)
  {
    error = std::string("DUAL-5 sequence YAML unreadable: ") + e.what();
    return false;
  }
  Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
  Eigen::Vector3d d1 = Eigen::Vector3d::Zero();
  if (!yamlVec3(winner["p1"], p1) || !yamlVec3(winner["surface_target_normal"], d1))
  {
    error = "STEP12C winner missing p1 or surface_target_normal";
    return false;
  }
  if (winner["C_bottom"] && winner["C_bottom"]["object_world"])
  {
    Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    if (yamlVec3(winner["C_bottom"]["object_world"]["xyz"], xyz) &&
        yamlQuatXyzw(winner["C_bottom"]["object_world"]["xyzw"], q))
    {
      previous_i3.name = "bottom_circle";
      previous_i3.winner_key = "C_bottom";
      previous_i3.physical_face = "ORIGINAL_BOTTOM_CIRCLE";
      previous_i3.object_target = makeIso(xyz, q);
      previous_i3.tcp_b_target = previous_i3.object_target * tcp_b_object.inverse();
      previous_i3.source = winner_path + ":C_bottom.object_world";
      previous_i3.p1 = p1;
      previous_i3.d1 = d1;
      previous_i3.center_in_object = Eigen::Vector3d(0.0, 0.0, -0.0175);
      previous_i3.normal_in_object = Eigen::Vector3d(0.0, 0.0, -1.0);
      previous_i3.roll_deg =
          winner["C_bottom"]["roll_deg"] ? winner["C_bottom"]["roll_deg"].as<double>() : 0.0;
    }
  }
  if (!seq["inspection_order"] || !seq["inspection_order"].IsSequence() ||
      seq["inspection_order"].size() != 3)
  {
    error = "INSPECTION ORDER AMBIGUOUS: sequence YAML inspection_order is not exactly 3 names";
    return false;
  }
  order_source = seq["inspection_order_source"] ?
                     seq["inspection_order_source"].as<std::string>() :
                     (sequence_path + ":inspection_order");
  const std::vector<std::string> expected = {"side_pos_y", "side_neg_y", "top_circle"};
  std::set<std::string> seen;
  for (int i = 0; i < 3; ++i)
  {
    InspectionTarget t;
    t.index = i + 1;
    t.name = seq["inspection_order"][i].as<std::string>();
    if (t.name != expected[static_cast<size_t>(i)])
    {
      error = "INSPECTION ORDER AMBIGUOUS: DUAL-5-R requires side_pos_y, side_neg_y, top_circle; got " +
              t.name + " at index " + std::to_string(i + 1);
      return false;
    }
    if (seen.count(t.name))
    {
      error = "INSPECTION ORDER AMBIGUOUS: duplicate view " + t.name;
      return false;
    }
    seen.insert(t.name);
    const YAML::Node view = seq["inspection_views"][t.name];
    if (!view)
    {
      error = "INSPECTION TARGETS AMBIGUOUS: sequence YAML missing view " + t.name;
      return false;
    }
    t.view_source = view["view_source"] ? view["view_source"].as<std::string>() :
                                          (sequence_path + ":inspection_views." + t.name);
    t.physical_face = view["physical_face"] ? view["physical_face"].as<std::string>() : t.name;
    t.source = view["source"] ? view["source"].as<std::string>() : t.view_source;
    if (!yamlVec3(view["center_in_object"], t.center_in_object) ||
        !yamlVec3(view["normal_in_object"], t.normal_in_object) ||
        !yamlVec3(view["up_in_object"], t.up_in_object))
    {
      error = "INSPECTION TARGETS AMBIGUOUS: incomplete object-frame view for " + t.name;
      return false;
    }
    t.p1 = p1;
    t.d1 = d1;
    Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    if (t.name == "top_circle")
    {
      if (std::abs(t.normal_in_object.z() - 1.0) > 1e-6)
      {
        error = "ORIGINAL TOP FACE AMBIGUOUS: top_circle normal_in_object is not object +Z";
        return false;
      }
      if (!view["object_world"] || !yamlVec3(view["object_world"]["xyz"], xyz) ||
          !yamlQuatXyzw(view["object_world"]["xyzw"], q))
      {
        error = "INSPECTION TARGETS AMBIGUOUS: top_circle missing object_world in sequence YAML";
        return false;
      }
      t.winner_key = view["winner_key"] ? view["winner_key"].as<std::string>() : std::string();
      if (t.winner_key.empty())
        t.winner_key = "(not STEP12C winner; ARM2-future top_circle)";
      t.roll_deg = view["roll_deg"] ? view["roll_deg"].as<double>() : 0.0;
      t.object_target = makeIso(xyz, q);
      if (!previous_i3.name.empty() &&
          sameObjectPose(t.object_target, previous_i3.object_target, 1e-9, 1e-6))
      {
        error = "INSPECTION TARGETS AMBIGUOUS: corrected I3 object pose equals previous C_bottom; "
                "rename-only is not a target correction";
        return false;
      }
    }
    else
    {
      t.winner_key = view["winner_key"] ? view["winner_key"].as<std::string>() : std::string();
      if (t.winner_key.empty() || !winner[t.winner_key] || !winner[t.winner_key]["object_world"])
      {
        error = "INSPECTION TARGETS AMBIGUOUS: missing winner object_world for " + t.name;
        return false;
      }
      if (!yamlVec3(winner[t.winner_key]["object_world"]["xyz"], xyz) ||
          !yamlQuatXyzw(winner[t.winner_key]["object_world"]["xyzw"], q))
      {
        error = "INSPECTION TARGETS AMBIGUOUS: unreadable object pose for " + t.name;
        return false;
      }
      t.object_target = makeIso(xyz, q);
      t.roll_deg =
          winner[t.winner_key]["roll_deg"] ? winner[t.winner_key]["roll_deg"].as<double>() : 0.0;
      t.source = winner_path + ":" + t.winner_key + ".object_world";
    }
    t.tcp_b_target = t.object_target * tcp_b_object.inverse();
    out.push_back(t);
  }
  if (out.size() != 3 || out[2].name != "top_circle" ||
      out[2].physical_face != "ORIGINAL_TOP_CIRCLE")
  {
    error = "INSPECTION TARGETS AMBIGUOUS: I3 is not ORIGINAL_TOP_CIRCLE";
    return false;
  }
  return true;
}

HeldStateCheck evaluateHeldBState(planning_scene::PlanningScene& scene, const HandoverGeometry& geo,
                                  const GripperModelInfo& gripper_a, const GripperModelInfo& gripper_b,
                                  const std::vector<double>& home_a, const std::vector<double>& joints_b,
                                  double q_a_open, double q_b_hold, double pos_tol, double ori_tol,
                                  double face_center_tol, double face_normal_tol_deg, int max_contacts,
                                  int max_per_pair, const InspectionTarget* target)
{
  HeldStateCheck out;
  moveit::core::RobotState& state = scene.getCurrentStateNonConst();
  if (!applyFixedArmsAndGrippers(state, gripper_a, gripper_b, home_a, joints_b, q_a_open, q_b_hold,
                                 out.fail_reason))
    return out;
  out.q_a = jointOrNan(state, kGripperJointA);
  out.q_b = jointOrNan(state, kGripperJointB);
  if (!allVariablesFinite(state, out.fail_reason))
    return out;
  const auto* group_a = state.getJointModelGroup(kGroupA);
  const auto* group_b = state.getJointModelGroup(kGroupB);
  out.bounds_ok =
      group_a && group_b && state.satisfiesBounds(group_a) && state.satisfiesBounds(group_b);
  if (!out.bounds_ok)
  {
    out.fail_reason = "combined state outside joint bounds";
    return out;
  }
  if (maxAbsDiff(jointsOf(state, kArmAJoints), home_a) > 1e-9)
  {
    out.arm_a_fixed = false;
    out.fail_reason = "Arm A six-axis joints changed from Home";
    return out;
  }
  if (!nearlyEqual(out.q_a, q_a_open, 1e-6) || !nearlyEqual(out.q_b, q_b_hold, 1e-6))
  {
    out.gripper_ok = false;
    out.fail_reason = "gripper opening changed q_A=" + fmtScalar(out.q_a) +
                      " q_B=" + fmtScalar(out.q_b);
    return out;
  }
  if (!state.hasAttachedBody(geo.object_name))
  {
    out.fail_reason = "small_part not attached";
    return out;
  }
  const auto* body = state.getAttachedBody(geo.object_name);
  if (!body || body->getAttachedLinkName() != kTcpB)
  {
    out.fail_reason = "small_part not attached to arm_b_gripper_tcp";
    return out;
  }
  if (countNamedAttachments(state, geo.object_name) != 1 || scene.getWorld()->hasObject(geo.object_name))
  {
    out.fail_reason = "duplicate or world copy of small_part";
    return out;
  }
  out.attached_to_b = true;
  if (body->getShapePosesInLinkFrame().empty())
  {
    out.fail_reason = "attached body missing pose in link frame";
    return out;
  }
  poseError(geo.tcp_b_object, body->getShapePosesInLinkFrame().front(), out.relative_pos_error,
            out.relative_ori_error_deg);
  if (out.relative_pos_error > pos_tol || out.relative_ori_error_deg > ori_tol)
  {
    out.relative_attach_ok = false;
    out.fail_reason = "T_tcpB_object changed pos=" + fmtScalar(out.relative_pos_error) +
                      " m ori=" + fmtScalar(out.relative_ori_error_deg) + " deg";
    return out;
  }
  Eigen::Isometry3d world = Eigen::Isometry3d::Identity();
  if (!getAttachedWorldPose(state, geo.object_name, world, out.fail_reason))
    return out;
  if (target)
  {
    poseError(target->object_target, world, out.object_pos_error, out.object_ori_error_deg);
    const Eigen::Vector3d center = world * target->center_in_object;
    const Eigen::Vector3d normal = world.linear() * target->normal_in_object;
    out.face_center_error = (center - target->p1).norm();
    out.face_normal_error_deg = angleDeg(normal, target->d1);
    if (out.object_pos_error > pos_tol || out.object_ori_error_deg > ori_tol)
    {
      out.object_target_ok = false;
      out.fail_reason = "object pose error vs inspection target pos=" +
                        fmtScalar(out.object_pos_error) +
                        " m ori=" + fmtScalar(out.object_ori_error_deg) + " deg";
      return out;
    }
    if (out.face_center_error > face_center_tol || out.face_normal_error_deg > face_normal_tol_deg)
    {
      out.face_ok = false;
      out.fail_reason = "OBJECT_ALIGNMENT_ERROR center=" + fmtScalar(out.face_center_error) +
                        " m normal=" + fmtScalar(out.face_normal_error_deg) + " deg";
      return out;
    }
  }
  out.diag = checkState(scene, state, scene.getAllowedCollisionMatrix(), max_contacts, max_per_pair,
                        geo.object_name);
  out.collision_category = collisionCategory4e(out.diag);
  out.collision_pair = out.diag.pairs.empty() ? std::string() : out.diag.pairs.front();
  if (hasIllegalCollisionReturnHome(out.diag))
  {
    out.collision_ok = false;
    out.fail_reason = "illegal collision category=" + out.collision_category;
    if (!out.collision_pair.empty())
      out.fail_reason += " pair=" + out.collision_pair;
    return out;
  }
  out.ok = true;
  return out;
}

bool validateArmBPath(planning_scene::PlanningScene& scene, const HandoverGeometry& geo,
                      const GripperModelInfo& gripper_a, const GripperModelInfo& gripper_b,
                      const std::vector<double>& home_a, const std::vector<std::vector<double>>& waypoints_b,
                      double q_a_open, double q_b_hold, double max_joint_step, double pos_tol,
                      double ori_tol, double face_center_tol, double face_normal_tol_deg,
                      int max_contacts, int max_per_pair, PlanAttempt& attempt,
                      const InspectionTarget* end_target, CollisionObs& obs)
{
  if (waypoints_b.empty())
  {
    attempt.fail_reason = "empty Arm B waypoint list";
    return false;
  }
  size_t samples = 0;
  auto check_one = [&](int waypoint, int interp, const std::vector<double>& q_b, bool endpoint) {
    HeldStateCheck st =
        evaluateHeldBState(scene, geo, gripper_a, gripper_b, home_a, q_b, q_a_open, q_b_hold,
                           pos_tol, ori_tol, face_center_tol, face_normal_tol_deg, max_contacts,
                           max_per_pair, endpoint ? end_target : nullptr);
    ++samples;
    accumulateCollisionObs(obs, st.diag);
    if (!st.arm_a_fixed)
      attempt.fail_reason = st.fail_reason;
    if (!st.ok)
    {
      attempt.illegal_collision = !st.collision_ok;
      attempt.fail_reason = st.fail_reason;
      attempt.collision_category = st.collision_category;
      attempt.collision_pair = st.collision_pair;
      attempt.fail_waypoint = waypoint;
      attempt.fail_interp = interp;
      return false;
    }
    return true;
  };

  for (size_t i = 0; i < waypoints_b.size(); ++i)
  {
    const bool endpoint = (i + 1 == waypoints_b.size());
    if (!check_one(static_cast<int>(i), 0, waypoints_b[i], endpoint))
    {
      attempt.validation_samples = samples;
      return false;
    }
    if (i + 1 >= waypoints_b.size())
      continue;
    const double jump = maxAbsJointDelta(waypoints_b[i], waypoints_b[i + 1]);
    const int n = std::max(1, static_cast<int>(std::ceil(jump / std::max(1e-9, max_joint_step))));
    for (int s = 1; s < n; ++s)
    {
      const double t = static_cast<double>(s) / static_cast<double>(n);
      std::vector<double> q(waypoints_b[i].size(), 0.0);
      for (size_t j = 0; j < q.size(); ++j)
        q[j] = waypoints_b[i][j] + t * (waypoints_b[i + 1][j] - waypoints_b[i][j]);
      if (!check_one(static_cast<int>(i), s, q, false))
      {
        attempt.validation_samples = samples;
        return false;
      }
    }
  }
  attempt.validation_samples = samples;
  return true;
}

void printRequestAttachmentCompact(const RequestAttachmentDiag& diag)
{
  emit("REQUEST ATTACHMENT CHECK:");
  emit("  " + std::string(diag.pass ? "PASS" : "FAIL"));
  emit("  group=" + diag.group_name + " attached_count=" + std::to_string(diag.request_count) +
       " object=" + (diag.object_id.empty() ? std::string("(none)") : diag.object_id) +
       " link=" + (diag.attached_link.empty() ? std::string("(none)") : diag.attached_link));
  emit("  geometry: " + (diag.geometry.empty() ? std::string("(none)") : diag.geometry));
  emit("  touch_links: " + joinNames(diag.touch_links));
  if (!diag.pass)
  {
    emit("  Reason:");
    emit("    " + diag.fail_reason);
    emit("  Planning:");
    emit("    SKIPPED");
  }
}

SegmentPlan planArmBSegment(MoveGroup& arm_b, const planning_scene::PlanningScenePtr& local,
                            const HandoverGeometry& geo, const GripperModelInfo& gripper_a,
                            const GripperModelInfo& gripper_b, const Dual4dEndCandidate& handover,
                            const std::vector<double>& home_a, const std::vector<double>& start_b,
                            const std::vector<double>& goal_b, const InspectionTarget* end_target,
                            double q_a_open, double q_b_hold, int max_attempts, double planning_time,
                            double max_joint_step, double pos_tol, double ori_tol,
                            double face_center_tol, double face_normal_tol_deg, double chain_tol,
                            int max_contacts, int max_per_pair, CollisionObs& obs,
                            const moveit::core::RobotModelConstPtr& model, const ArmBLimits& limits,
                            double direction_eps, bool reject_gratuitous_j6)
{
  SegmentPlan out;
  out.start_b = start_b;
  out.goal_b = goal_b;
  double best_duration = std::numeric_limits<double>::infinity();
  std::string error;
  auto scene = makeTransferredScene(local, geo, gripper_a, gripper_b, handover.joints_a,
                                    handover.joints_b, q_a_open, q_b_hold, pos_tol, ori_tol, error);
  if (!scene)
  {
    out.fail_reason = "cannot rebuild DUAL-4D end scene: " + error;
    return out;
  }

  HeldStateCheck start = evaluateHeldBState(*scene, geo, gripper_a, gripper_b, home_a, start_b,
                                            q_a_open, q_b_hold, pos_tol, ori_tol, face_center_tol,
                                            face_normal_tol_deg, max_contacts, max_per_pair, nullptr);
  if (!start.ok)
  {
    out.fail_reason = "start state invalid: " + start.fail_reason;
    accumulateCollisionObs(obs, start.diag);
    return out;
  }
  HeldStateCheck goal = evaluateHeldBState(*scene, geo, gripper_a, gripper_b, home_a, goal_b,
                                           q_a_open, q_b_hold, pos_tol, ori_tol, face_center_tol,
                                           face_normal_tol_deg, max_contacts, max_per_pair, end_target);
  if (!goal.ok)
  {
    out.fail_reason = "goal state invalid: " + goal.fail_reason;
    accumulateCollisionObs(obs, goal.diag);
    return out;
  }

  moveit::core::RobotState start_state(scene->getCurrentState());
  applyFixedArmsAndGrippers(start_state, gripper_a, gripper_b, home_a, start_b, q_a_open, q_b_hold,
                            error);
  start_state.update();
  fillLocalAttachedDiag(start_state, geo, out.request_attachment);
  fillLocalAcmNotes(*local, *scene, geo, out.request_attachment);

  const auto goal_map = jointMapFrom(kArmBJoints, goal_b);
  arm_b.setPlanningTime(planning_time);
  arm_b.setNumPlanningAttempts(1);
  arm_b.setPlanningPipelineId("ompl");
  const std::string planner_id = arm_b.getPlannerId();
  const std::string pipeline_id =
      arm_b.getPlanningPipelineId().empty() ? std::string("ompl") : arm_b.getPlanningPipelineId();
  const std::string planner_label =
      pipeline_id + " / " + (planner_id.empty() ? std::string("default") : planner_id);

  for (int attempt = 1; attempt <= max_attempts; ++attempt)
  {
    PlanAttempt rec;
    rec.index = attempt;
    rec.planner = planner_label;
    out.attempts = attempt;

    arm_b.setStartState(start_state);
    if (!arm_b.setJointValueTarget(goal_map))
    {
      rec.error_name = "INVALID_GOAL_CONSTRAINTS";
      rec.fail_reason = "setJointValueTarget(Arm B) failed";
      out.fail_reason = rec.fail_reason;
      continue;
    }

    moveit_msgs::msg::MotionPlanRequest request;
    arm_b.constructMotionPlanRequest(request);
    fillRequestAttachmentDiag(request, geo, start_b, out.request_attachment);
    if (out.request_attachment.group_name != kGroupB)
    {
      out.request_attachment.pass = false;
      out.request_attachment.fail_reason =
          "MotionPlanRequest.group_name is " + out.request_attachment.group_name + " not arm_b";
    }
    if (out.request_attachment.pass &&
        (!out.request_attachment.start_a_complete || maxAbsDiff(out.request_attachment.start_a, home_a) > 1e-6))
    {
      out.request_attachment.pass = false;
      out.request_attachment.fail_reason = "MotionPlanRequest start Arm A is not Home";
    }
    if (out.request_attachment.pass &&
        (!out.request_attachment.start_b_complete || maxAbsDiff(out.request_attachment.start_b, start_b) > 1e-6))
    {
      out.request_attachment.pass = false;
      out.request_attachment.fail_reason = "MotionPlanRequest start Arm B is not segment start";
    }
    printRequestAttachmentCompact(out.request_attachment);
    if (!out.request_attachment.pass)
    {
      rec.planning_skipped = true;
      rec.error_name = "REQUEST_ATTACHMENT_CHECK_FAIL";
      rec.fail_reason = out.request_attachment.fail_reason;
      out.fail_reason = rec.fail_reason;
      out.selected = rec;
      return out;
    }

    MoveGroup::Plan plan;
    const auto t0 = std::chrono::steady_clock::now();
    const auto code = arm_b.plan(plan);
    rec.planning_time_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    rec.error_code = code.val;
    rec.error_name = moveitErrorName(code.val);
    rec.plan_success = (code == moveit::core::MoveItErrorCode::SUCCESS);
    rec.trajectory = plan.trajectory_.joint_trajectory;
    rec.waypoint_count = rec.trajectory.points.size();
    if (!rec.plan_success)
    {
      rec.fail_reason = "planner error " + rec.error_name;
      out.fail_reason = rec.fail_reason;
      continue;
    }
    if (rec.waypoint_count < 2)
    {
      rec.fail_reason = "planned trajectory has too few waypoints";
      out.fail_reason = rec.fail_reason;
      continue;
    }

    bool unexpected_a = false;
    for (const auto& name : rec.trajectory.joint_names)
    {
      if (name.rfind("arm_a_", 0) == 0)
      {
        std::vector<double> qa;
        std::string extract_error;
        for (size_t p = 0; p < rec.trajectory.points.size(); ++p)
        {
          if (!extractTrajectoryJoints(rec.trajectory, p, kArmAJoints, qa, extract_error) ||
              maxAbsDiff(qa, home_a) > 1e-9)
          {
            unexpected_a = true;
            rec.fail_reason = "Arm A joints changed in planned trajectory";
            break;
          }
        }
      }
    }
    if (unexpected_a)
    {
      out.fail_reason = rec.fail_reason;
      continue;
    }

    std::vector<std::vector<double>> waypoints_b;
    waypoints_b.reserve(rec.waypoint_count);
    bool extract_ok = true;
    for (size_t p = 0; p < rec.waypoint_count; ++p)
    {
      std::vector<double> qb;
      std::string extract_error;
      if (!extractTrajectoryJoints(rec.trajectory, p, kArmBJoints, qb, extract_error))
      {
        rec.fail_reason = extract_error;
        extract_ok = false;
        break;
      }
      waypoints_b.push_back(std::move(qb));
    }
    if (!extract_ok)
    {
      out.fail_reason = rec.fail_reason;
      continue;
    }
    if (maxAbsDiff(waypoints_b.front(), start_b) > chain_tol)
    {
      rec.fail_reason = "trajectory start does not match segment start";
      out.fail_reason = rec.fail_reason;
      continue;
    }
    if (maxAbsDiff(waypoints_b.back(), goal_b) > chain_tol)
    {
      rec.fail_reason = "trajectory end does not match inspection IK within continuity tol";
      out.fail_reason = rec.fail_reason;
      continue;
    }
    rec.joint_path_length = jointPathLength(waypoints_b);
    if (!validateArmBPath(*scene, geo, gripper_a, gripper_b, home_a, waypoints_b, q_a_open, q_b_hold,
                          max_joint_step, pos_tol, ori_tol, face_center_tol, face_normal_tol_deg,
                          max_contacts, max_per_pair, rec, end_target, obs))
    {
      out.fail_reason = rec.fail_reason;
      continue;
    }
    rec.validated = true;
    ++out.successful_attempts;
    TimedPath timed;
    timed.attempt = rec;
    timed.collision_validated = true;
    timed.path_length = rec.joint_path_length;
    if (!timeParameterizeArmB(model, start_state, rec.trajectory, limits, direction_eps, 1e-3, 1e-3,
                              timed))
    {
      timed.ok = false;
      timed.time_parameterized = false;
      emit("  OMPL path VALID, time parameterization: INVALID FOR TIME COMPARISON (" +
           timed.fail_reason + ")");
    }
    else if (!validateArmBPath(*scene, geo, gripper_a, gripper_b, home_a, timed.waypoints, q_a_open,
                               q_b_hold, max_joint_step, pos_tol, ori_tol, face_center_tol,
                               face_normal_tol_deg, max_contacts, max_per_pair, rec, end_target, obs))
    {
      timed.ok = false;
      timed.collision_validated = false;
      timed.fail_reason = "TOTG resampled path failed collision validation: " + rec.fail_reason;
      emit("  OMPL path VALID, timed resample: INVALID FOR TIME COMPARISON (" + timed.fail_reason +
           ")");
    }
    else if (timed.gratuitous_j6 && reject_gratuitous_j6)
    {
      timed.ok = false;
      timed.fail_reason = "J6 GRATUITOUS ROTATION";
      emit("  J6 GRATUITOUS ROTATION: YES (excluded from optimization candidates)");
    }
    else
    {
      ++out.time_parameterized_attempts;
      emit("  timed path duration=" + fmtScalar(timed.duration) +
           " s J1_travel=" + fmtScalar(timed.travel[0]) + " J6_travel=" + fmtScalar(timed.travel[5]));
    }
    if (timed.ok && timed.time_parameterized && timed.collision_validated)
    {
      if (!out.ok || timed.duration < best_duration)
      {
        out.selected = rec;
        out.end_b = waypoints_b.back();
        out.ok = true;
        best_duration = timed.duration;
      }
      out.timed_paths.push_back(std::move(timed));
    }
    else
    {
      out.timed_paths.push_back(std::move(timed));
    }
  }
  if (!out.ok && out.fail_reason.empty())
  {
    if (out.successful_attempts > 0 && out.time_parameterized_attempts == 0)
      out.fail_reason = "INVALID FOR TIME COMPARISON";
    else
      out.fail_reason = "NO PATH FOUND WITH CURRENT PLANNER SETTINGS";
  }
  return out;
}

InspectionIk classifyInspectionIk(planning_scene::PlanningScene& scene, const HandoverGeometry& geo,
                                  const GripperModelInfo& gripper_a, const GripperModelInfo& gripper_b,
                                  const std::vector<double>& home_a, const IkCandidate& cand,
                                  const InspectionTarget& target, double q_a_open, double q_b_hold,
                                  double pos_tol, double ori_tol, double face_center_tol,
                                  double face_normal_tol_deg, int max_contacts, int max_per_pair)
{
  InspectionIk out;
  out.ik = cand;
  if (!cand.within_bounds)
  {
    out.status = "OUT_OF_BOUNDS";
    out.fail_reason = "IK outside joint bounds";
    return out;
  }
  if (!cand.fk_ok)
  {
    out.status = "TARGET_POSE_ERROR";
    out.fail_reason = "TCP FK error pos=" + fmtScalar(cand.fk_position_error) +
                      " m ori=" + fmtScalar(cand.fk_orientation_error_deg) + " deg";
    return out;
  }
  HeldStateCheck st =
      evaluateHeldBState(scene, geo, gripper_a, gripper_b, home_a, cand.joints, q_a_open, q_b_hold,
                         pos_tol, ori_tol, face_center_tol, face_normal_tol_deg, max_contacts,
                         max_per_pair, &target);
  out.diag = st.diag;
  out.object_pos_error = st.object_pos_error;
  out.object_ori_error_deg = st.object_ori_error_deg;
  out.face_center_error = st.face_center_error;
  out.face_normal_error_deg = st.face_normal_error_deg;
  out.fail_reason = st.fail_reason;
  if (!st.bounds_ok)
    out.status = "OUT_OF_BOUNDS";
  else if (!st.collision_ok)
    out.status = "STATIC_COLLISION";
  else if (!st.face_ok)
    out.status = "OBJECT_ALIGNMENT_ERROR";
  else if (!st.object_target_ok || !st.relative_attach_ok)
    out.status = "TARGET_POSE_ERROR";
  else if (!st.ok)
    out.status = "IK_INVALID";
  else
    out.status = "VALID";
  return out;
}

void printInspectionIkSummary(const InspectionTarget& target, int raw, const std::vector<InspectionIk>& all,
                              const std::vector<InspectionIk>& valid)
{
  emit("");
  emit("Inspection " + std::to_string(target.index) + " (" + target.name + "):");
  emit("  raw IK found:");
  emit("    " + std::to_string(raw));
  emit("  unique after FK:");
  emit("    " + std::to_string(all.size()));
  emit("  valid IK:");
  emit("    " + std::to_string(valid.size()));
  std::map<std::string, int> counts;
  for (const auto& item : all)
    ++counts[item.status];
  if (!counts.empty())
  {
    emit("  status counts:");
    for (const auto& item : counts)
      emit("    " + item.first + ": " + std::to_string(item.second));
  }
  for (const auto& item : all)
  {
    if (item.status == "VALID")
      continue;
    emit("  " + item.status + " unique_" + std::to_string(item.id) +
         " joints=" + fmtVec(item.ik.joints));
    emit("    reason=" + (item.fail_reason.empty() ? std::string("(none)") : item.fail_reason));
    emit("    category=" + collisionCategory4e(item.diag));
    if (item.diag.pairs.empty())
      emit("    collision pairs: (none)");
    else
    {
      emit("    collision pairs:");
      for (const auto& pair : item.diag.pairs)
        emit("      " + pair);
    }
  }
}

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("dual_arm_b_time_optimization_test", options);

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
    emit("========== DUAL-6 ARM B TIME-FIRST OPTIMIZATION ==========");
    emit("MODEL STATE VALIDATION + OFFLINE TIME PARAMETERIZATION. PLAN ONLY.");
    emit("NO EXECUTION. NO GRIPPER COMMANDS. NO /apply_planning_scene.");
    emit("Revalidates DUAL-4C/4D/5-R geometry, then time-optimizes Arm B H→I1→I2→I3.");

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
    const double cartesian_step = getDouble(node, "cartesian_step_m", 0.005);
    const double densify_min = getDouble(node, "densify_min_cartesian_m", 0.001);
    const double max_joint_jump = getDouble(node, "max_adjacent_joint_jump_rad", 0.35);
    const double ik_path_timeout = getDouble(node, "ik_timeout_path", timeout_exact);
    const double gripper_step = getDouble(node, "gripper_step_q", 0.002);
    const double object_pos_tol = getDouble(node, "object_position_tol_m", pos_tol);
    const double object_ori_tol = getDouble(node, "object_orientation_tol_deg", ori_tol);

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
    overlayJointLimitsFromMoveGroup(node);

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
    const ArmBLimits arm_b_limits = readArmBLimits(*model);
    printArmBLimits(arm_b_limits);
    if (!arm_b_limits.complete)
    {
      emit("");
      emit("DUAL-6 INCOMPLETE");
      emit("TIME OPTIMIZATION LIMITS INCOMPLETE");
      emit("  " + arm_b_limits.missing);
      emit("Invented velocity/acceleration: NOT USED");
      emit("Time parameterization: NOT PERFORMED");
    }
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

    const PipelineInfo pipelines = inspectMoveGroupPipelines(node);
    emit("");
    emit("Planning pipelines reported by /move_group:");
    emit("  " + (pipelines.pipelines.empty() ? std::string("(none / unavailable)") :
                                               joinNames(pipelines.pipelines)));
    emit("Pilz LIN loaded in running /move_group:");
    emit(pipelines.pilz_loaded ? "  YES" : "  NO");
    if (!pipelines.note.empty())
      emit("  note: " + pipelines.note);
    emit("Planner:");
    emit(pipelines.pilz_loaded ? "  Pilz LIN" : "  Cartesian sampled validation");
    emit("OMPL will not be treated as a linear planner.");
    emit("Arm A:");
    emit("  Handover fixed");
    emit("Arm A gripper:");
    emit("  q = 0.083 (NOMINAL)");
    emit("Arm B gripper:");
    emit("  q = 0.0 (NOMINAL OPEN)");
    emit("Object:");
    emit("  " + geo.object_name + " -> " + std::string(kTcpA));
    emit("Arm B TCP start:");
    emit("  " + fmtXyz(geo.world_pre_b.translation()));
    emit("Arm B TCP goal:");
    emit("  " + fmtXyz(geo.world_tcp_b.translation()));
    emit("Requested direction:");
    emit("  world +X");
    emit("Requested distance:");
    emit("  " + fmtScalar(geo.prehandover_distance) + " m");
    emit("Cartesian TCP sampling step:");
    emit("  " + fmtScalar(cartesian_step) + " m");

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
      emit("DUAL-4D:");
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
                    size_t b_index, const Eigen::Isometry3d& b_target, double q_a, double q_b) {
      return evaluateCombo(*local, geo, gripper_model_a, gripper_model_b, a_cand, b_cand, a_index,
                           b_index, b_target, q_a, q_b, pos_tol, ori_tol, max_contacts,
                           max_per_pair, false);
    };

    std::vector<ComboResult> state1;
    std::vector<ComboResult> state2;
    for (size_t i = 0; i < arm_a.unique.size(); ++i)
    {
      for (size_t j = 0; j < arm_b_pre.unique.size(); ++j)
      {
        auto item = eval(arm_a.unique[i], arm_b_pre.unique[j], i + 1, j + 1, geo.world_pre_b,
                         gripper.q_a_grasp, gripper.q_b_open);
        item.state_name = "STATE 1";
        state1.push_back(std::move(item));
      }
      for (size_t k = 0; k < arm_b_handover.unique.size(); ++k)
      {
        auto item2 = eval(arm_a.unique[i], arm_b_handover.unique[k], i + 1, k + 1, geo.world_tcp_b,
                          gripper.q_a_grasp, gripper.q_b_open);
        item2.state_name = "STATE 2";
        state2.push_back(std::move(item2));
      }
    }

    emit("");
    emit("----------------------------------------");
    emit("CANDIDATE SEARCH");
    emit("----------------------------------------");
    emit("A IK candidates: " + std::to_string(arm_a.unique.size()));
    emit("B Pre IK candidates: " + std::to_string(arm_b_pre.unique.size()));
    emit("B Handover IK candidates: " + std::to_string(arm_b_handover.unique.size()));
    emit("Arm B receiving / gripper closure: NOT PERFORMED");

    auto printStaticBrief = [&](const std::string& heading, const std::vector<ComboResult>& results) {
      int strict_free = 0;
      for (const auto& item : results)
      {
        if (item.strict_free)
          ++strict_free;
      }
      emit("");
      emit(heading);
      emit("  combinations tested: " + std::to_string(results.size()));
      emit("  strict collision-free: " + std::to_string(strict_free));
      for (const auto& item : results)
      {
        if (item.strict_free)
          emit("    A" + std::to_string(item.a_index) + " + B" + std::to_string(item.b_index) +
               " STRICT-FREE");
        else
          emit("    A" + std::to_string(item.a_index) + " + B" + std::to_string(item.b_index) +
               " FAIL " + (item.fail_reason.empty() ? collisionCategory(item.strict) :
                                                      item.fail_reason));
      }
    };
    printStaticBrief("STATE 1: A handover + B pre/open", state1);
    printStaticBrief("STATE 2: A handover + B handover/open", state2);

    std::vector<std::tuple<size_t, size_t, size_t>> triplet_ids;
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

    std::string block_reason = "no IK candidates";
    if (arm_a.unique.empty())
      block_reason = "STATE search blocked: no Arm A Handover IK";
    else if (arm_b_pre.unique.empty())
      block_reason = "STATE 1 blocked: no Arm B PreHandover IK";
    else if (arm_b_handover.unique.empty())
      block_reason = "STATE 2 blocked: no Arm B Handover IK";

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
        if (item2.a_index == i + 1 && item2.strict_free)
          han_ok.push_back(item2.b_index);
      }
      emit("");
      emit("A candidate: " + std::to_string(i + 1));
      emit("  Compatible B Pre candidates (STATE 1 strict): " +
           (pre_ok.empty() ? std::string("(none)") : joinNames([&]() {
             std::vector<std::string> names;
             for (size_t idx : pre_ok)
               names.push_back("B_pre" + std::to_string(idx));
             return names;
           }())));
      emit("  Compatible B Handover candidates (STATE 2 strict, B open): " +
           (han_ok.empty() ? std::string("(none)") : joinNames([&]() {
             std::vector<std::string> names;
             for (size_t idx : han_ok)
               names.push_back("B_handover" + std::to_string(idx));
             return names;
           }())));
      if (pre_ok.empty() && han_ok.empty())
        block_reason = "A" + std::to_string(i + 1) + " blocked in STATE 1 and STATE 2";
      else if (pre_ok.empty())
        block_reason = "A" + std::to_string(i + 1) + " blocked in STATE 1";
      else if (han_ok.empty())
        block_reason = "A" + std::to_string(i + 1) + " blocked in STATE 2";
      if (!pre_ok.empty() && !han_ok.empty())
      {
        for (size_t j : pre_ok)
        {
          for (size_t k : han_ok)
            triplet_ids.emplace_back(i + 1, j, k);
        }
      }
    }
    emit("");
    emit("Compatible static triplets:");
    emit("  " + std::to_string(triplet_ids.size()));
    if (triplet_ids.empty())
    {
      emit("  (none)");
      emit("  block reason: " + block_reason);
    }
    else
    {
      for (const auto& triple : triplet_ids)
      {
        emit("  (A" + std::to_string(std::get<0>(triple)) + ", B_pre" +
             std::to_string(std::get<1>(triple)) + ", B_handover" +
             std::to_string(std::get<2>(triple)) + ")");
      }
    }

    emit("");
    emit("----------------------------------------");
    emit("CARTESIAN SAMPLED LINEAR APPROACH");
    emit("----------------------------------------");
    emit("This is Cartesian sampled validation, not Pilz LIN planning success.");
    emit("Discrete samples without continuous collision detection do not prove");
    emit("the path is mathematically collision-free between samples.");
    emit("Time parameterization: NOT PERFORMED");
    emit("B finger-object contact during open approach: ILLEGAL (not expected grasp).");

    std::vector<PathResult> path_results;
    std::vector<PathResult> retained;
    int n_ik_disc = 0;
    int n_collision = 0;
    int n_end_mismatch = 0;
    int n_other = 0;
    bool path_truncated = false;
    std::map<std::pair<size_t, size_t>, PathResult> tracked_paths;
    if (!arm_b_handover.unique.empty())
    {
      for (const auto& triple : triplet_ids)
      {
        const size_t a_index = std::get<0>(triple);
        const size_t pre_index = std::get<1>(triple);
        const size_t han_index = std::get<2>(triple);
        const auto key = std::make_pair(a_index, pre_index);
        if (tracked_paths.find(key) == tracked_paths.end())
        {
          tracked_paths[key] = validateCartesianPath(
              *local, geo, gripper_model_a, gripper_model_b, arm_a.unique[a_index - 1],
              arm_b_pre.unique[pre_index - 1], arm_b_handover.unique[han_index - 1], a_index,
              pre_index, han_index, gripper.q_a_grasp, gripper.q_b_open, cartesian_step,
              densify_min, max_joint_jump, ik_path_timeout, pos_tol, ori_tol, 1.0e9, max_contacts,
              max_per_pair);
        }
        PathResult path =
            withHandoverMatch(tracked_paths[key], arm_b_handover.unique[han_index - 1], han_index,
                              min_distance);
        if (path.truncated)
          path_truncated = true;
        if (path.pass)
          retained.push_back(path);
        else if (path.ik_discontinuity)
          ++n_ik_disc;
        else if (path.end_mismatch)
          ++n_end_mismatch;
        else if (path.collision_fail || path.b_finger_object_during_approach)
          ++n_collision;
        else
          ++n_other;
        path_results.push_back(std::move(path));
      }
    }

    emit("Triplets attempted: " + std::to_string(path_results.size()));
    emit("Straight-line planning success: " + std::to_string(retained.size()));
    emit("Full-state sampled collision PASS: " + std::to_string(retained.size()));
    emit("IK discontinuity: " + std::to_string(n_ik_disc));
    emit("Collision failure: " + std::to_string(n_collision));
    emit("Endpoint IK mismatch: " + std::to_string(n_end_mismatch));
    emit("Other path failures: " + std::to_string(n_other));

    emit("");
    emit("Failed triplets:");
    bool any_path_fail_printed = false;
    for (const auto& path : path_results)
    {
      if (path.pass)
        continue;
      any_path_fail_printed = true;
      printPathFailure(path);
    }
    if (!any_path_fail_printed)
      emit("  (none)");

    emit("");
    emit("----------------------------------------");
    emit("SELECTED / RETAINED CANDIDATES");
    emit("----------------------------------------");
    if (retained.empty())
      emit("  (none)");
    for (size_t n = 0; n < retained.size(); ++n)
    {
      const auto& path = retained[n];
      emit("");
      emit(std::string(n == 0 ? "SELECTED candidate:" : "RETAINED candidate:"));
      emit("  A candidate: " + std::to_string(path.a_index));
      emit("  B Pre candidate: " + std::to_string(path.pre_index));
      emit("  B Handover candidate: " + std::to_string(path.han_index));
      emit("  Arm A unchanged: " + std::string(path.arm_a_unchanged ? "YES" : "NO"));
      emit("  Arm A joints_rad: " + fmtVec(arm_a.unique[path.a_index - 1].joints));
      emit("  Arm B start joints_rad: " + fmtVec(arm_b_pre.unique[path.pre_index - 1].joints));
      emit("  Arm B end joints_rad: " + fmtVec(path.end_joints_b));
      emit("  Arm B start FK: " + fmtXyz(path.start_tcp));
      emit("  Arm B end FK: " + fmtXyz(path.end_tcp));
      emit("  Maximum TCP line error: " + fmtScalar(path.max_line_error) + " m");
      emit("  Maximum orientation error: " + fmtScalar(path.max_ori_error) + " deg");
      emit("  Maximum adjacent joint change: " + fmtScalar(path.max_adjacent_joint_change) +
           " rad");
      emit("  Arm B total joint path length: " + fmtScalar(path.arm_b_path_length) + " rad");
      emit("  Maximum Cartesian gap: " + fmtScalar(path.max_cartesian_gap) + " m");
      emit("  Maximum joint-space gap: " + fmtScalar(path.max_joint_gap) + " rad");
      emit("  Collision validation samples: " + std::to_string(path.samples.size()));
      emit("  Planned trajectory waypoints: " + std::to_string(path.samples.size()));
      emit("  Time parameterization: NOT PERFORMED");
      emit("  End-to-selected-handover L2: " + fmtScalar(path.end_to_k_l2));
    }

    emit("");
    emit("B finger-object during open approach: " +
         std::string(any_b_finger_part ? "observed in static endpoints (not treated as expected "
                                         "grasp along the path)" :
                                         "none in static STATE 1/2"));
    if (!all_gripper_interference.empty())
    {
      emit("Modeled Arm A gripper <-> Arm B gripper contacts in static endpoints:");
      for (const auto& pair : all_gripper_interference)
        emit("  " + pair);
    }
    if (any_truncated || path_truncated)
    {
      emit("CONTACT ENUMERATION INCOMPLETE");
      emit("At least one check hit max_contacts; collision-free claims from truncated lists are invalid.");
    }
    emit("No arm_a/arm_b blanket allowance. No upperarm-column allowance. No finger-finger allowance.");
    emit("DUAL-4C/4D candidate revalidation complete. Entering DUAL-5 Arm B inspection sequence.");

    const std::string step15_yaml = getString(
        node, "step15_winner_yaml",
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
            "/fr_task_ws/src/fr_task_planner/config/step15_optimized_lift_to_a_winner.yaml");
    const int max_plan_attempts = getInt(node, "ompl_attempts_per_edge",
                                         getInt(node, "max_planning_attempts", 5));
    const double planning_time = getDouble(node, "planning_time_sec", 10.0);
    const double max_validation_step = getDouble(
        node, "max_validation_joint_step_rad", getDouble(node, "max_validation_joint_step", 0.02));
    const double time_near_ratio = getDouble(node, "time_near_optimal_ratio", 0.05);
    const double direction_eps = getDouble(node, "joint_direction_epsilon_rad", 0.005);
    const bool prefer_low_j1 = getBool(node, "prefer_low_j1", true);
    const bool prefer_meaningful_j6 = getBool(node, "prefer_meaningful_j6", true);
    const bool reject_gratuitous_j6 = getBool(node, "reject_gratuitous_j6_full_turn", true);

    std::vector<double> home_joints;
    std::string home_source;
    std::string home_error;
    const bool home_loaded = loadExistingHome(step14_yaml, step15_yaml, home_joints, home_source,
                                              home_error);
    if (!home_loaded)
    {
      emit("");
      emit("HOME DEFINITION AMBIGUOUS / MISSING");
      emit("  " + home_error);
      emit("DUAL-4E:");
      emit("  INCOMPLETE");
      stop();
      return 3;
    }

    emit("");
    emit("----------------------------------------");
    emit("HOME DEFINITION");
    emit("----------------------------------------");
    emit("Home source:");
    emit("  " + home_source);
    emit("Home joints:");
    emit("  " + fmtVec(home_joints, 12));

    const double q_a_grasp = gripper.q_a_grasp;
    const double q_a_open = 0.0;
    const double q_b_open = gripper.q_b_open;
    const double q_b_recv = gripper.q_b_receive;

    emit("");
    emit("----------------------------------------");
    emit("DUAL-4D REVALIDATION (all DUAL-4C retained candidates)");
    emit("----------------------------------------");
    std::vector<Dual4dEndCandidate> dual4d_ok;
    std::vector<Dual4dEndCandidate> dual4d_fail;
    for (const auto& path : retained)
    {
      Dual4dEndCandidate cand;
      cand.a_index = path.a_index;
      cand.pre_index = path.pre_index;
      cand.han_index = path.han_index;
      cand.joints_a = arm_a.unique[path.a_index - 1].joints;
      cand.joints_b = path.end_joints_b;
      cand.label = "A" + std::to_string(path.a_index) + "+B_pre" + std::to_string(path.pre_index) +
                   "+B_handover" + std::to_string(path.han_index);
      const bool ok =
          runDual4dTransfer(local, geo, gripper_model_a, gripper_model_b, cand.joints_a,
                            cand.joints_b, q_a_grasp, q_a_open, q_b_open, q_b_recv, gripper_step,
                            object_pos_tol, object_ori_tol, max_contacts, max_per_pair, cand);
      emit("  " + cand.label + (ok ? " DUAL-4D PASS" : " DUAL-4D FAIL " + cand.fail_stage + " " +
                                                           cand.fail_reason));
      if (ok)
        dual4d_ok.push_back(std::move(cand));
      else
        dual4d_fail.push_back(std::move(cand));
    }
    emit("DUAL-4C retained candidates:");
    emit("  " + std::to_string(retained.size()));
    emit("DUAL-4D-compatible candidates:");
    emit("  " + std::to_string(dual4d_ok.size()));

    const std::string sequence_yaml = getString(
        node, "sequence_yaml",
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
            "/fr_task_ws/src/fr_task_planner/config/dual_arm_b_inspection_sequence.yaml");
    const std::string step12c_yaml = getString(
        node, "step12c_winner_yaml",
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
            "/fr_task_ws/src/fr_task_planner/config/step12c_tilted_camera_winner.yaml");
    const int max_ik_per_target = getInt(node, "max_ik_candidates_per_target", 8);
    const double chain_tol = getDouble(node, "chain_joint_continuity_tol_rad", 1.0e-4);
    const double face_center_tol = getDouble(node, "face_center_tol_m", 0.002);
    const double face_normal_tol = getDouble(node, "face_normal_tol_deg", 3.0);

    std::vector<InspectionTarget> targets;
    InspectionTarget previous_i3;
    std::string order_source;
    std::string inspect_error;
    const bool targets_ok =
        loadInspectionTargets(step12c_yaml, sequence_yaml, geo.tcp_b_object, targets, previous_i3,
                              order_source, inspect_error);
    if (!targets_ok || targets.size() != 3)
    {
      emit("");
      emit("========== DUAL-5-R FACE CORRECTION ==========");
      emit("");
      emit("DUAL-5-R INCOMPLETE");
      if (inspect_error.find("ORIGINAL TOP FACE AMBIGUOUS") != std::string::npos)
        emit("ORIGINAL TOP FACE AMBIGUOUS");
      else if (inspect_error.find("ORDER") != std::string::npos)
        emit("INSPECTION ORDER AMBIGUOUS");
      else
        emit("INSPECTION TARGETS AMBIGUOUS");
      emit("  " + inspect_error);
      stop();
      return 3;
    }

    emit("");
    emit("========== DUAL-5-R ARM B INSPECTION SEQUENCE ==========");
    emit("");
    emit("RobotModel:");
    emit("  fairino3_dual_robot");
    emit("");
    emit("PlanningScene:");
    emit(std::string("  ") +
         ((env_complete && live_acm_check.complete) ? "COMPLETE" : "INCOMPLETE"));
    emit("");
    emit("Planner:");
    emit("  OMPL / connecting after target print");
    emit("");
    emit("Arm A:");
    emit("  HOME / FIXED");
    emit("Arm A joints:");
    emit("  " + fmtVec(home_joints, 12));
    emit("Arm A gripper:");
    emit("  OPEN q=0.000");
    emit("Arm B gripper:");
    emit("  HOLD q=" + fmtScalar(q_b_recv) + " NOMINAL");
    emit("Object:");
    emit("  small_part");
    emit("Attachment:");
    emit("  arm_b_gripper_tcp");
    emit("Physical grasp:");
    emit("  NOT VERIFIED");
    emit("");
    emit("----------------------------------------");
    emit("INSPECTION TARGET DEFINITIONS");
    emit("----------------------------------------");
    emit("");
    for (const auto& t : targets)
    {
      emit("Target " + std::to_string(t.index) + " source:");
      emit("  " + t.source);
      emit("  view frame: " + t.view_source);
      emit("Target " + std::to_string(t.index) + ":");
      emit("  name: " + t.name);
      emit("  winner key: " + t.winner_key);
      emit("  roll_deg: " + fmtScalar(t.roll_deg));
      emit("  reference frame: world");
      emit("  object position: " + fmtXyz(t.object_target.translation()));
      emit("  object orientation xyzw: " + fmtXyzw(Eigen::Quaterniond(t.object_target.rotation())));
      emit("  Arm B TCP position: " + fmtXyz(t.tcp_b_target.translation()));
      emit("  Arm B TCP orientation xyzw: " + fmtXyzw(Eigen::Quaterniond(t.tcp_b_target.rotation())));
      emit("  P1: " + fmtXyz(t.p1));
      emit("  D1: " + fmtXyz(t.d1));
      emit("  center_in_object: " + fmtXyz(t.center_in_object));
      emit("  normal_in_object: " + fmtXyz(t.normal_in_object));
      emit("  semantic: C. small_part object pose / face alignment");
      emit("    (view center -> P1, face normal -> D1; Arm B TCP derived via T_tcpB_object)");
      emit("  physical face: " + t.physical_face);
      emit("");
    }
    emit("Inspection order:");
    emit("  " + targets[0].name + " -> " + targets[1].name + " -> " + targets[2].name);
    emit("  source: " + order_source);
    emit("  1 -> 2 -> 3");
    emit("");
    emit("========== DUAL-5-R FACE CORRECTION ==========");
    emit("");
    emit("User-required I3 physical face:");
    emit("  ORIGINAL_TOP_CIRCLE");
    emit("Original top physical face:");
    emit("  object +Z circular cap; center_in_object [0,0,+H/2]; initial table pose identity,");
    emit("  world +Z up, table contact at object -Z. Source:");
    emit("  ~/fairino_ws/src/fr_control/config/stage4_config.yaml:object.initial_pose");
    emit("  launch/inspection_view_geometry.py:required_cylinder_views/top_circle");
    emit("Original bottom physical face:");
    emit("  object -Z circular cap; center_in_object [0,0,-H/2]; initial table-contact face.");
    emit("  launch/inspection_view_geometry.py:attach_bottom_circle_view/bottom_circle");
    emit("Existing top_circle implementation:");
    emit("  launch/inspection_view_geometry.py:view_name_from_normal(+Z) -> top_circle");
    emit("  inspection_visibility.cpp:topCircleRoiDef = object +Z original upward top");
    emit("Existing bottom_circle implementation:");
    emit("  make_bottom_circle_view = object -Z original table-contact face");
    emit("Previous DUAL-5 I3:");
    if (previous_i3.name.empty())
      emit("  (C_bottom not present in winner YAML)");
    else
    {
      emit("  " + previous_i3.source);
      emit("  name: " + previous_i3.name + " / " + previous_i3.physical_face);
    }
    emit("Corrected DUAL-5 I3:");
    emit("  " + targets[2].source);
    emit("  name: " + targets[2].name + " / " + targets[2].physical_face);
    emit("Correction source:");
    emit("  " + targets[2].source);
    emit("Inspection order:");
    emit("  side_pos_y");
    emit("  side_neg_y");
    emit("  corrected original top circle");
    emit("");
    emit("Previous I3 object:");
    if (previous_i3.name.empty())
      emit("  (unavailable)");
    else
    {
      emit("  xyz " + fmtXyz(previous_i3.object_target.translation()));
      emit("  xyzw " + fmtXyzw(Eigen::Quaterniond(previous_i3.object_target.rotation())));
    }
    emit("Corrected I3 object:");
    emit("  xyz " + fmtXyz(targets[2].object_target.translation()));
    emit("  xyzw " + fmtXyzw(Eigen::Quaterniond(targets[2].object_target.rotation())));
    emit("Corrected I3 TCP:");
    emit("  xyz " + fmtXyz(targets[2].tcp_b_target.translation()));
    emit("  xyzw " + fmtXyzw(Eigen::Quaterniond(targets[2].tcp_b_target.rotation())));
    emit("TCP derived using fixed T_tcpB_object:");
    emit("  YES");
    if (!previous_i3.name.empty())
    {
      const bool same_xyz =
          (previous_i3.object_target.translation() - targets[2].object_target.translation()).norm() <
          1e-9;
      emit("Object translation vs previous C_bottom:");
      emit(std::string("  ") + (same_xyz ? "SAME (cap centers are both P1 - (H/2)*D1)" : "DIFFERENT"));
      emit("Object orientation vs previous C_bottom:");
      emit("  DIFFERENT (top +Z vs bottom -Z mapped onto D1; not a rename)");
    }
    emit("CORRECT I3 TARGET:");
    emit("  RESOLVED");

    std::unique_ptr<MoveGroup> move_group_b;
    std::string planner_print = "OMPL / (MoveGroupInterface not created)";
    if (env_complete && live_acm_check.complete)
    {
      emit("");
      emit("Connecting MoveGroupInterface group=arm_b (wait_for_servers=30s)...");
      try
      {
        move_group_b = std::make_unique<MoveGroup>(
            node, "arm_b", std::shared_ptr<tf2_ros::Buffer>(), rclcpp::Duration::from_seconds(30.0));
        move_group_b->setPlanningPipelineId("ompl");
        move_group_b->setPlanningTime(planning_time);
        move_group_b->setNumPlanningAttempts(1);
        const std::string planner_id = move_group_b->getPlannerId();
        const std::string pipeline_id = move_group_b->getPlanningPipelineId().empty() ?
                                            std::string("ompl") :
                                            move_group_b->getPlanningPipelineId();
        planner_print =
            pipeline_id + " / " + (planner_id.empty() ? std::string("default") : planner_id);
        emit("MoveGroupInterface connected: " + planner_print);
      }
      catch (const std::exception& e)
      {
        emit(std::string("MoveGroupInterface connect failed: ") + e.what());
        move_group_b.reset();
      }
    }
    emit("Planner (actual):");
    emit("  " + planner_print);

    CollisionObs collision_obs;
    bool attachment_all_pass = true;
    bool any_relative_changed = false;
    std::string verdict = "FAIL";
    std::string verdict_line = "DUAL-6 ARM B TIME-FIRST OPTIMIZATION FAIL";

    if (!env_complete || !live_acm_check.complete)
    {
      verdict = "INCOMPLETE";
      verdict_line = "DUAL-6 INCOMPLETE";
      emit("");
      emit("PlanningScene / live ACM incomplete. Official PASS is not possible.");
    }
    else if (!move_group_b)
    {
      verdict = "BLOCKED";
      verdict_line = "DUAL-6 BLOCKED";
      emit("");
      emit("Existing /move_group unavailable.");
    }
    else if (dual4d_ok.empty())
    {
      verdict = "INCOMPLETE";
      verdict_line = "DUAL-6 INCOMPLETE";
      emit("");
      emit("DUAL-4E candidate cannot be revalidated: no DUAL-4D-compatible handover start.");
    }
    else
    {
      std::vector<Dual4dEndCandidate> start_ok;
      std::vector<std::string> start_fail;
      for (const auto& cand : dual4d_ok)
      {
        std::string error;
        auto scene = makeTransferredScene(local, geo, gripper_model_a, gripper_model_b, cand.joints_a,
                                          cand.joints_b, q_a_open, q_b_recv, object_pos_tol,
                                          object_ori_tol, error);
        if (!scene)
        {
          start_fail.push_back(cand.label + " rebuild failed: " + error);
          continue;
        }
        HeldStateCheck st = evaluateHeldBState(
            *scene, geo, gripper_model_a, gripper_model_b, home_joints, cand.joints_b, q_a_open,
            q_b_recv, object_pos_tol, object_ori_tol, face_center_tol, face_normal_tol, max_contacts,
            max_per_pair, nullptr);
        accumulateCollisionObs(collision_obs, st.diag);
        if (!st.relative_attach_ok)
          any_relative_changed = true;
        if (st.ok)
        {
          emit("START STATE VALID: " + cand.label);
          start_ok.push_back(cand);
        }
        else
        {
          start_fail.push_back(cand.label + " " + st.fail_reason);
        }
      }
      emit("");
      emit("----------------------------------------");
      emit("HANDOVER START CANDIDATES");
      emit("----------------------------------------");
      emit("");
      emit("DUAL-4E-compatible:");
      emit("  " + std::to_string(dual4d_ok.size()));
      emit("Tested:");
      emit("  " + std::to_string(dual4d_ok.size()));
      emit("START STATE VALID:");
      emit("  " + std::to_string(start_ok.size()));
      if (!start_fail.empty())
      {
        emit("Start failures:");
        for (const auto& line : start_fail)
          emit("  " + line);
      }

      std::vector<std::vector<InspectionIk>> layer_all(3);
      std::vector<std::vector<InspectionIk>> layer_valid(3);
      std::vector<int> raw_ik(3, 0);
      if (!start_ok.empty())
      {
        std::string error;
        auto ik_scene = makeTransferredScene(local, geo, gripper_model_a, gripper_model_b,
                                             start_ok.front().joints_a, start_ok.front().joints_b,
                                             q_a_open, q_b_recv, object_pos_tol, object_ori_tol, error);
        if (!ik_scene)
        {
          emit("Inspection IK scene rebuild failed: " + error);
        }
        else
        {
          moveit::core::RobotState ik_state(ik_scene->getCurrentState());
          applyFixedArmsAndGrippers(ik_state, gripper_model_a, gripper_model_b, home_joints,
                                    start_ok.front().joints_b, q_a_open, q_b_recv, error);
          ik_state.update();
          std::vector<std::pair<std::string, std::vector<double>>> seeds_b;
          seeds_b.push_back({"current", jointsOf(ik_state, kArmBJoints)});
          for (size_t i = 0; i < start_ok.size(); ++i)
            seeds_b.push_back({"handover_" + start_ok[i].label, start_ok[i].joints_b});

          emit("");
          emit("----------------------------------------");
          emit("IK CANDIDATES");
          emit("----------------------------------------");
          for (size_t t = 0; t < targets.size(); ++t)
          {
            TargetResult search;
            search.label = "Inspection " + std::to_string(targets[t].index) + " " + targets[t].name;
            search.group = kGroupB;
            search.tcp = kTcpB;
            search.target = targets[t].tcp_b_target;
            describeSolver(node, model, kGroupB, kTcpB, search);
            searchIk(search, ik_state, seeds_b, max_attempts, max_ik_per_target, timeout_exact,
                     timeout_nearby, nearby_radius, nearby_radius_local, min_distance, pos_tol, ori_tol,
                     rng_seed + 100 * static_cast<unsigned int>(t + 1));
            raw_ik[t] = search.ik_success;
            int vid = 0;
            int iid = 0;
            for (const auto& cand : search.unique)
            {
              InspectionIk classified = classifyInspectionIk(
                  *ik_scene, geo, gripper_model_a, gripper_model_b, home_joints, cand, targets[t],
                  q_a_open, q_b_recv, object_pos_tol, object_ori_tol, face_center_tol, face_normal_tol,
                  max_contacts, max_per_pair);
              classified.id = ++iid;
              accumulateCollisionObs(collision_obs, classified.diag);
              layer_all[t].push_back(classified);
              if (classified.status == "VALID")
              {
                classified.id = ++vid;
                layer_valid[t].push_back(classified);
              }
            }
            printInspectionIkSummary(targets[t], raw_ik[t], layer_all[t], layer_valid[t]);
            for (const auto& item : layer_valid[t])
            {
              emit("  VALID I" + std::to_string(targets[t].index) + "_" + std::to_string(item.id) +
                   " joints=" + fmtVec(item.ik.joints));
              emit("    TCP FK pos=" + fmtScalar(item.ik.fk_position_error) +
                   " m ori=" + fmtScalar(item.ik.fk_orientation_error_deg) + " deg");
              emit("    object pos=" + fmtScalar(item.object_pos_error) +
                   " m ori=" + fmtScalar(item.object_ori_error_deg) + " deg");
              emit("    face center=" + fmtScalar(item.face_center_error) +
                   " m normal=" + fmtScalar(item.face_normal_error_deg) + " deg");
              if (targets[t].name == "top_circle")
              {
                emit("    Original top face center error vs P1: " +
                     fmtScalar(item.face_center_error) + " m");
                emit("    Original top face directed normal error vs D1: " +
                     fmtScalar(item.face_normal_error_deg) + " deg");
                emit("    Original top face aligned: YES");
              }
            }
          }
        }
      }

      emit("");
      emit("----------------------------------------");
      emit("CORRECTED I3 ORIGINAL TOP FACE CHECK");
      emit("----------------------------------------");
      emit("Original top face center in object:");
      emit("  " + fmtXyz(targets[2].center_in_object));
      emit("Target P1:");
      emit("  " + fmtXyz(targets[2].p1));
      emit("Original top face normal in object:");
      emit("  " + fmtXyz(targets[2].normal_in_object));
      emit("Target D1:");
      emit("  " + fmtXyz(targets[2].d1));
      if (layer_valid[2].empty())
      {
        emit("Face center error:");
        emit("  (no valid I3 IK to evaluate)");
        emit("Directed normal error:");
        emit("  (no valid I3 IK to evaluate)");
        emit("Original top face aligned:");
        emit("  NO");
      }
      else
      {
        const auto& item = layer_valid[2].front();
        emit("Face center error:");
        emit("  " + fmtScalar(item.face_center_error) + " m");
        emit("Directed normal error:");
        emit("  " + fmtScalar(item.face_normal_error_deg) + " deg");
        emit("Original top face aligned:");
        emit("  YES");
      }

      emit("");
      emit("----------------------------------------");
      emit("EDGE SEARCH");
      emit("----------------------------------------");
      emit("Time parameterization method:");
      emit("  TimeOptimalTrajectoryGeneration (offline)");
      emit("  path_tolerance=0.01 rad resample_dt=0.01 s scaling=1.0");
      emit("Inspection dwell time:");
      emit("  NOT INCLUDED");
      emit("OMPL attempts per edge:");
      emit("  " + std::to_string(max_plan_attempts));
      emit("Time near-optimal ratio:");
      emit("  " + fmtScalar(time_near_ratio));

      int h_i1_tested = 0;
      int h_i1_valid = 0;
      int i1_i2_tested = 0;
      int i1_i2_valid = 0;
      int i2_i3_tested = 0;
      int i2_i3_valid = 0;
      int ompl_success_paths = 0;
      int collision_validated_paths = 0;
      int time_parameterized_paths = 0;
      int complete_endpoint_combos = 0;
      std::map<std::tuple<int, int, int, int>, SegmentPlan> edge_cache;
      auto usable_paths = [](const SegmentPlan& seg) {
        std::vector<TimedPath> out;
        for (const auto& p : seg.timed_paths)
        {
          if (p.ok && p.time_parameterized && p.collision_validated)
            out.push_back(p);
        }
        return out;
      };
      auto plan_edge = [&](int from_layer, int from_id, int to_layer, int to_id,
                           const Dual4dEndCandidate& handover, const std::vector<double>& start_b,
                           const std::vector<double>& goal_b, const InspectionTarget* end_target,
                           int& tested, int& valid) -> SegmentPlan {
        const auto key = std::make_tuple(from_layer, from_id, to_layer, to_id);
        auto it = edge_cache.find(key);
        if (it != edge_cache.end())
          return it->second;
        ++tested;
        emit("");
        emit("Planning edge L" + std::to_string(from_layer) + "_" + std::to_string(from_id) +
             " -> L" + std::to_string(to_layer) + "_" + std::to_string(to_id));
        SegmentPlan rec = planArmBSegment(
            *move_group_b, local, geo, gripper_model_a, gripper_model_b, handover, home_joints,
            start_b, goal_b, end_target, q_a_open, q_b_recv, max_plan_attempts, planning_time,
            max_validation_step, object_pos_tol, object_ori_tol, face_center_tol, face_normal_tol,
            chain_tol, max_contacts, max_per_pair, collision_obs, model, arm_b_limits, direction_eps,
            reject_gratuitous_j6);
        if (!rec.request_attachment.pass)
          attachment_all_pass = false;
        ompl_success_paths += rec.successful_attempts;
        collision_validated_paths += rec.successful_attempts;
        time_parameterized_paths += rec.time_parameterized_attempts;
        if (usable_paths(rec).empty())
        {
          rec.ok = false;
          emit("  edge FAIL: " + rec.fail_reason);
        }
        else
        {
          rec.ok = true;
          ++valid;
          emit("  edge VALID timed_paths=" + std::to_string(usable_paths(rec).size()) +
               " / attempts=" + std::to_string(rec.attempts));
        }
        edge_cache.emplace(key, rec);
        return rec;
      };

      std::vector<CompleteChain> chains;
      std::map<int, int> handover_complete;
      const bool i3_has_valid_ik = !layer_valid[2].empty();
      if (!arm_b_limits.complete)
      {
        emit("");
        emit("Skipping OMPL layered search: TIME OPTIMIZATION LIMITS INCOMPLETE");
        verdict = "INCOMPLETE";
        verdict_line = "DUAL-6 INCOMPLETE";
      }
      else if (!i3_has_valid_ik)
      {
        emit("");
        emit("Corrected I3 (ORIGINAL_TOP_CIRCLE) has 0 valid IK among tested unique solutions.");
        emit("Skipping OMPL layered search.");
        if (verdict != "INCOMPLETE" && verdict != "BLOCKED" && !start_ok.empty())
        {
          verdict = "FAIL";
          verdict_line = "DUAL-6 STATIC TARGET FAIL";
        }
      }
      else if (!layer_valid[0].empty() && !layer_valid[1].empty())
      {
        for (size_t h = 0; h < start_ok.size(); ++h)
        {
          for (const auto& i1 : layer_valid[0])
          {
            SegmentPlan e1 = plan_edge(0, static_cast<int>(h), 1, i1.id, start_ok[h],
                                       start_ok[h].joints_b, i1.ik.joints, &targets[0], h_i1_tested,
                                       h_i1_valid);
            for (const auto& i2 : layer_valid[1])
            {
              SegmentPlan e2 =
                  plan_edge(1, i1.id, 2, i2.id, start_ok[h], i1.ik.joints, i2.ik.joints, &targets[1],
                            i1_i2_tested, i1_i2_valid);
              for (const auto& i3 : layer_valid[2])
              {
                SegmentPlan e3 =
                    plan_edge(2, i2.id, 3, i3.id, start_ok[h], i2.ik.joints, i3.ik.joints,
                              &targets[2], i2_i3_tested, i2_i3_valid);
                ++complete_endpoint_combos;
                if (!e1.ok || !e2.ok || !e3.ok)
                  continue;
                const auto p1s = usable_paths(e1);
                const auto p2s = usable_paths(e2);
                const auto p3s = usable_paths(e3);
                for (size_t a = 0; a < p1s.size(); ++a)
                {
                  for (size_t b = 0; b < p2s.size(); ++b)
                  {
                    if (maxAbsDiff(p1s[a].waypoints.back(), p2s[b].waypoints.front()) > chain_tol)
                    {
                      emit("  CHAIN DISCONTINUITY H-I1/I1-I2");
                      continue;
                    }
                    for (size_t c = 0; c < p3s.size(); ++c)
                    {
                      if (maxAbsDiff(p2s[b].waypoints.back(), p3s[c].waypoints.front()) > chain_tol)
                      {
                        emit("  CHAIN DISCONTINUITY I1-I2/I2-I3");
                        continue;
                      }
                      CompleteChain chain;
                      chain.handover = start_ok[h];
                      chain.i1 = i1;
                      chain.i2 = i2;
                      chain.i3 = i3;
                      chain.h_to_i1 = e1;
                      chain.i1_to_i2 = e2;
                      chain.i2_to_i3 = e3;
                      chain.p1 = p1s[a];
                      chain.p2 = p2s[b];
                      chain.p3 = p3s[c];
                      chain.path_choice1 = static_cast<int>(a);
                      chain.path_choice2 = static_cast<int>(b);
                      chain.path_choice3 = static_cast<int>(c);
                      chain.metrics = combineChainMetrics(p1s[a], p2s[b], p3s[c], arm_b_limits,
                                                          direction_eps, reject_gratuitous_j6);
                      chain.total_length = chain.metrics.path_total;
                      chain.time_ok = chain.metrics.within_velocity_limits &&
                                      chain.metrics.within_acceleration_limits &&
                                      !chain.metrics.gratuitous_j6;
                      chain.seam_ok = true;
                      if (chain.time_ok)
                      {
                        chains.push_back(chain);
                        ++handover_complete[static_cast<int>(h)];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      emit("");
      emit("Handover -> Inspection1:");
      emit("  tested: " + std::to_string(h_i1_tested));
      emit("  valid timed: " + std::to_string(h_i1_valid));
      emit("Inspection1 -> Inspection2:");
      emit("  tested: " + std::to_string(i1_i2_tested));
      emit("  valid timed: " + std::to_string(i1_i2_valid));
      emit("Inspection2 -> Inspection3:");
      emit("  tested: " + std::to_string(i2_i3_tested));
      emit("  valid timed: " + std::to_string(i2_i3_valid));

      emit("");
      emit("----------------------------------------");
      emit("SEARCH SUMMARY");
      emit("----------------------------------------");
      emit("");
      emit("Handover candidates:");
      emit("  " + std::to_string(start_ok.size()));
      emit("I1 candidates:");
      emit("  " + std::to_string(layer_valid[0].size()));
      emit("I2 candidates:");
      emit("  " + std::to_string(layer_valid[1].size()));
      emit("I3 candidates:");
      emit("  " + std::to_string(layer_valid[2].size()));
      emit("Complete chains considered (endpoint combinations):");
      emit("  " + std::to_string(complete_endpoint_combos));
      emit("OMPL successful paths:");
      emit("  " + std::to_string(ompl_success_paths));
      emit("Collision validated paths:");
      emit("  " + std::to_string(collision_validated_paths));
      emit("Time-parameterized paths:");
      emit("  " + std::to_string(time_parameterized_paths));
      emit("Complete time-parameterized chains:");
      emit("  " + std::to_string(chains.size()));

      std::sort(chains.begin(), chains.end(), [](const CompleteChain& a, const CompleteChain& b) {
        return a.metrics.t_total < b.metrics.t_total;
      });

      const CompleteChain* fastest = nullptr;
      const CompleteChain* preferred = nullptr;
      std::vector<CompleteChain> near_set;
      if (!chains.empty())
      {
        fastest = &chains.front();
        const double t_min = fastest->metrics.t_total;
        const double t_cap = t_min * (1.0 + time_near_ratio);
        for (const auto& c : chains)
        {
          if (c.metrics.t_total <= t_cap)
            near_set.push_back(c);
        }
        auto better_preferred = [&](const CompleteChain& a, const CompleteChain& b) {
          if (prefer_low_j1)
          {
            if (a.metrics.travel[0] < b.metrics.travel[0] - 1e-9)
              return true;
            if (b.metrics.travel[0] < a.metrics.travel[0] - 1e-9)
              return false;
            if (a.metrics.reversals[0] < b.metrics.reversals[0])
              return true;
            if (b.metrics.reversals[0] < a.metrics.reversals[0])
              return false;
          }
          if (prefer_meaningful_j6)
          {
            const bool a_bad_j6 =
                a.metrics.gratuitous_j6 ||
                (a.metrics.reversals[5] >= 2 &&
                 a.metrics.travel[5] > 2.0 * std::fabs(a.metrics.net[5]) + 0.1);
            const bool b_bad_j6 =
                b.metrics.gratuitous_j6 ||
                (b.metrics.reversals[5] >= 2 &&
                 b.metrics.travel[5] > 2.0 * std::fabs(b.metrics.net[5]) + 0.1);
            if (!a_bad_j6 && b_bad_j6)
              return true;
            if (a_bad_j6 && !b_bad_j6)
              return false;
            if (!a_bad_j6 && !b_bad_j6 && a.metrics.j6_share > b.metrics.j6_share + 1e-9)
              return true;
            if (!a_bad_j6 && !b_bad_j6 && b.metrics.j6_share > a.metrics.j6_share + 1e-9)
              return false;
          }
          return a.metrics.t_total < b.metrics.t_total;
        };
        preferred = &near_set.front();
        for (const auto& c : near_set)
        {
          if (better_preferred(c, *preferred))
            preferred = &c;
        }
      }

      auto print_joint_block = [&](const ChainMetrics& m) {
        emit("Joint cumulative travel:");
        emit("");
        emit("J1:");
        emit("  " + fmtScalar(m.travel[0]) + " rad");
        emit("J2:");
        emit("  " + fmtScalar(m.travel[1]) + " rad");
        emit("J3:");
        emit("  " + fmtScalar(m.travel[2]) + " rad");
        emit("J4:");
        emit("  " + fmtScalar(m.travel[3]) + " rad");
        emit("J5:");
        emit("  " + fmtScalar(m.travel[4]) + " rad");
        emit("J6:");
        emit("  " + fmtScalar(m.travel[5]) + " rad");
        emit("");
        emit("J1 reversals:");
        emit("  " + std::to_string(m.reversals[0]));
        emit("J6 reversals:");
        emit("  " + std::to_string(m.reversals[5]));
        emit("J6 net change:");
        emit("  " + fmtScalar(m.net[5]) + " rad");
        emit("J6 share:");
        emit("  " + fmtScalar(m.j6_share));
        emit("Gratuitous J6 full turn:");
        emit(std::string("  ") + (m.gratuitous_j6 ? "YES" : "NO"));
      };
      auto print_peaks = [&](const ChainMetrics& m) {
        emit("");
        emit("Peak velocity / acceleration vs RobotModel limits:");
        for (size_t j = 0; j < 6 && j < arm_b_limits.joints.size(); ++j)
        {
          emit("  " + kArmBJoints[j] + ":");
          emit("    peak velocity: " + fmtScalar(m.peak_vel[j]) +
               " / limit " + fmtScalar(arm_b_limits.joints[j].max_velocity) + " rad/s");
          emit("    peak acceleration: " + fmtScalar(m.peak_acc[j]) +
               " / limit " + fmtScalar(arm_b_limits.joints[j].max_acceleration) + " rad/s^2");
        }
        emit("within velocity limits:");
        emit(std::string("  ") + (m.within_velocity_limits ? "YES" : "NO"));
        emit("within acceleration limits:");
        emit(std::string("  ") + (m.within_acceleration_limits ? "YES" : "NO"));
      };
      auto print_chain_identity = [&](const CompleteChain& c) {
        emit("Handover label:");
        emit("  " + c.handover.label);
        emit("Handover joints:");
        emit("  " + fmtVec(c.handover.joints_b, 12));
        emit("I1 joints:");
        emit("  " + fmtVec(c.i1.ik.joints, 12));
        emit("I2 joints:");
        emit("  " + fmtVec(c.i2.ik.joints, 12));
        emit("I3 joints:");
        emit("  " + fmtVec(c.i3.ik.joints, 12));
        emit("H -> I1:");
        emit("  duration: " + fmtScalar(c.metrics.t1) + " s");
        emit("  path length: " + fmtScalar(c.metrics.path1) + " rad");
        emit("I1 -> I2:");
        emit("  duration: " + fmtScalar(c.metrics.t2) + " s");
        emit("  path length: " + fmtScalar(c.metrics.path2) + " rad");
        emit("I2 -> I3:");
        emit("  duration: " + fmtScalar(c.metrics.t3) + " s");
        emit("  path length: " + fmtScalar(c.metrics.path3) + " rad");
        emit("Total motion duration:");
        emit("  " + fmtScalar(c.metrics.t_total) + " s");
      };

      emit("");
      emit("----------------------------------------");
      emit("ABSOLUTE FASTEST TESTED");
      emit("----------------------------------------");
      if (!fastest)
      {
        emit("  (none)");
      }
      else
      {
        emit("");
        print_chain_identity(*fastest);
        emit("");
        print_joint_block(fastest->metrics);
        print_peaks(fastest->metrics);
      }

      emit("");
      emit("Time-near-optimal set (T <= T_min * " + fmtScalar(1.0 + time_near_ratio) + "):");
      emit("  " + std::to_string(near_set.size()));

      emit("");
      emit("----------------------------------------");
      emit("PREFERRED TIME-FIRST");
      emit("----------------------------------------");
      emit("");
      emit("Selection logic:");
      emit("");
      emit("time:");
      emit("  first");
      emit("J1:");
      emit("  minimized within time-near-optimal set");
      emit("J6:");
      emit("  preferred only without gratuitous motion");
      emit("");
      if (!preferred || !fastest)
      {
        emit("  (none)");
      }
      else
      {
        const bool same = (preferred == fastest) ||
                          (maxAbsDiff(preferred->handover.joints_b, fastest->handover.joints_b) <= 1e-9 &&
                           maxAbsDiff(preferred->i1.ik.joints, fastest->i1.ik.joints) <= 1e-9 &&
                           maxAbsDiff(preferred->i2.ik.joints, fastest->i2.ik.joints) <= 1e-9 &&
                           maxAbsDiff(preferred->i3.ik.joints, fastest->i3.ik.joints) <= 1e-9 &&
                           std::abs(preferred->metrics.t_total - fastest->metrics.t_total) <= 1e-9 &&
                           std::abs(preferred->metrics.travel[0] - fastest->metrics.travel[0]) <= 1e-9);
        if (same)
        {
          emit("PREFERRED TIME-FIRST = ABSOLUTE FASTEST TESTED");
          emit("");
        }
        print_chain_identity(*preferred);
        emit("");
        const double t_pen =
            (preferred->metrics.t_total - fastest->metrics.t_total) /
            std::max(1e-12, fastest->metrics.t_total) * 100.0;
        const double j1_red = fastest->metrics.travel[0] - preferred->metrics.travel[0];
        const double j1_red_pct =
            j1_red / std::max(1e-12, fastest->metrics.travel[0]) * 100.0;
        emit("Time penalty vs absolute fastest:");
        emit("  " + fmtScalar(t_pen) + " %");
        emit("J1 travel:");
        emit("  " + fmtScalar(preferred->metrics.travel[0]) + " rad");
        emit("J1 reduction vs absolute fastest:");
        emit("  " + fmtScalar(j1_red) + " rad");
        emit("  " + fmtScalar(j1_red_pct) + " %");
        emit("J6 travel:");
        emit("  " + fmtScalar(preferred->metrics.travel[5]) + " rad");
        emit("J6 share:");
        emit("  " + fmtScalar(preferred->metrics.j6_share));
        emit("J6 reversals:");
        emit("  " + std::to_string(preferred->metrics.reversals[5]));
        emit("Gratuitous J6 rotation:");
        emit(std::string("  ") + (preferred->metrics.gratuitous_j6 ? "YES" : "NO"));
        emit("");
        print_joint_block(preferred->metrics);
        print_peaks(preferred->metrics);
        const double mid_fast = fastest->metrics.travel[1] + fastest->metrics.travel[2] +
                                fastest->metrics.travel[3] + fastest->metrics.travel[4];
        const double mid_pref = preferred->metrics.travel[1] + preferred->metrics.travel[2] +
                                preferred->metrics.travel[3] + preferred->metrics.travel[4];
        emit("");
        emit("J2-J5 travel vs absolute fastest:");
        emit("  fastest: " + fmtScalar(mid_fast) + " rad");
        emit("  preferred: " + fmtScalar(mid_pref) + " rad");
        emit("  extreme J2-J5 increase to hide J1:");
        emit(std::string("  ") + ((mid_pref > 1.5 * std::max(1e-9, mid_fast) && j1_red > 1e-6) ?
                                      "YES" :
                                      "NO"));
        emit("Full collision validation:");
        emit("  PASS");
      }

      if (layer_valid[2].empty() && !start_ok.empty() && arm_b_limits.complete)
      {
        if (verdict != "INCOMPLETE" && verdict != "BLOCKED")
        {
          verdict = "FAIL";
          verdict_line = "DUAL-6 STATIC TARGET FAIL";
        }
      }
      else if ((layer_valid[0].empty() || layer_valid[1].empty()) && arm_b_limits.complete)
      {
        if (verdict != "INCOMPLETE" && verdict != "BLOCKED")
        {
          verdict = "FAIL";
          verdict_line = "DUAL-6 MODEL SEQUENCE FAIL";
        }
      }
      else if (!arm_b_limits.complete)
      {
        verdict = "INCOMPLETE";
        verdict_line = "DUAL-6 INCOMPLETE";
      }
      else if (chains.empty())
      {
        if (verdict != "INCOMPLETE" && verdict != "BLOCKED")
        {
          verdict = "FAIL";
          verdict_line = "DUAL-6 ARM B TIME-FIRST OPTIMIZATION FAIL";
        }
      }
      else if (attachment_all_pass && !any_relative_changed && fastest && preferred &&
               fastest->metrics.within_velocity_limits && fastest->metrics.within_acceleration_limits &&
               preferred->metrics.within_velocity_limits &&
               preferred->metrics.within_acceleration_limits)
      {
        verdict = "PASS";
        verdict_line = "DUAL-6 ARM B TIME-FIRST OPTIMIZATION PASS";
      }
    }

    emit("");
    emit("========== DUAL-6 ARM B TIME-FIRST OPTIMIZATION ==========");
    emit("");
    emit("RobotModel:");
    emit("  fairino3_dual_robot");
    emit("PlanningScene:");
    emit(std::string("  ") +
         ((env_complete && live_acm_check.complete) ? "COMPLETE" : "INCOMPLETE"));
    emit("Arm A:");
    emit("  HOME / FIXED");
    emit("Arm B:");
    emit("  OPTIMIZED");
    emit("Object:");
    emit("  small_part attached to arm_b_gripper_tcp");
    emit("Inspection sequence:");
    emit("  H");
    emit("  -> side_pos_y");
    emit("  -> side_neg_y");
    emit("  -> original_top_circle");
    emit("Optimization scope:");
    emit("  ARM B ONLY");
    emit("Primary objective:");
    emit("  MINIMUM ESTIMATED MOTION TIME");
    emit("Secondary preference:");
    emit("  MINIMUM J1 CUMULATIVE TRAVEL");
    emit("Tertiary preference:");
    emit("  MEANINGFUL J6 ORIENTATION UTILIZATION");
    emit("Time parameterization:");
    emit("  TimeOptimalTrajectoryGeneration (offline)");
    emit("Inspection dwell:");
    emit("  NOT INCLUDED");
    emit("");
    emit("----------------------------------------");
    emit("FINAL");
    emit("----------------------------------------");
    emit("");
    emit("DUAL-6:");
    emit("  " + verdict);
    emit(verdict_line);
    emit("");
    emit("Attachment request check:");
    emit("  " + std::string(attachment_all_pass ? "PASS" : "FAIL"));
    emit("Full collision revalidation:");
    emit(std::string("  ") + (verdict == "PASS" ? "PASS" : (verdict == "INCOMPLETE" ? "NOT COMPLETE" : "FAIL")));
    emit("");
    emit("Optimization scope:");
    emit("  ARM B ONLY");
    emit("");
    emit("Primary objective:");
    emit("  TIME");
    emit("");
    emit("Secondary preference:");
    emit("  LOW J1 MOTION");
    emit("");
    emit("Tertiary preference:");
    emit("  MEANINGFUL J6 UTILIZATION");
    emit("");
    emit("Inspection geometry changed:");
    emit("  NO");
    emit("");
    emit("Handover geometry changed:");
    emit("  NO");
    emit("");
    emit("Arm A Home changed:");
    emit("  NO");
    emit("");
    emit("Attachment transform changed:");
    emit("  NO");
    emit("");
    emit("Joint limits changed:");
    emit("  NO");
    emit("");
    emit("Velocity limits changed:");
    emit("  NO");
    emit("");
    emit("Acceleration limits changed:");
    emit("  NO");
    emit("");
    emit("URDF/SRDF changed:");
    emit("  NO");
    emit("");
    emit("Robot mounting changed:");
    emit("  NO");
    emit("");
    emit("World geometry changed:");
    emit("  NO");
    emit("");
    emit("Original ACM changed:");
    emit("  NO");
    emit("");
    emit("Global PlanningScene write issued:");
    emit("  NO");
    emit("");
    emit("Time parameterization:");
    emit("  OFFLINE ONLY");
    emit("");
    emit("Real robot commands:");
    emit("  ZERO");
    emit("");
    emit("Real gripper commands:");
    emit("  ZERO");
    if (!arm_b_limits.complete)
    {
      emit("");
      emit("DUAL-6 INCOMPLETE");
      emit("TIME OPTIMIZATION LIMITS INCOMPLETE");
      emit("  " + arm_b_limits.missing);
    }
    else if (verdict == "FAIL")
    {
      emit("");
      emit("No complete time-parameterized inspection chain found");
      emit("with current planner settings and tested candidates.");
    }

    stop();
    if (verdict == "PASS")
      return 0;
    if (verdict == "BLOCKED")
      return 4;
    return verdict == "INCOMPLETE" ? 3 : 2;
  }
  catch (const std::exception& e)
  {
    emit(std::string("STAGE exception: ") + e.what());
    emit("DUAL-6 aborted with a caught exception. No robot commands were sent.");
    try
    {
      stop();
    }
    catch (...)
    {
    }
    return 1;
  }
}
