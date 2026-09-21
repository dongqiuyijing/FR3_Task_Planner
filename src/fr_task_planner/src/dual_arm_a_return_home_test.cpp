// DUAL-4E: Arm A Handover → existing Home while Arm B stays fixed holding small_part.
// Reuses DUAL-4C Cartesian + DUAL-4D transfer candidate search. Local PlanningScene only.
// PLAN ONLY. No execute, no gripper commands, no scene apply.
#include <algorithm>
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
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/attached_body.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_state/robot_state.h>
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
  emit("DUAL-4E: BLOCKED");
  emit("reason: " + why);
  emit("Return-home planning: NOT PERFORMED");
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

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("dual_arm_a_return_home_test", options);

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
    emit("========== DUAL-4E ARM A RETURN HOME ==========");
    emit("MODEL STATE VALIDATION ONLY. PLAN ONLY. NO EXECUTION. NO GRIPPER COMMANDS.");
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
    emit("DUAL-4C Cartesian revalidation complete. Entering DUAL-4E Arm A return-home.");

    const std::string step15_yaml = getString(
        node, "step15_winner_yaml",
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
            "/fr_task_ws/src/fr_task_planner/config/step15_optimized_lift_to_a_winner.yaml");
    const int max_plan_attempts = getInt(node, "max_planning_attempts", 5);
    const double planning_time = getDouble(node, "planning_time_sec", 10.0);
    const double max_validation_step = getDouble(node, "max_validation_joint_step", 0.02);
    const double home_tol = getDouble(node, "home_arrival_tol_rad", 1.0e-4);

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

    std::unique_ptr<MoveGroup> move_group_a;
    std::string planner_print = "OMPL / (MoveGroupInterface not created)";
    if (env_complete && live_acm_check.complete)
    {
      move_group_a = std::make_unique<MoveGroup>(node, "arm_a");
      move_group_a->setPlanningPipelineId("ompl");
      move_group_a->setPlanningTime(planning_time);
      move_group_a->setNumPlanningAttempts(1);
      const std::string planner_id = move_group_a->getPlannerId();
      const std::string pipeline_id = move_group_a->getPlanningPipelineId().empty() ?
                                          std::string("ompl") :
                                          move_group_a->getPlanningPipelineId();
      planner_print = pipeline_id + " / " + (planner_id.empty() ? std::string("default") : planner_id);
    }

    emit("");
    emit("========== DUAL-4E ARM A RETURN HOME ==========");
    emit("");
    emit("RobotModel:");
    emit("  fairino3_dual_robot");
    emit("");
    emit("PlanningScene:");
    emit(std::string("  ") +
         ((env_complete && live_acm_check.complete) ? "COMPLETE" : "INCOMPLETE"));
    emit("");
    emit("Planner:");
    emit("  " + planner_print);
    emit("");
    emit("Home source:");
    emit("  " + home_source);
    emit("");
    emit("Home joints:");
    emit("  " + fmtVec(home_joints, 12));
    emit("");
    emit("Object:");
    emit("  small_part -> arm_b_gripper_tcp");
    emit("");
    emit("Arm B:");
    emit("  FIXED");
    emit("");
    emit("Arm B gripper:");
    emit("  q = " + fmtScalar(q_b_recv) + " NOMINAL");
    emit("");
    emit("Arm A gripper:");
    emit("  q = " + fmtScalar(q_a_open) + " OPEN");

    emit("");
    emit("----------------------------------------");
    emit("START CANDIDATES");
    emit("----------------------------------------");
    emit("");
    emit("DUAL-4D-compatible candidates:");
    emit("  " + std::to_string(dual4d_ok.size()));
    for (const auto& cand : dual4d_ok)
    {
      emit("  " + cand.label);
      emit("    Arm A: " + fmtVec(cand.joints_a));
      emit("    Arm B: " + fmtVec(cand.joints_b));
    }
    emit("");
    emit("Candidates tested:");
    emit("  " + std::to_string(dual4d_ok.size()));

    std::vector<ReturnHomeResult> results;
    if (move_group_a)
    {
      for (const auto& cand : dual4d_ok)
      {
        ReturnHomeResult rec = planArmAReturnHome(
            *move_group_a, local, geo, gripper_model_a, gripper_model_b, cand, home_joints,
            q_a_open, q_b_recv, max_plan_attempts, planning_time, max_validation_step,
            object_pos_tol, object_ori_tol, home_tol, max_contacts, max_per_pair);
        results.push_back(std::move(rec));
      }
    }

    bool any_a_part = false;
    bool any_a_b = false;
    bool any_a_env = false;
    bool any_b_moved = false;
    bool any_object_jump = false;
    double shortest_length = std::numeric_limits<double>::infinity();
    int retained_count = 0;
    for (const auto& rec : results)
    {
      emit("");
      emit("----------------------------------------");
      emit("CANDIDATE " + rec.candidate.label);
      emit("----------------------------------------");
      emit("");
      emit("Arm A start:");
      emit("  " + fmtVec(rec.candidate.joints_a));
      emit("");
      emit("Arm B fixed:");
      emit("  " + fmtVec(rec.candidate.joints_b));
      emit("");
      emit("Start state valid:");
      emit(std::string("  ") + (rec.start_valid ? "YES" : "NO"));
      if (!rec.start_valid && !rec.start_reason.empty())
        emit("  " + rec.start_reason);
      emit("");
      emit("Goal state valid:");
      emit(std::string("  ") + (rec.goal_valid ? "YES" : "NO"));
      if (!rec.goal_valid)
      {
        emit(std::string("  ") + (rec.home_state_invalid ? "HOME STATE INVALID" :
                                                          "GOAL STATE INVALID"));
        if (!rec.goal_reason.empty())
          emit("  " + rec.goal_reason);
      }
      emit("");
      emit("Planning attempts:");
      emit("  " + std::to_string(rec.attempts));
      for (const auto& attempt : rec.attempt_log)
      {
        emit("  attempt " + std::to_string(attempt.index) + " planner=" + attempt.planner +
             " time=" + fmtScalar(attempt.planning_time_sec) + "s " +
             (attempt.planning_skipped ?
                  std::string("plan=SKIPPED REQUEST ATTACHMENT CHECK FAIL") :
                  (attempt.plan_success ? "plan=SUCCESS" : "plan=FAIL " + attempt.error_name)) +
             " waypoints=" + std::to_string(attempt.waypoint_count) +
             (attempt.validated ? " validated=YES" : " validated=NO"));
        if (!attempt.fail_reason.empty() && !attempt.validated)
        {
          emit("    reason: " + attempt.fail_reason);
          if (attempt.fail_waypoint >= 0)
          {
            emit("    trajectory segment: waypoint " + std::to_string(attempt.fail_waypoint) +
                 " interpolation sample " + std::to_string(attempt.fail_interp));
            emit("    Arm A joints: " + fmtVec(attempt.fail_joints_a));
            if (!attempt.collision_pair.empty())
              emit("    collision: " + attempt.collision_pair);
            if (!attempt.collision_category.empty() && attempt.collision_category != "none")
              emit("    collision category: " + attempt.collision_category);
          }
        }
        if (attempt.collision_category == "Arm A <-> small_part")
          any_a_part = true;
        if (attempt.collision_category == "Arm A <-> Arm B")
          any_a_b = true;
        if (attempt.collision_category == "Arm A <-> mounting_column" ||
            attempt.collision_category == "Arm A <-> table")
          any_a_env = true;
      }
      emit("");
      emit("Successful attempts:");
      emit("  " + std::to_string(rec.successful_attempts));
      emit("");
      emit("Selected geometric path:");
      if (rec.retained)
      {
        emit("  attempt " + std::to_string(rec.selected.index) + " " + rec.selected.planner);
        retained_count += 1;
        shortest_length = std::min(shortest_length, rec.selected.joint_path_length);
      }
      else if (!rec.start_valid)
        emit("  none (start invalid)");
      else if (!rec.goal_valid)
        emit("  none (HOME STATE INVALID)");
      else
        emit("  none (NO PATH FOUND WITH CURRENT PLANNER SETTINGS)");
      emit("");
      emit("Trajectory waypoints:");
      emit("  " + (rec.retained ? std::to_string(rec.selected.waypoint_count) : std::string("0")));
      emit("");
      emit("Validation samples:");
      emit("  " + (rec.retained ? std::to_string(rec.selected.validation_samples) : std::string("0")));
      emit("");
      emit("Arm B moved:");
      emit(std::string("  ") + ((rec.retained && !rec.selected.arm_b_moved) || !rec.retained ?
                                    (rec.retained ? "NO" : "N/A") :
                                    "YES"));
      if (rec.retained && rec.selected.arm_b_moved)
        any_b_moved = true;
      emit("");
      emit("Object pose changed:");
      emit(std::string("  ") + ((rec.retained && !rec.selected.object_pose_changed) || !rec.retained ?
                                    (rec.retained ? "NO" : "N/A") :
                                    "YES"));
      if (rec.retained && rec.selected.object_pose_changed)
        any_object_jump = true;
      emit("");
      emit("Illegal collision:");
      emit(std::string("  ") + (rec.retained && !rec.selected.illegal_collision ? "NO" :
                                (rec.retained ? "YES" : "N/A")));
      emit("");
      emit("Arm A joint path length:");
      emit("  " + (rec.retained ? fmtScalar(rec.selected.joint_path_length) + " rad" :
                                  std::string("n/a")));
      emit("");
      emit("Final Home error:");
      emit("  " + (rec.retained ? fmtScalar(rec.selected.home_error, 8) + " rad" :
                                  std::string("n/a")));
      emit("  tolerance used: " + fmtScalar(home_tol, 8) + " rad");
    }

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
    emit("After world IDs: " + (after_ok ? joinNames(after_global.world_ids) : after_error));
    emit("After attached IDs: " + (after_ok ? joinNames(after_global.attached_ids) : std::string("?")));
    emit(std::string("Global scene changed: ") + (global_same ? "NO" : "YES"));
    if (!global_same)
    {
      if (after_ok)
        printSnapshotDiff(before_global, after_global);
      emit("Cause: UNDETERMINED");
    }

    RequestAttachmentDiag report_diag;
    for (const auto& rec : results)
    {
      if (rec.request_attachment.local_checked && !report_diag.local_checked)
        report_diag = rec.request_attachment;
      if (!rec.request_attachment.request_constructed)
        continue;
      if (!report_diag.request_constructed)
        report_diag = rec.request_attachment;
      if (!rec.request_attachment.pass)
        report_diag = rec.request_attachment;
    }
    const std::string global_attached_line =
        "before=" + joinNames(before_global.attached_ids) +
        " after=" + (after_ok ? joinNames(after_global.attached_ids) : std::string("?"));
    emitRequestAttachmentCheck(report_diag, global_attached_line, live_acm_check.a_status,
                               live_acm_check.b_status);

    emit("");
    emit("----------------------------------------");
    emit("RETAINED CANDIDATES");
    emit("----------------------------------------");
    int printed = 0;
    for (const auto& rec : results)
    {
      if (!rec.retained)
        continue;
      ++printed;
      emit("");
      emit("candidate " + std::to_string(printed) + ":");
      emit("  " + rec.candidate.label);
      emit("  Arm A start: " + fmtVec(rec.candidate.joints_a));
      emit("  Arm B fixed: " + fmtVec(rec.candidate.joints_b));
      emit("  waypoints: " + std::to_string(rec.selected.waypoint_count));
      emit("  validation samples: " + std::to_string(rec.selected.validation_samples));
      emit("  Arm A joint path length: " + fmtScalar(rec.selected.joint_path_length) + " rad");
      emit("  planning time: " + fmtScalar(rec.selected.planning_time_sec) + " s");
      emit("  final Home error: " + fmtScalar(rec.selected.home_error, 8) + " rad");
    }
    if (printed == 0)
      emit("  (none)");

    const bool scene_complete = env_complete && live_acm_check.complete && object_pose_pass &&
                                gripper.opening_found && gripper.finger_collision_geometry &&
                                !any_truncated && !path_truncated && global_same && after_ok;
    const bool any_home_invalid =
        !results.empty() &&
        std::all_of(results.begin(), results.end(),
                    [](const ReturnHomeResult& rec) { return rec.home_state_invalid; });
    const bool any_start_invalid =
        !results.empty() &&
        std::all_of(results.begin(), results.end(),
                    [](const ReturnHomeResult& rec) { return !rec.start_valid; });

    std::string verdict = "INCOMPLETE";
    std::string verdict_line = "DUAL-4E INCOMPLETE";
    std::string return_home_line = "NOT VALIDATED";
    if (!move_group_ok || !after_ok)
    {
      verdict = "BLOCKED";
      verdict_line = "DUAL-4E BLOCKED";
    }
    else if (!scene_complete || !home_loaded || !live_acm_check.complete || !env_complete)
    {
      verdict = "INCOMPLETE";
      verdict_line = "DUAL-4E INCOMPLETE";
    }
    else if (retained.empty())
    {
      verdict = "INCOMPLETE";
      verdict_line = "DUAL-4E INCOMPLETE";
      emit("DUAL-4C/4D starting candidates could not be revalidated.");
    }
    else if (dual4d_ok.empty())
    {
      verdict = "INCOMPLETE";
      verdict_line = "DUAL-4E INCOMPLETE";
      emit("No DUAL-4D-compatible candidate survived revalidation.");
    }
    else if (!move_group_a)
    {
      verdict = "BLOCKED";
      verdict_line = "DUAL-4E BLOCKED";
    }
    else if (any_home_invalid && retained_count == 0)
    {
      verdict = "INCOMPLETE";
      verdict_line = "DUAL-4E INCOMPLETE";
      emit("HOME STATE INVALID in the current dual-arm attached scene.");
    }
    else if (any_start_invalid && retained_count == 0)
    {
      verdict = "INCOMPLETE";
      verdict_line = "DUAL-4E INCOMPLETE";
      emit("START STATE INVALID; cannot attribute failure to OMPL.");
    }
    else if (retained_count == 0)
    {
      verdict = "FAIL";
      verdict_line = "DUAL-4E MODEL PATH FAIL";
      return_home_line = "NOT VALIDATED";
    }
    else
    {
      verdict = "PASS";
      verdict_line = "DUAL-4E MODEL PATH PASS";
      return_home_line = "MODEL PATH VALIDATED";
    }

    emit("");
    emit("----------------------------------------");
    emit("FINAL");
    emit("----------------------------------------");
    emit("");
    emit("DUAL-4E:");
    emit("  " + verdict);
    emit(verdict_line);
    emit("");
    emit("Arm A return Home:");
    emit("  " + return_home_line);
    emit("");
    emit("Arm B:");
    emit(std::string("  ") + (any_b_moved ? "MOVED (FAIL)" : "FIXED THROUGHOUT"));
    emit("");
    emit("Object:");
    emit(std::string("  ") + (any_object_jump ? "POSE CHANGED (FAIL)" : "ATTACHED TO ARM B THROUGHOUT"));
    emit("");
    emit("Time parameterization:");
    emit("  NOT PERFORMED");
    emit("");
    emit("Arm B inspection:");
    emit("  NOT STARTED");
    emit("");
    emit("Real robot commands:");
    emit("  ZERO");
    emit("");
    emit("Real gripper commands:");
    emit("  ZERO");
    emit("");
    emit("Handover geometry changed:");
    emit("  NO");
    emit("Arm B pose changed:");
    emit("  NO");
    emit("Arm B X offset changed:");
    emit("  NO");
    emit("URDF/SRDF changed:");
    emit("  NO");
    emit("World geometry changed:");
    emit("  NO");
    emit("Original ACM changed:");
    emit("  NO");
    emit("Global PlanningScene write issued:");
    emit("  NO");
    if (retained_count == 0 && verdict == "FAIL")
    {
      emit("");
      emit("No validated Arm A return-home path found");
      emit("with current planner settings and tested candidates.");
    }
    emit("");
    emit("Arm A <-> small_part departure collision observed:");
    emit(std::string("  ") + (any_a_part ? "YES" : "NO"));
    emit("Arm A <-> Arm B collision observed:");
    emit(std::string("  ") + (any_a_b ? "YES" : "NO"));
    emit("Arm A <-> column/table collision observed:");
    emit(std::string("  ") + (any_a_env ? "YES" : "NO"));
    emit("Shortest Arm A joint-space path length:");
    emit("  " + (retained_count > 0 ? fmtScalar(shortest_length) + " rad" : std::string("n/a")));
    emit("Retained return-home candidates:");
    emit("  " + std::to_string(retained_count) + " / " + std::to_string(dual4d_ok.size()));

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
    emit("DUAL-4E aborted with a caught exception. No robot commands were sent.");
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
