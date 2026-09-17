#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/robot_state.h>
#include <rclcpp/logger.hpp>

namespace fr_task_planner
{
const std::vector<std::string> kArmJoints = { "j1", "j2", "j3", "j4", "j5", "j6" };
const std::vector<std::string> kArmBodyLinks = { "base_link",    "shoulder_link", "upperarm_link",
                                                 "forearm_link", "wrist1_link",   "wrist2_link",
                                                 "wrist3_link" };

struct ViewGeom
{
  std::string name;
  Eigen::Vector3d center_in_object = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal_in_object = Eigen::Vector3d::Zero();
  Eigen::Vector3d up_in_object = Eigen::Vector3d::Zero();
};

struct RollPose
{
  std::string view;
  double roll_deg = 0.0;
  int pose_index = 0;
  geometry_msgs::msg::PoseStamped object;
  geometry_msgs::msg::PoseStamped tcp;
};

struct EndpointCandidate
{
  std::string view_name;
  double roll_deg = 0.0;
  int pose_index = 0;
  int ik_index = 0;
  std::string candidate_id;
  std::string digest;
  std::map<std::string, double> joints;
  geometry_msgs::msg::PoseStamped tcp_target;
  geometry_msgs::msg::PoseStamped object_target;
  double up_error_deg = 0.0;
  double view_center_error = 0.0;
  double normal_error = 0.0;
  double tcp_object_error = 0.0;
  double joint_distance_from_lift = 0.0;
  bool bounds_ok = false;
  bool collision_free = false;
  bool attached_object_present = false;
  bool valid = false;
  std::string failure_reason = "ok";
};

struct EndpointGenerationConfig
{
  std::string group = "fairino3_v6_group";
  std::string ee_link = "gripper_tcp";
  std::string object_id = "small_part";
  std::string table_name = "table";
  std::vector<std::string> touch_links;
  Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
  Eigen::Vector3d d1 = Eigen::Vector3d::UnitY();
  Eigen::Vector3d preferred_up = Eigen::Vector3d::UnitZ();
  Eigen::Isometry3d t_tcp_object = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d t_model_base = Eigen::Isometry3d::Identity();
  double pos_tol = 0.005;
  double ori_tol_deg = 3.0;
  uint32_t max_ik_solutions_per_pose = 8;
  double min_ik_solution_distance = 0.1;
};

struct EndpointGenerationResult
{
  std::vector<EndpointCandidate> raw;
  std::vector<EndpointCandidate> valid;
};

Eigen::Isometry3d poseToIso(const geometry_msgs::msg::Pose& pose);
geometry_msgs::msg::Pose isoToPose(const Eigen::Isometry3d& transform);
void poseError(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b, double& position_m,
               double& orientation_deg);
double jointL2(const std::map<std::string, double>& a, const std::map<std::string, double>& b);
double maxJointError(const std::map<std::string, double>& a, const std::map<std::string, double>& b);
std::map<std::string, double> jointsFromState(const moveit::core::RobotState& state);
void applyJoints(moveit::core::RobotState& state, const std::map<std::string, double>& joints);
Eigen::Isometry3d tcpInBase(const moveit::core::RobotState& state, const std::string& ee_link);

std::string formatRollDeg(double roll_deg);
std::string makeCandidateId(const std::string& view, double roll_deg, int ik_index);
std::string makeCandidateDigest(const std::string& view, double roll_deg,
                                const std::map<std::string, double>& joints);
std::string makeStageName(const EndpointCandidate& cand);

void sortEndpointCandidates(std::vector<EndpointCandidate>& cands);
std::vector<RollPose> filterRollsByView(const std::vector<RollPose>& rolls, const std::string& view);
std::vector<EndpointCandidate> capCandidates(const std::vector<EndpointCandidate>& valid,
                                             int max_endpoint_candidates);

EndpointGenerationResult generateValidEndpointCandidates(
    const planning_scene::PlanningScene& lift_scene, const std::vector<RollPose>& rolls,
    const ViewGeom& view, const EndpointGenerationConfig& cfg, const rclcpp::Logger& logger);
}  // namespace fr_task_planner
