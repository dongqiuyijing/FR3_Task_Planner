#pragma once

#include "fr_task_planner/inspection_endpoint_candidates.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <rclcpp/logger.hpp>

namespace fr_task_planner
{
struct NamedJointSeed
{
  std::string name;
  std::map<std::string, double> joints;
};

struct PreGraspIkConfig
{
  std::string group = "fairino3_v6_group";
  std::string ee_link = "gripper_tcp";
  std::string object_id = "small_part";
  std::string table_name = "table";
  std::string column_name = "mounting_column";
  std::vector<std::string> touch_links;
  double pos_tol = 0.005;
  double ori_tol_deg = 3.0;
  double min_ik_solution_distance = 0.1;
  uint32_t max_unique_candidates = 24;
  uint32_t max_ik_attempts = 96;
  double nearby_radius = 2.0;
  double nearby_radius_local = 0.8;
};

struct PreGraspIkCandidate
{
  std::string candidate_id;
  std::string seed_name = "unknown";
  std::map<std::string, double> joints;
  double fk_position_error = 0.0;
  double fk_orientation_error_deg = 0.0;
  double distance_from_home = 0.0;
  double joint_l1_from_home = 0.0;
  double j1_abs_displacement = 0.0;
  double j1_shortest_valid_displacement = 0.0;
  double joint_limit_margin = 0.0;
  double column_clearance = 0.0;
  bool bounds_ok = false;
  bool fk_ok = false;
  bool collision_free = false;
  bool self_collision_ok = false;
  bool table_collision_ok = false;
  bool column_collision_ok = false;
  bool attached_object_present = false;
  bool is_baseline = false;
  bool valid = false;
  std::string failure_reason = "ok";
};

struct PreGraspIkGenerationResult
{
  int attempts = 0;
  int ik_success = 0;
  int unique_raw = 0;
  int fk_valid = 0;
  int bounds_valid = 0;
  int collision_valid = 0;
  int unique_valid = 0;
  bool baseline_included = false;
  std::vector<PreGraspIkCandidate> raw;
  std::vector<PreGraspIkCandidate> unique;
};

double shortestValidJointDisplacement(double from, double to, double lower, double upper);
bool isSameIkBranch(const std::map<std::string, double>& a, const std::map<std::string, double>& b,
                    double min_distance);
int findDuplicateIndex(const std::vector<PreGraspIkCandidate>& cands,
                       const std::map<std::string, double>& joints, double min_distance);
double jointLimitMargin(const moveit::core::RobotState& state,
                        const moveit::core::JointModelGroup* jmg);

void validatePreGraspCandidate(PreGraspIkCandidate& cand,
                               const planning_scene::PlanningScene& pregrasp_scene,
                               const geometry_msgs::msg::PoseStamped& pregrasp_pose,
                               const std::map<std::string, double>& home,
                               const PreGraspIkConfig& cfg);

PreGraspIkGenerationResult generatePreGraspIkCandidates(
    const planning_scene::PlanningScene& pregrasp_scene,
    const geometry_msgs::msg::PoseStamped& pregrasp_pose, const std::map<std::string, double>& home,
    const std::map<std::string, double>& baseline_pregrasp,
    const std::vector<NamedJointSeed>& extra_seeds, const PreGraspIkConfig& cfg,
    const rclcpp::Logger& logger);
}  // namespace fr_task_planner
