#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace fr_task_planner
{
struct WinnerEndpoints
{
  std::string source_path;
  std::string task_version;
  std::map<std::string, double> home_rad;
  std::map<std::string, double> a_rad;
  std::map<std::string, double> b_rad;
  std::map<std::string, double> c_rad;
  double a_roll_deg = 0.0;
  double b_roll_deg = 0.0;
  double c_roll_deg = 0.0;
  double search_score_total_time = 0.0;
  double search_score_joint_path_length = 0.0;
};

struct TrajectoryPointRecord
{
  std::vector<double> positions;
  std::vector<double> velocities;
  std::vector<double> accelerations;
  int32_t sec = 0;
  uint32_t nanosec = 0;
};

struct MultiDofTransformRecord
{
  std::vector<double> translation;
  std::vector<double> rotation_xyzw;
};

struct MultiDofPointRecord
{
  std::vector<MultiDofTransformRecord> transforms;
  int32_t sec = 0;
  uint32_t nanosec = 0;
};

struct MultiDofRecord
{
  std::vector<std::string> joint_names;
  std::vector<MultiDofPointRecord> points;
};

struct TrajectorySegmentRecord
{
  std::string name;
  std::string planning_stage_name;
  std::string logical_segment;
  int subsegment_index = 0;
  bool deployable = false;
  bool runtime_replan_required = false;
  std::vector<std::string> joint_names;
  std::vector<double> start_joints;
  std::vector<double> end_joints;
  double duration = 0.0;
  bool velocities_present = false;
  bool accelerations_present = false;
  bool multi_dof_present = false;
  std::vector<TrajectoryPointRecord> points;
  MultiDofRecord multi_dof;
};

struct EventRecord
{
  std::string after_segment;
  std::string event;
  bool executed_now = false;
};

struct PersistedTrajectory
{
  int trajectory_format_version = 1;
  std::string task_version = "STEP12C";
  std::string source_winner_file;
  std::string source_winner_path;
  std::string label = "FROZEN STEP12C WINNER TRAJECTORY";
  bool execution_performed = false;
  bool real_robot_validated = false;
  bool requires_runtime_current_to_home = true;
  bool requires_real_gripper_close = true;
  bool requires_real_retime_or_scaling = true;
  std::string runtime_current_to_home = "REPLAN_REQUIRED";
  std::string fixed_trajectory_start = "HOME";
  std::string real_execution_speed_scaling = "NOT APPLIED";
  double recommended_initial_real_validation_scaling = 0.05;
  double search_score_total_time = 0.0;
  double search_score_joint_path_length = 0.0;
  double persisted_full_plan_total_time = 0.0;
  double persisted_fixed_task_total_duration = 0.0;
  double persisted_total_joint_path_length = 0.0;
  double persisted_fixed_task_joint_path_length = 0.0;
  std::vector<std::string> joint_names;
  std::vector<EventRecord> events;
  std::vector<TrajectorySegmentRecord> segments;
  WinnerEndpoints winner;
};

struct PersistValidation
{
  bool ok = true;
  std::string reason = "ok";
  double position_max_error = 0.0;
  double velocity_max_error = 0.0;
  double acceleration_max_error = 0.0;
  bool time_identical = true;
  bool point_count_identical = true;
  bool joint_names_identical = true;
  double home_start_error = 0.0;
  double a_max_error = 0.0;
  double b_max_error = 0.0;
  double c_max_error = 0.0;
  double max_discontinuity = 0.0;
  std::map<std::string, double> discontinuities;
};

const std::vector<double> kStep12cHomeRad = { -2.271411675804, -1.642863047968, -1.869634009789,
                                              -2.871167788011, -0.003375517130, 0.839900045153 };
const std::vector<double> kStep12cARad = { 0.761386084, -0.642747728, 1.360379768,
                                           0.853167960, 1.570792654, -0.024008243 };
const std::vector<double> kStep12cBRad = { 0.128839903, -0.765711154, 2.071148432,
                                           0.157964462, 0.879595649, 2.523736939 };
const std::vector<double> kStep12cCRad = { 0.398510667, 0.059882338, 0.469679891,
                                           -0.529563111, -1.172285497, 2.443454499 };

inline constexpr const char* kLogicalCurrentToHome = "Current_to_Home";
inline constexpr const char* kLogicalHomeToPreGrasp = "Home_to_PreGrasp";
inline constexpr const char* kLogicalPreGraspToGrasp = "PreGrasp_to_Grasp";
inline constexpr const char* kLogicalGraspToLift = "Grasp_to_Lift";
inline constexpr const char* kLogicalLiftToA = "Lift_to_A";
inline constexpr const char* kLogicalAToB = "A_to_B";
inline constexpr const char* kLogicalBToC = "B_to_C_original_bottom";

std::string logicalSegmentFromStage(const std::string& stage_name);
bool isFixedDeployableLogical(const std::string& logical);
std::vector<double> jointsToVec(const std::map<std::string, double>& joints);
std::map<std::string, double> vecToJoints(const std::vector<double>& values);
std::vector<double> extractArmPositions(const std::vector<std::string>& names,
                                        const std::vector<double>& values);
double maxAbsError(const std::vector<double>& a, const std::vector<double>& b);
double maxAbsErrorJoints(const std::map<std::string, double>& a,
                         const std::map<std::string, double>& b);
double segmentDuration(const TrajectorySegmentRecord& seg);
double computePathLength(const std::vector<TrajectorySegmentRecord>& segments, bool deployable_only);
double computeDuration(const std::vector<TrajectorySegmentRecord>& segments, bool deployable_only);
void fillDerivedTotals(PersistedTrajectory& traj);
void attachStandardEvents(PersistedTrajectory& traj);

bool loadWinnerYaml(const std::string& path, WinnerEndpoints& out, std::string& error);
bool winnerMatchesFrozenStep12c(const WinnerEndpoints& winner, std::string& error, double tol = 1e-8);
bool writeTrajectoryYaml(const std::string& path, const PersistedTrajectory& traj, std::string& error);
bool readTrajectoryYaml(const std::string& path, PersistedTrajectory& traj, std::string& error);
PersistValidation validateRoundTrip(const PersistedTrajectory& original,
                                    const PersistedTrajectory& loaded);
PersistValidation validateContinuity(const PersistedTrajectory& traj, double tol = 1e-4);
PersistValidation validateEndpoints(const PersistedTrajectory& traj, const WinnerEndpoints& winner,
                                    double tol = 1e-4);
bool timesMonotonic(const TrajectorySegmentRecord& seg);
}  // namespace fr_task_planner
