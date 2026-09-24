#pragma once

#include "fr_task_planner/winner_trajectory_io.hpp"

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace fr_task_planner
{
inline constexpr const char* kRealRobotConfirmation =
    "I_UNDERSTAND_THIS_WILL_MOVE_THE_REAL_ROBOT";
inline constexpr const char* kDefaultControllerName = "fairino3_controller";
inline constexpr const char* kDefaultTrajectoryActionNs = "follow_joint_trajectory";
inline constexpr const char* kDefaultTrajectoryActionName =
    "/fairino3_controller/follow_joint_trajectory";
inline constexpr const char* kDefaultGripperService = "/fairino_gripper/command";
inline constexpr const char* kDefaultPlanningGroup = "fairino3_v6_group";
inline constexpr const char* kDefaultAttachLink = "gripper_tcp";
inline constexpr const char* kDefaultObjectId = "small_part";
inline constexpr const char* kDefaultTableName = "table";
inline constexpr const char* kErrorFrozenStartMismatch = "FROZEN_START_STATE_MISMATCH";
inline constexpr const char* kErrorSegmentEndMismatch = "SEGMENT_END_STATE_MISMATCH";
inline constexpr const char* kErrorSegmentEndSettleTimeout = "SEGMENT_END_SETTLE_TIMEOUT";
inline constexpr const char* kErrorHomeSettleTimeout = "HOME_SETTLE_TIMEOUT";
inline constexpr const char* kErrorGripperCloseFailed = "REAL_GRIPPER_CLOSE_FAILED";
inline constexpr const char* kErrorGripperOpenFailed = "REAL_GRIPPER_OPEN_FAILED";
inline constexpr const char* kErrorGripperServiceUnavailable = "GRIPPER_SERVICE_UNAVAILABLE";
inline constexpr const char* kErrorGripperServiceTimeout = "GRIPPER_SERVICE_TIMEOUT";
inline constexpr const char* kErrorGripperNullResponse = "GRIPPER_NULL_RESPONSE";
inline constexpr const char* kErrorGripperBridgeError = "GRIPPER_BRIDGE_ERROR";
inline constexpr const char* kErrorGripperPingFailed = "GRIPPER_PING_FAILED";
inline constexpr const char* kErrorGripperMotionPending = "GRIPPER_MOTION_PENDING";
inline constexpr const char* kErrorGripperMotionDoneTimeout = "GRIPPER_MOTION_DONE_TIMEOUT";
inline constexpr const char* kErrorGripperServoJResumeFailed = "GRIPPER_SERVOJ_RESUME_FAILED";
inline constexpr const char* kErrorGripperCompletionUnknown = "GRIPPER_COMPLETION_UNKNOWN";
inline constexpr const char* kGripperBridgeMotionDoneMessage = "MoveGripper done";
inline constexpr const char* kGripperCommandMove = "move";
inline constexpr const char* kGripperCommandPing = "ping";
inline constexpr const char* kErrorSimTimeInvalid = "REAL_EXECUTION_SIM_TIME_INVALID";
inline constexpr const char* kErrorMissingJoint = "MISSING_JOINT";
inline constexpr const char* kErrorNanTrajectory = "NAN_TRAJECTORY";
inline constexpr const char* kErrorNonmonotonicTime = "NONMONOTONIC_TIME";
inline constexpr const char* kErrorInvalidScale = "INVALID_SCALE";
inline constexpr const char* kErrorCurrentToHomeDeployable = "CURRENT_TO_HOME_NOT_DEPLOYABLE";
inline constexpr const char* kErrorStartupInterfacesNotReady = "STARTUP_INTERFACES_NOT_READY";
inline constexpr double kDefaultStartupReadyTimeoutSec = 15.0;
inline constexpr double kDefaultStartupReadyPollSec = 0.05;

const std::vector<std::string>& deployableLogicalOrder();
const std::vector<std::string>& defaultTouchLinks();

enum class ExecutorPhase
{
  INIT,
  VALIDATE_CONFIG,
  WAIT_FOR_JOINT_STATE,
  CHECK_REAL_MODE,
  PLAN_CURRENT_TO_HOME,
  WAIT_USER_EXECUTION_GATE,
  GRIPPER_PING_PREFLIGHT,
  EXECUTE_CURRENT_TO_HOME,
  VERIFY_HOME,
  OPEN_REAL_GRIPPER,
  WAIT_GRIPPER_OPEN_DONE,
  LOAD_FROZEN_TRAJECTORY,
  VERIFY_FROZEN_START,
  EXECUTE_HOME_TO_PREGRASP,
  VERIFY_PREGRASP,
  EXECUTE_PREGRASP_TO_GRASP,
  VERIFY_GRASP,
  CLOSE_REAL_GRIPPER,
  WAIT_GRIPPER_CLOSE_DONE,
  VERIFY_GRIPPER_CLOSED,
  ATTACH_PLANNING_SCENE_OBJECT,
  EXECUTE_GRASP_TO_LIFT,
  VERIFY_LIFT,
  RESTORE_PART_TABLE_COLLISION,
  EXECUTE_LIFT_TO_A,
  VERIFY_A,
  EXECUTE_A_TO_B,
  VERIFY_B,
  EXECUTE_B_TO_C,
  VERIFY_C,
  COMPLETE,
  ABORT
};

std::string phaseName(ExecutorPhase phase);

struct ExecutorConfig
{
  bool execute = false;
  std::string real_robot_confirmation;
  bool plan_current_to_home = false;
  bool gripper_enabled = true;
  double trajectory_speed_scale = 0.05;
  double home_tolerance_rad = 0.02;
  double start_state_tolerance_rad = 0.02;
  double segment_start_tolerance_rad = 0.02;
  double segment_end_tolerance_rad = 0.02;
  double joint_settle_timeout_sec = 10.0;
  double joint_settle_poll_period_sec = 0.10;
  int joint_settle_required_samples = 3;
  double joint_state_max_age_sec = 0.5;
  double startup_ready_timeout_sec = kDefaultStartupReadyTimeoutSec;
  double timeout_factor = 1.5;
  double timeout_margin_sec = 10.0;
  double gripper_timeout_sec = 15.0;
  double gripper_post_close_wait_sec = 0.5;
  int gripper_id = 1;
  int gripper_open_position = 0;
  int gripper_close_position = 85;
  int gripper_velocity = 20;
  int gripper_force = 20;
  int gripper_max_time_ms = 5000;
  int gripper_block = 1;
  int gripper_type = 0;
  double gripper_rot_num = 0.0;
  int gripper_rot_vel = 0;
  int gripper_rot_torque = 0;
  bool gripper_ping_only = false;
  std::string trajectory_file;
  std::string controller_name = kDefaultControllerName;
  std::string trajectory_action_name = kDefaultTrajectoryActionName;
  std::string gripper_service_name = kDefaultGripperService;
  std::string planning_group = kDefaultPlanningGroup;
  std::string attach_link = kDefaultAttachLink;
  std::string object_id = kDefaultObjectId;
  std::string table_name = kDefaultTableName;
  std::vector<std::string> controller_joint_names{ "j1", "j2", "j3", "j4", "j5", "j6" };
  std::vector<std::string> touch_links = defaultTouchLinks();
};

struct JointSnapshot
{
  std::map<std::string, double> joints;
  std::map<std::string, double> velocities;
  double age_sec = 0.0;
  bool stamp_valid = false;
};

struct StartupReadyHooks
{
  std::function<std::optional<JointSnapshot>()> readJoints;
  std::function<bool()> actionReady;
  std::function<bool()> gripperReady;
  std::function<void(double)> sleepSec;
  std::function<double()> nowSec;
  std::function<void(const std::string&)> log;
};

struct StartupInterfaceStatus
{
  bool joints_ready = false;
  bool action_ready = false;
  bool gripper_ready = false;
  bool received_joint_msg = false;
  double first_joint_msg_elapsed_sec = -1.0;
  std::vector<std::string> joint_names;
  double joint_age_sec = 0.0;
  std::string joints_reason;
  bool action_available = false;
  bool gripper_available = false;
  double joints_wait_sec = 0.0;
  double action_wait_sec = 0.0;
  double gripper_wait_sec = 0.0;
  double total_wait_sec = 0.0;
  std::string timeout_reason;
  std::vector<std::string> logs;

  bool allReady() const
  {
    return joints_ready && action_ready && gripper_ready;
  }
};

struct SettleResult
{
  bool ok = false;
  std::string error;
  std::string segment;
  std::vector<double> expected;
  std::vector<double> actual;
  double last_max_error = 0.0;
  double immediate_end_error_rad = 0.0;
  double settled_end_error_rad = 0.0;
  double best_max_error = 0.0;
  double settle_elapsed_sec = 0.0;
  int samples_received = 0;
  int consecutive_ok = 0;
  int motion_commands_sent = 0;
  bool velocity_unavailable = false;
  bool had_valid_sample = false;
};

struct StateCheckResult
{
  bool ok = false;
  std::string error;
  std::string segment;
  std::vector<double> expected;
  std::vector<double> actual;
  double max_error = 0.0;
};

struct ScaleResult
{
  bool ok = false;
  std::string error;
  TrajectorySegmentRecord segment;
};

enum class GripperCompletionKind
{
  CommandAccepted,
  BridgeMotionDone,
  MotionPending,
  MotionDoneTimeout,
  ServoJResumeFailure,
  Unknown
};

struct SendResult
{
  bool attempted = false;
  bool sent = false;
  bool success = false;
  std::string error;
  int error_code = 0;
  std::string message;
  double elapsed_sec = 0.0;
  bool response_received = false;
  GripperCompletionKind gripper_completion = GripperCompletionKind::Unknown;
};

struct GripperBridgeRequestFields
{
  std::string command = kGripperCommandMove;
  int gripper_id = 1;
  int position = 85;
  int velocity = 20;
  int force = 20;
  int max_time_ms = 5000;
  int block = 1;
  int gripper_type = 0;
  double rot_num = 0.0;
  int rot_vel = 0;
  int rot_torque = 0;
};

enum class GripperCallKind
{
  Success,
  ServiceUnavailable,
  Timeout,
  NullResponse,
  BridgeError
};

struct GripperCallOutcome
{
  bool ok = false;
  GripperCallKind kind = GripperCallKind::Success;
  std::string error;
  int error_code = 0;
  std::string message;
  double elapsed_sec = 0.0;
  bool response_received = false;
};

struct PlanResult
{
  bool attempted = false;
  bool success = false;
  bool used_frozen_current_to_home = false;
  std::string error;
  TrajectorySegmentRecord planned;
};

struct ExecutorHooks
{
  std::function<void(const std::string&)> log;
  std::function<std::optional<JointSnapshot>()> readJoints;
  std::function<PlanResult()> planCurrentToHome;
  std::function<SendResult(const std::string& logical, const TrajectorySegmentRecord& scaled)>
      sendSegment;
  std::function<SendResult()> openGripper;
  std::function<SendResult()> closeGripper;
  std::function<SendResult()> pingGripper;
  std::function<SendResult()> attachObject;
  std::function<SendResult()> restoreTableCollision;
  std::function<bool()> useSimTimeInvalid;
  std::function<bool()> gazeboDetected;
  std::function<bool()> controllerReady;
  std::function<void(double)> sleepSec;
  std::function<double()> nowSec;
};

struct ExecutorTrace
{
  bool ok = false;
  bool dry_run = true;
  bool motion_enabled = false;
  bool aborted = false;
  std::string abort_reason;
  ExecutorPhase final_phase = ExecutorPhase::INIT;
  std::vector<std::string> phases;
  std::vector<std::string> events;
  std::vector<std::string> segments_prepared;
  std::vector<std::string> segments_sent;
  int trajectories_sent = 0;
  int gripper_commands_sent = 0;
  int attach_calls = 0;
  int restore_table_calls = 0;
  int current_to_home_plans = 0;
  int current_to_home_executes = 0;
  int frozen_replans = 0;
  bool gripper_open_before_pregrasp = false;
  bool gripper_before_attach = false;
  bool gripper_close_before_lift = false;
  bool attach_before_lift = false;
  GripperCompletionKind last_gripper_completion = GripperCompletionKind::Unknown;
  bool used_frozen_current_to_home = false;
  double original_duration = 0.0;
  double scaled_duration = 0.0;
};

bool motionAllowed(const ExecutorConfig& cfg, std::string& reason);
bool validSpeedScale(double scale, std::string& error);
double actionTimeoutSec(double scaled_duration, double factor, double margin);
double timeFromStartSec(const TrajectoryPointRecord& pt);
void setTimeFromStart(TrajectoryPointRecord& pt, double seconds);

bool trajectoryHasFiniteValues(const TrajectorySegmentRecord& seg, std::string& error);
bool rejectCurrentToHomeAsDeployable(const PersistedTrajectory& traj, std::string& error);
std::vector<TrajectorySegmentRecord> collectDeployableSegments(const PersistedTrajectory& traj,
                                                               std::string& error);
ScaleResult scaleSegment(const TrajectorySegmentRecord& in, double scale);
bool scaleDeployableTrajectory(const PersistedTrajectory& in, double scale,
                               std::vector<TrajectorySegmentRecord>& out, std::string& error);

bool remapByJointName(const std::vector<std::string>& source_names,
                      const std::vector<double>& source_values,
                      const std::vector<std::string>& target_names, std::vector<double>& out,
                      std::string& error);
bool remapSegmentByJointName(const TrajectorySegmentRecord& in,
                             const std::vector<std::string>& controller_names,
                             TrajectorySegmentRecord& out, std::string& error);

StateCheckResult checkNamedJoints(const JointSnapshot& snap, const std::vector<std::string>& names,
                                  const std::vector<double>& expected, double tolerance_rad,
                                  const std::string& error_code, const std::string& segment);
StateCheckResult checkHome(const JointSnapshot& snap, const std::vector<double>& home,
                           double tolerance_rad);
StateCheckResult checkSegmentStart(const JointSnapshot& snap, const TrajectorySegmentRecord& seg,
                                   double tolerance_rad);
StateCheckResult checkSegmentEnd(const JointSnapshot& snap, const TrajectorySegmentRecord& seg,
                                 double tolerance_rad);
bool snapshotHasRequiredJoints(const JointSnapshot& snap, std::string& error);
bool snapshotFresh(const JointSnapshot& snap, double max_age_sec, std::string& error);
StartupInterfaceStatus waitForStartupInterfaces(const ExecutorConfig& cfg,
                                                const StartupReadyHooks& hooks,
                                                double timeout_sec);
bool snapshotVelocityUnavailable(const JointSnapshot& snap);
SettleResult waitForJointConvergence(const std::string& segment,
                                     const std::vector<std::string>& target_names,
                                     const std::vector<double>& target_values, double tolerance_rad,
                                     const ExecutorConfig& cfg, ExecutorHooks hooks);

GripperBridgeRequestFields makeGripperCloseRequest(const ExecutorConfig& cfg);
GripperBridgeRequestFields makeGripperOpenRequest(const ExecutorConfig& cfg);
GripperBridgeRequestFields makeGripperPingRequest(const ExecutorConfig& cfg);
std::string formatGripperRequestLog(const GripperBridgeRequestFields& req);
std::string formatGripperResponseLog(int error_code, const std::string& message, double elapsed_sec,
                                     bool response_received);
std::string gripperCompletionName(GripperCompletionKind kind);
GripperCallOutcome classifyGripperCall(bool service_available, bool timed_out, bool null_response,
                                       int error_code, const std::string& message,
                                       double elapsed_sec, const char* nonzero_prefix);
GripperCompletionKind classifyGripperCompletion(const GripperCallOutcome& call,
                                                const GripperBridgeRequestFields& req);
bool gripperBridgeMotionDone(GripperCompletionKind kind);
std::string formatGripperCompletionFailure(const GripperCallOutcome& call,
                                           GripperCompletionKind kind, const char* prefix);

ExecutorTrace runFrozenExecutor(const ExecutorConfig& cfg, const PersistedTrajectory& traj,
                                ExecutorHooks hooks);
}  // namespace fr_task_planner
