#include "fr_task_planner/frozen_executor.hpp"
#include "fr_task_planner/inspection_endpoint_candidates.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <thread>

namespace fr_task_planner
{
namespace
{
void logLine(const ExecutorHooks& hooks, const std::string& text)
{
  if (hooks.log)
  {
    hooks.log(text);
  }
}

std::string formatJoints(const std::vector<double>& values)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(9);
  oss << "[";
  for (size_t i = 0; i < kArmJoints.size(); ++i)
  {
    if (i)
    {
      oss << ", ";
    }
    oss << kArmJoints[i] << "=";
    if (i < values.size())
    {
      oss << values[i];
    }
    else
    {
      oss << "MISSING";
    }
  }
  oss << "]";
  return oss.str();
}

std::vector<double> snapshotArm(const JointSnapshot& snap)
{
  return jointsToVec(snap.joints);
}

bool finiteVec(const std::vector<double>& values)
{
  for (double v : values)
  {
    if (!std::isfinite(v))
    {
      return false;
    }
  }
  return true;
}

void abortTo(ExecutorTrace& trace, const std::string& reason)
{
  trace.aborted = true;
  trace.ok = false;
  trace.abort_reason = reason;
  trace.final_phase = ExecutorPhase::ABORT;
  trace.phases.push_back(phaseName(ExecutorPhase::ABORT));
}

double defaultNowSec()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void defaultSleepSec(double seconds)
{
  if (seconds > 0.0)
  {
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
  }
}

double nowFrom(const ExecutorHooks& hooks)
{
  return hooks.nowSec ? hooks.nowSec() : defaultNowSec();
}

void sleepFrom(const ExecutorHooks& hooks, double seconds)
{
  if (hooks.sleepSec)
  {
    hooks.sleepSec(seconds);
    return;
  }
  defaultSleepSec(seconds);
}

std::string formatSettle(const SettleResult& r, double elapsed, double tolerance, const char* title)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(6);
  oss << title << " segment=" << r.segment << " elapsed=" << elapsed
      << " max_error=" << r.last_max_error << " tolerance=" << tolerance
      << " consecutive=" << r.consecutive_ok << " samples=" << r.samples_received
      << " best_max_error=" << r.best_max_error;
  return oss.str();
}

std::vector<std::string> armNames()
{
  return std::vector<std::string>(kArmJoints.begin(), kArmJoints.end());
}

const TrajectorySegmentRecord* findLogical(const std::vector<TrajectorySegmentRecord>& segs,
                                           const std::string& logical)
{
  for (const auto& seg : segs)
  {
    if (seg.logical_segment == logical)
    {
      return &seg;
    }
  }
  return nullptr;
}
}  // namespace

const std::vector<std::string>& deployableLogicalOrder()
{
  static const std::vector<std::string> kOrder = {
    kLogicalHomeToPreGrasp, kLogicalPreGraspToGrasp, kLogicalGraspToLift,
    kLogicalLiftToA,        kLogicalAToB,            kLogicalBToC
  };
  return kOrder;
}

const std::vector<std::string>& defaultTouchLinks()
{
  static const std::vector<std::string> kLinks = { "gripper_base_link", "rail_155", "slider_l",
                                                   "slider_r",         "finger_l", "finger_r",
                                                   "gripper_gap_link" };
  return kLinks;
}

std::string phaseName(ExecutorPhase phase)
{
  switch (phase)
  {
    case ExecutorPhase::INIT:
      return "INIT";
    case ExecutorPhase::VALIDATE_CONFIG:
      return "VALIDATE_CONFIG";
    case ExecutorPhase::WAIT_FOR_JOINT_STATE:
      return "WAIT_FOR_JOINT_STATE";
    case ExecutorPhase::CHECK_REAL_MODE:
      return "CHECK_REAL_MODE";
    case ExecutorPhase::PLAN_CURRENT_TO_HOME:
      return "PLAN_CURRENT_TO_HOME";
    case ExecutorPhase::WAIT_USER_EXECUTION_GATE:
      return "WAIT_USER_EXECUTION_GATE";
    case ExecutorPhase::GRIPPER_PING_PREFLIGHT:
      return "GRIPPER_PING_PREFLIGHT";
    case ExecutorPhase::EXECUTE_CURRENT_TO_HOME:
      return "EXECUTE_CURRENT_TO_HOME";
    case ExecutorPhase::VERIFY_HOME:
      return "VERIFY_HOME";
    case ExecutorPhase::OPEN_REAL_GRIPPER:
      return "OPEN_REAL_GRIPPER";
    case ExecutorPhase::WAIT_GRIPPER_OPEN_DONE:
      return "WAIT_GRIPPER_OPEN_DONE";
    case ExecutorPhase::LOAD_FROZEN_TRAJECTORY:
      return "LOAD_FROZEN_TRAJECTORY";
    case ExecutorPhase::VERIFY_FROZEN_START:
      return "VERIFY_FROZEN_START";
    case ExecutorPhase::EXECUTE_HOME_TO_PREGRASP:
      return "EXECUTE_HOME_TO_PREGRASP";
    case ExecutorPhase::VERIFY_PREGRASP:
      return "VERIFY_PREGRASP";
    case ExecutorPhase::EXECUTE_PREGRASP_TO_GRASP:
      return "EXECUTE_PREGRASP_TO_GRASP";
    case ExecutorPhase::VERIFY_GRASP:
      return "VERIFY_GRASP";
    case ExecutorPhase::CLOSE_REAL_GRIPPER:
      return "CLOSE_REAL_GRIPPER";
    case ExecutorPhase::WAIT_GRIPPER_CLOSE_DONE:
      return "WAIT_GRIPPER_CLOSE_DONE";
    case ExecutorPhase::VERIFY_GRIPPER_CLOSED:
      return "VERIFY_GRIPPER_CLOSED";
    case ExecutorPhase::ATTACH_PLANNING_SCENE_OBJECT:
      return "ATTACH_PLANNING_SCENE_OBJECT";
    case ExecutorPhase::EXECUTE_GRASP_TO_LIFT:
      return "EXECUTE_GRASP_TO_LIFT";
    case ExecutorPhase::VERIFY_LIFT:
      return "VERIFY_LIFT";
    case ExecutorPhase::RESTORE_PART_TABLE_COLLISION:
      return "RESTORE_PART_TABLE_COLLISION";
    case ExecutorPhase::EXECUTE_LIFT_TO_A:
      return "EXECUTE_LIFT_TO_A";
    case ExecutorPhase::VERIFY_A:
      return "VERIFY_A";
    case ExecutorPhase::EXECUTE_A_TO_B:
      return "EXECUTE_A_TO_B";
    case ExecutorPhase::VERIFY_B:
      return "VERIFY_B";
    case ExecutorPhase::EXECUTE_B_TO_C:
      return "EXECUTE_B_TO_C";
    case ExecutorPhase::VERIFY_C:
      return "VERIFY_C";
    case ExecutorPhase::COMPLETE:
      return "COMPLETE";
    case ExecutorPhase::ABORT:
      return "ABORT";
  }
  return "UNKNOWN";
}

bool motionAllowed(const ExecutorConfig& cfg, std::string& reason)
{
  if (!cfg.execute)
  {
    reason = "execute=false; DRY RUN ONLY";
    return false;
  }
  if (cfg.real_robot_confirmation != kRealRobotConfirmation)
  {
    reason = "real_robot_confirmation mismatch; DRY RUN ONLY";
    return false;
  }
  reason = "execute=true and confirmation matched";
  return true;
}

bool validSpeedScale(double scale, std::string& error)
{
  if (!(scale > 0.0) || scale > 1.0 || !std::isfinite(scale))
  {
    error = kErrorInvalidScale;
    return false;
  }
  error.clear();
  return true;
}

double actionTimeoutSec(double scaled_duration, double factor, double margin)
{
  return std::max(1.0, scaled_duration) * factor + margin;
}

namespace
{
std::string formatElapsedSec(double elapsed)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(3);
  oss << elapsed;
  return oss.str();
}
}  // namespace

GripperBridgeRequestFields makeGripperCloseRequest(const ExecutorConfig& cfg)
{
  GripperBridgeRequestFields req;
  req.command = kGripperCommandMove;
  req.gripper_id = cfg.gripper_id;
  req.position = cfg.gripper_close_position;
  req.velocity = cfg.gripper_velocity;
  req.force = cfg.gripper_force;
  req.max_time_ms = cfg.gripper_max_time_ms;
  req.block = cfg.gripper_block;
  req.gripper_type = cfg.gripper_type;
  req.rot_num = cfg.gripper_rot_num;
  req.rot_vel = cfg.gripper_rot_vel;
  req.rot_torque = cfg.gripper_rot_torque;
  return req;
}

GripperBridgeRequestFields makeGripperOpenRequest(const ExecutorConfig& cfg)
{
  GripperBridgeRequestFields req = makeGripperCloseRequest(cfg);
  req.position = cfg.gripper_open_position;
  return req;
}

GripperBridgeRequestFields makeGripperPingRequest(const ExecutorConfig& cfg)
{
  GripperBridgeRequestFields req = makeGripperCloseRequest(cfg);
  req.command = kGripperCommandPing;
  req.position = 0;
  return req;
}

std::string gripperCompletionName(GripperCompletionKind kind)
{
  switch (kind)
  {
    case GripperCompletionKind::CommandAccepted:
      return "COMMAND_ACCEPTED";
    case GripperCompletionKind::BridgeMotionDone:
      return "BRIDGE_MOTION_DONE";
    case GripperCompletionKind::MotionPending:
      return "MOTION_PENDING";
    case GripperCompletionKind::MotionDoneTimeout:
      return "MOTION_DONE_TIMEOUT";
    case GripperCompletionKind::ServoJResumeFailure:
      return "SERVOJ_RESUME_FAILURE";
    case GripperCompletionKind::Unknown:
      return "COMPLETION_UNKNOWN";
  }
  return "COMPLETION_UNKNOWN";
}

namespace
{
bool messageContains(const std::string& haystack, const char* needle)
{
  if (!needle || needle[0] == '\0')
  {
    return false;
  }
  return haystack.find(needle) != std::string::npos;
}

bool isMoveCommand(const GripperBridgeRequestFields& req)
{
  return req.command == kGripperCommandMove;
}
}  // namespace

GripperCompletionKind classifyGripperCompletion(const GripperCallOutcome& call,
                                                const GripperBridgeRequestFields& req)
{
  if (call.kind == GripperCallKind::Timeout)
  {
    return GripperCompletionKind::MotionDoneTimeout;
  }
  if (messageContains(call.error, "GetGripperMotionDone timeout") ||
      messageContains(call.message, "GetGripperMotionDone timeout") ||
      messageContains(call.error, "timed out waiting for motion done") ||
      messageContains(call.message, "timed out waiting for motion done"))
  {
    return GripperCompletionKind::MotionDoneTimeout;
  }
  if (messageContains(call.error, "ServoMove") || messageContains(call.message, "ServoMove"))
  {
    return GripperCompletionKind::ServoJResumeFailure;
  }
  if (messageContains(call.error, "pending") || messageContains(call.message, "pending") ||
      messageContains(call.error, "in motion") || messageContains(call.message, "in motion") ||
      messageContains(call.message, "still pending"))
  {
    return GripperCompletionKind::MotionPending;
  }
  if (!isMoveCommand(req))
  {
    return call.ok ? GripperCompletionKind::CommandAccepted : GripperCompletionKind::Unknown;
  }
  if (!call.ok)
  {
    return GripperCompletionKind::Unknown;
  }
  if (messageContains(call.message, kGripperBridgeMotionDoneMessage) ||
      messageContains(call.message, "MoveGripper done"))
  {
    return GripperCompletionKind::BridgeMotionDone;
  }
  if (call.message == "ok" || messageContains(call.message, "COMMAND_ACCEPTED"))
  {
    return GripperCompletionKind::CommandAccepted;
  }
  return GripperCompletionKind::Unknown;
}

bool gripperBridgeMotionDone(GripperCompletionKind kind)
{
  return kind == GripperCompletionKind::BridgeMotionDone;
}

std::string formatGripperCompletionFailure(const GripperCallOutcome& call,
                                           GripperCompletionKind kind, const char* prefix)
{
  const char* head = (prefix && prefix[0] != '\0') ? prefix : kErrorGripperCloseFailed;
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(3);
  switch (kind)
  {
    case GripperCompletionKind::MotionPending:
      oss << kErrorGripperMotionPending;
      break;
    case GripperCompletionKind::MotionDoneTimeout:
      oss << kErrorGripperMotionDoneTimeout;
      break;
    case GripperCompletionKind::ServoJResumeFailure:
      oss << kErrorGripperServoJResumeFailed;
      break;
    case GripperCompletionKind::CommandAccepted:
      oss << kErrorGripperCompletionUnknown << " COMMAND_ACCEPTED_IS_NOT_MOTION_DONE";
      break;
    case GripperCompletionKind::Unknown:
      oss << kErrorGripperCompletionUnknown;
      break;
    case GripperCompletionKind::BridgeMotionDone:
      oss << head;
      break;
  }
  oss << " " << head << " completion=" << gripperCompletionName(kind)
      << " error_code=" << call.error_code << " message=\"" << call.message
      << "\" elapsed_sec=" << call.elapsed_sec
      << " response_received=" << (call.response_received ? "YES" : "NO");
  if (!call.error.empty() && call.error != oss.str())
  {
    oss << " detail=\"" << call.error << "\"";
  }
  return oss.str();
}

std::string formatGripperRequestLog(const GripperBridgeRequestFields& req)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(6);
  oss << "GRIPPER REQUEST command=" << req.command << " gripper_id=" << req.gripper_id
      << " position=" << req.position << " velocity=" << req.velocity << " force=" << req.force
      << " max_time_ms=" << req.max_time_ms << " block=" << req.block
      << " gripper_type=" << req.gripper_type << " rot_num=" << req.rot_num
      << " rot_vel=" << req.rot_vel << " rot_torque=" << req.rot_torque;
  return oss.str();
}

std::string formatGripperResponseLog(int error_code, const std::string& message, double elapsed_sec,
                                     bool response_received)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(3);
  oss << "GRIPPER RESPONSE error_code=" << error_code << " message=\"" << message
      << "\" elapsed_sec=" << elapsed_sec
      << " response_received=" << (response_received ? "YES" : "NO");
  return oss.str();
}

GripperCallOutcome classifyGripperCall(bool service_available, bool timed_out, bool null_response,
                                       int error_code, const std::string& message,
                                       double elapsed_sec, const char* nonzero_prefix)
{
  GripperCallOutcome out;
  out.error_code = error_code;
  out.message = message;
  out.elapsed_sec = elapsed_sec;
  const char* prefix =
      (nonzero_prefix && nonzero_prefix[0] != '\0') ? nonzero_prefix : kErrorGripperCloseFailed;

  if (!service_available)
  {
    out.kind = GripperCallKind::ServiceUnavailable;
    out.error = std::string(kErrorGripperServiceUnavailable) + " elapsed_sec=" +
                formatElapsedSec(elapsed_sec);
    return out;
  }
  if (timed_out)
  {
    out.kind = GripperCallKind::Timeout;
    out.error =
        std::string(kErrorGripperServiceTimeout) + " elapsed=" + formatElapsedSec(elapsed_sec);
    return out;
  }
  if (null_response)
  {
    out.kind = GripperCallKind::NullResponse;
    out.error =
        std::string(kErrorGripperNullResponse) + " elapsed_sec=" + formatElapsedSec(elapsed_sec);
    return out;
  }
  out.response_received = true;
  if (error_code != 0)
  {
    out.kind = GripperCallKind::BridgeError;
    std::ostringstream oss;
    oss << prefix << " " << kErrorGripperBridgeError << " error_code=" << error_code
        << " message=\"" << message << "\" elapsed_sec=" << formatElapsedSec(elapsed_sec);
    out.error = oss.str();
    return out;
  }
  out.ok = true;
  out.kind = GripperCallKind::Success;
  return out;
}

double timeFromStartSec(const TrajectoryPointRecord& pt)
{
  return static_cast<double>(pt.sec) + 1e-9 * static_cast<double>(pt.nanosec);
}

void setTimeFromStart(TrajectoryPointRecord& pt, double seconds)
{
  if (!std::isfinite(seconds) || seconds < 0.0)
  {
    pt.sec = 0;
    pt.nanosec = 0;
    return;
  }
  const double whole = std::floor(seconds);
  pt.sec = static_cast<int32_t>(whole);
  long long ns = static_cast<long long>(std::llround((seconds - whole) * 1e9));
  if (ns >= 1000000000LL)
  {
    pt.sec += 1;
    ns -= 1000000000LL;
  }
  if (ns < 0)
  {
    pt.sec -= 1;
    ns += 1000000000LL;
  }
  pt.nanosec = static_cast<uint32_t>(ns);
}

bool trajectoryHasFiniteValues(const TrajectorySegmentRecord& seg, std::string& error)
{
  auto bad = [&](const std::vector<double>& values, const char* field) {
    if (!finiteVec(values))
    {
      error = std::string(kErrorNanTrajectory) + " field=" + field + " segment=" + seg.name;
      return true;
    }
    return false;
  };
  if (bad(seg.start_joints, "start_joints") || bad(seg.end_joints, "end_joints"))
  {
    return false;
  }
  for (size_t i = 0; i < seg.points.size(); ++i)
  {
    const auto& pt = seg.points[i];
    if (bad(pt.positions, "positions") || bad(pt.velocities, "velocities") ||
        bad(pt.accelerations, "accelerations"))
    {
      error += " point=" + std::to_string(i);
      return false;
    }
  }
  return true;
}

bool rejectCurrentToHomeAsDeployable(const PersistedTrajectory& traj, std::string& error)
{
  for (const auto& seg : traj.segments)
  {
    if (seg.logical_segment == kLogicalCurrentToHome && seg.deployable)
    {
      error = kErrorCurrentToHomeDeployable;
      return false;
    }
  }
  return true;
}

std::vector<TrajectorySegmentRecord> collectDeployableSegments(const PersistedTrajectory& traj,
                                                               std::string& error)
{
  std::vector<TrajectorySegmentRecord> out;
  if (!rejectCurrentToHomeAsDeployable(traj, error))
  {
    return {};
  }
  std::map<std::string, std::vector<TrajectorySegmentRecord>> by_logical;
  for (const auto& seg : traj.segments)
  {
    if (seg.logical_segment == kLogicalCurrentToHome)
    {
      continue;
    }
    if (!isFixedDeployableLogical(seg.logical_segment) || !seg.deployable)
    {
      continue;
    }
    if (seg.runtime_replan_required)
    {
      error = "deployable segment marked runtime_replan_required: " + seg.logical_segment;
      return {};
    }
    std::string local_error;
    if (!trajectoryHasFiniteValues(seg, local_error))
    {
      error = local_error;
      return {};
    }
    if (!timesMonotonic(seg))
    {
      error = std::string(kErrorNonmonotonicTime) + " segment=" + seg.name;
      return {};
    }
    by_logical[seg.logical_segment].push_back(seg);
  }
  for (const auto& logical : deployableLogicalOrder())
  {
    const auto it = by_logical.find(logical);
    if (it == by_logical.end() || it->second.empty())
    {
      error = "missing deployable logical segment: " + logical;
      return {};
    }
    out.insert(out.end(), it->second.begin(), it->second.end());
  }
  if (out.size() != deployableLogicalOrder().size())
  {
    error = "expected exactly 6 deployable segments, got " + std::to_string(out.size());
    return {};
  }
  return out;
}

ScaleResult scaleSegment(const TrajectorySegmentRecord& in, double scale)
{
  ScaleResult result;
  std::string error;
  if (!validSpeedScale(scale, error))
  {
    result.error = error;
    return result;
  }
  result.segment = in;
  if (scale == 1.0)
  {
    result.ok = true;
    result.segment.duration = segmentDuration(result.segment);
    return result;
  }
  const double inv = 1.0 / scale;
  const double acc_scale = scale * scale;
  for (auto& pt : result.segment.points)
  {
    for (double& v : pt.velocities)
    {
      v *= scale;
    }
    for (double& a : pt.accelerations)
    {
      a *= acc_scale;
    }
    setTimeFromStart(pt, timeFromStartSec(pt) * inv);
  }
  result.segment.duration = segmentDuration(result.segment);
  result.ok = true;
  return result;
}

bool scaleDeployableTrajectory(const PersistedTrajectory& in, double scale,
                               std::vector<TrajectorySegmentRecord>& out, std::string& error)
{
  out.clear();
  if (!validSpeedScale(scale, error))
  {
    return false;
  }
  const auto segs = collectDeployableSegments(in, error);
  if (segs.empty())
  {
    return false;
  }
  for (const auto& seg : segs)
  {
    auto scaled = scaleSegment(seg, scale);
    if (!scaled.ok)
    {
      error = scaled.error;
      return false;
    }
    out.push_back(std::move(scaled.segment));
  }
  return true;
}

bool remapByJointName(const std::vector<std::string>& source_names,
                      const std::vector<double>& source_values,
                      const std::vector<std::string>& target_names, std::vector<double>& out,
                      std::string& error)
{
  std::map<std::string, double> lookup;
  const size_t n = std::min(source_names.size(), source_values.size());
  for (size_t i = 0; i < n; ++i)
  {
    lookup[source_names[i]] = source_values[i];
  }
  out.clear();
  out.reserve(target_names.size());
  for (const auto& name : target_names)
  {
    const auto it = lookup.find(name);
    if (it == lookup.end())
    {
      error = std::string(kErrorMissingJoint) + " joint=" + name;
      out.clear();
      return false;
    }
    out.push_back(it->second);
  }
  return true;
}

bool remapSegmentByJointName(const TrajectorySegmentRecord& in,
                             const std::vector<std::string>& controller_names,
                             TrajectorySegmentRecord& out, std::string& error)
{
  out = in;
  out.joint_names = controller_names;
  const auto source_names = in.joint_names.empty() ? armNames() : in.joint_names;
  if (!remapByJointName(source_names, in.start_joints, controller_names, out.start_joints, error))
  {
    return false;
  }
  if (!remapByJointName(source_names, in.end_joints, controller_names, out.end_joints, error))
  {
    return false;
  }
  for (auto& pt : out.points)
  {
    std::vector<double> mapped;
    if (!remapByJointName(in.joint_names, pt.positions, controller_names, mapped, error))
    {
      return false;
    }
    pt.positions = mapped;
    if (!pt.velocities.empty())
    {
      if (!remapByJointName(in.joint_names, pt.velocities, controller_names, mapped, error))
      {
        return false;
      }
      pt.velocities = mapped;
    }
    if (!pt.accelerations.empty())
    {
      if (!remapByJointName(in.joint_names, pt.accelerations, controller_names, mapped, error))
      {
        return false;
      }
      pt.accelerations = mapped;
    }
  }
  return true;
}

StateCheckResult checkNamedJoints(const JointSnapshot& snap, const std::vector<std::string>& names,
                                  const std::vector<double>& expected, double tolerance_rad,
                                  const std::string& error_code, const std::string& segment)
{
  StateCheckResult result;
  result.segment = segment;
  std::string error;
  if (!snapshotHasRequiredJoints(snap, error))
  {
    result.error = error;
    return result;
  }
  std::vector<std::string> expected_names = names.empty() ? armNames() : names;
  if (!remapByJointName(expected_names, expected, armNames(), result.expected, error))
  {
    result.error = error;
    return result;
  }
  result.actual = snapshotArm(snap);
  result.max_error = maxAbsError(result.expected, result.actual);
  if (!std::isfinite(result.max_error) || result.max_error > tolerance_rad)
  {
    result.error = error_code;
    return result;
  }
  result.ok = true;
  return result;
}

StateCheckResult checkHome(const JointSnapshot& snap, const std::vector<double>& home,
                           double tolerance_rad)
{
  return checkNamedJoints(snap, armNames(), home, tolerance_rad, "HOME_STATE_MISMATCH", "Home");
}

StateCheckResult checkSegmentStart(const JointSnapshot& snap, const TrajectorySegmentRecord& seg,
                                   double tolerance_rad)
{
  std::vector<double> expected = seg.start_joints;
  std::vector<std::string> names = seg.joint_names;
  if (expected.size() < kArmJoints.size() && !seg.points.empty())
  {
    expected = seg.points.front().positions;
  }
  if (names.empty())
  {
    names = armNames();
  }
  return checkNamedJoints(snap, names, expected, tolerance_rad, kErrorFrozenStartMismatch,
                          seg.logical_segment);
}

StateCheckResult checkSegmentEnd(const JointSnapshot& snap, const TrajectorySegmentRecord& seg,
                                 double tolerance_rad)
{
  std::vector<double> expected = seg.end_joints;
  std::vector<std::string> names = seg.joint_names;
  if (expected.size() < kArmJoints.size() && !seg.points.empty())
  {
    expected = seg.points.back().positions;
  }
  if (names.empty())
  {
    names = armNames();
  }
  return checkNamedJoints(snap, names, expected, tolerance_rad, kErrorSegmentEndMismatch,
                          seg.logical_segment);
}

bool snapshotHasRequiredJoints(const JointSnapshot& snap, std::string& error)
{
  for (const auto& name : kArmJoints)
  {
    const auto it = snap.joints.find(name);
    if (it == snap.joints.end())
    {
      error = std::string(kErrorMissingJoint) + " joint=" + name;
      return false;
    }
    if (!std::isfinite(it->second))
    {
      error = std::string(kErrorNanTrajectory) + " joint=" + name;
      return false;
    }
  }
  return true;
}

bool snapshotFresh(const JointSnapshot& snap, double max_age_sec, std::string& error)
{
  if (!snap.stamp_valid)
  {
    error = "joint_states timestamp invalid";
    return false;
  }
  if (snap.age_sec > max_age_sec)
  {
    error = "joint_states stale age=" + std::to_string(snap.age_sec) + "s";
    return false;
  }
  return true;
}

bool snapshotVelocityUnavailable(const JointSnapshot& snap)
{
  if (snap.velocities.empty())
  {
    return true;
  }
  for (const auto& name : kArmJoints)
  {
    const auto it = snap.velocities.find(name);
    if (it == snap.velocities.end() || !std::isfinite(it->second))
    {
      return true;
    }
  }
  return false;
}

SettleResult waitForJointConvergence(const std::string& segment,
                                     const std::vector<std::string>& target_names,
                                     const std::vector<double>& target_values, double tolerance_rad,
                                     const ExecutorConfig& cfg, ExecutorHooks hooks)
{
  SettleResult result;
  result.segment = segment;
  result.best_max_error = std::numeric_limits<double>::infinity();
  result.last_max_error = std::numeric_limits<double>::infinity();
  result.immediate_end_error_rad = std::numeric_limits<double>::infinity();
  result.settled_end_error_rad = std::numeric_limits<double>::infinity();
  const int required = std::max(1, cfg.joint_settle_required_samples);
  const double timeout = std::max(0.0, cfg.joint_settle_timeout_sec);
  const double poll = std::max(0.0, cfg.joint_settle_poll_period_sec);
  const double t0 = nowFrom(hooks);
  double last_log_elapsed = -1.0;
  bool logged_velocity = false;
  bool first_log = true;

  auto emit = [&](const char* title, double elapsed) {
    logLine(hooks, formatSettle(result, elapsed, tolerance_rad, title));
    if (!result.expected.empty() || !result.actual.empty())
    {
      logLine(hooks, std::string("  target ") + formatJoints(result.expected) + " actual " +
                         formatJoints(result.actual));
    }
  };

  while (true)
  {
    const double elapsed = nowFrom(hooks) - t0;
    result.settle_elapsed_sec = elapsed;
    if (elapsed > timeout)
    {
      break;
    }
    if (!hooks.readJoints)
    {
      result.error = "joint_states unavailable";
      break;
    }
    auto snap = hooks.readJoints();
    if (!snap)
    {
      result.consecutive_ok = 0;
      if (first_log || elapsed - last_log_elapsed >= 1.0)
      {
        logLine(hooks, std::string("WAITING FOR JOINT SETTLE segment=") + segment +
                           " elapsed=" + std::to_string(elapsed) + " joint_states unavailable");
        last_log_elapsed = elapsed;
        first_log = false;
      }
    }
    else
    {
      if (snapshotVelocityUnavailable(*snap))
      {
        result.velocity_unavailable = true;
        if (!logged_velocity)
        {
          logLine(hooks, "velocity unavailable / NaN; position-only settle");
          logged_velocity = true;
        }
      }
      std::string local;
      const bool have_joints = snapshotHasRequiredJoints(*snap, local);
      const bool fresh = snapshotFresh(*snap, cfg.joint_state_max_age_sec, local);
      if (!have_joints || !fresh)
      {
        result.consecutive_ok = 0;
        result.error = local;
        if (first_log || elapsed - last_log_elapsed >= 1.0)
        {
          logLine(hooks, std::string("WAITING FOR JOINT SETTLE segment=") + segment +
                             " elapsed=" + std::to_string(elapsed) + " skipped=" + local);
          last_log_elapsed = elapsed;
          first_log = false;
        }
      }
      else
      {
        auto check = checkNamedJoints(*snap, target_names, target_values, tolerance_rad,
                                      kErrorSegmentEndMismatch, segment);
        result.expected = check.expected;
        result.actual = check.actual;
        result.last_max_error = check.max_error;
        result.samples_received += 1;
        if (!result.had_valid_sample)
        {
          result.immediate_end_error_rad = check.max_error;
          result.had_valid_sample = true;
        }
        result.best_max_error = std::min(result.best_max_error, check.max_error);
        if (check.ok)
        {
          result.consecutive_ok += 1;
          result.error.clear();
          if (result.consecutive_ok >= required)
          {
            result.ok = true;
            result.settled_end_error_rad = check.max_error;
            result.settle_elapsed_sec = nowFrom(hooks) - t0;
            emit("JOINT SETTLED", result.settle_elapsed_sec);
            logLine(hooks, "  immediate_end_error_rad=" +
                               std::to_string(result.immediate_end_error_rad) +
                               " settled_end_error_rad=" +
                               std::to_string(result.settled_end_error_rad) +
                               " settle_elapsed_sec=" + std::to_string(result.settle_elapsed_sec));
            return result;
          }
        }
        else
        {
          result.consecutive_ok = 0;
          result.error = check.error;
        }
        if (first_log || elapsed - last_log_elapsed >= 1.0)
        {
          emit("WAITING FOR JOINT SETTLE", elapsed);
          last_log_elapsed = elapsed;
          first_log = false;
        }
      }
    }
    if (nowFrom(hooks) - t0 > timeout)
    {
      break;
    }
    sleepFrom(hooks, poll);
  }

  result.ok = false;
  result.settle_elapsed_sec = nowFrom(hooks) - t0;
  if (result.error.empty() || result.error == kErrorSegmentEndMismatch)
  {
    result.error = (segment == "Home") ? kErrorHomeSettleTimeout : kErrorSegmentEndSettleTimeout;
  }
  else
  {
    result.error = std::string((segment == "Home") ? kErrorHomeSettleTimeout :
                                                      kErrorSegmentEndSettleTimeout) +
                   " " + result.error;
  }
  emit("JOINT SETTLE TIMEOUT", result.settle_elapsed_sec);
  logLine(hooks, "  timeout=" + std::to_string(timeout) + " required_samples=" +
                     std::to_string(required) + " consecutive_reached=" +
                     std::to_string(result.consecutive_ok) + " samples_received=" +
                     std::to_string(result.samples_received) + " last_max_error=" +
                     std::to_string(result.last_max_error) + " best_max_error=" +
                     std::to_string(result.best_max_error));
  return result;
}

ExecutorTrace runFrozenExecutor(const ExecutorConfig& cfg, const PersistedTrajectory& traj,
                                ExecutorHooks hooks)
{
  ExecutorTrace trace;
  std::string reason;
  trace.motion_enabled = motionAllowed(cfg, reason);
  trace.dry_run = !trace.motion_enabled;

  auto push = [&](ExecutorPhase phase) {
    trace.phases.push_back(phaseName(phase));
    trace.final_phase = phase;
    logLine(hooks, std::string("STATE ") + phaseName(phase));
  };

  push(ExecutorPhase::INIT);
  push(ExecutorPhase::VALIDATE_CONFIG);
  std::string error;
  if (!validSpeedScale(cfg.trajectory_speed_scale, error))
  {
    abortTo(trace, error);
    return trace;
  }
  auto deployable = collectDeployableSegments(traj, error);
  if (deployable.empty())
  {
    abortTo(trace, error);
    return trace;
  }
  std::vector<TrajectorySegmentRecord> scaled;
  if (!scaleDeployableTrajectory(traj, cfg.trajectory_speed_scale, scaled, error))
  {
    abortTo(trace, error);
    return trace;
  }
  trace.original_duration = computeDuration(deployable, false);
  for (const auto& seg : scaled)
  {
    trace.scaled_duration += segmentDuration(seg);
    trace.segments_prepared.push_back(seg.logical_segment);
  }
  const auto continuity = validateContinuity(traj);
  if (!continuity.ok)
  {
    abortTo(trace, "frozen continuity failed: " + continuity.reason);
    return trace;
  }
  if (!traj.winner.home_rad.empty())
  {
    const auto endpoint = validateEndpoints(traj, traj.winner);
    if (!endpoint.ok)
    {
      abortTo(trace, "frozen endpoint regression failed: " + endpoint.reason);
      return trace;
    }
  }

  push(ExecutorPhase::LOAD_FROZEN_TRAJECTORY);
  logLine(hooks, "Frozen path: NO REPLAN; Current→Home: RUNTIME REPLAN");

  push(ExecutorPhase::WAIT_FOR_JOINT_STATE);
  std::optional<JointSnapshot> snap;
  if (hooks.readJoints)
  {
    snap = hooks.readJoints();
  }
  if (trace.motion_enabled)
  {
    if (!snap)
    {
      abortTo(trace, "joint_states unavailable");
      return trace;
    }
    if (!snapshotHasRequiredJoints(*snap, error) ||
        !snapshotFresh(*snap, cfg.joint_state_max_age_sec, error))
    {
      abortTo(trace, error);
      return trace;
    }
  }

  push(ExecutorPhase::CHECK_REAL_MODE);
  if (hooks.useSimTimeInvalid && hooks.useSimTimeInvalid())
  {
    if (trace.motion_enabled)
    {
      abortTo(trace, kErrorSimTimeInvalid);
      return trace;
    }
    logLine(hooks, std::string("WARNING ") + kErrorSimTimeInvalid + " ignored in dry-run");
  }
  if (hooks.gazeboDetected && hooks.gazeboDetected() && trace.motion_enabled)
  {
    abortTo(trace, "Gazebo/sim hardware detected; refusing real executor motion");
    return trace;
  }
  if (trace.motion_enabled && hooks.controllerReady && !hooks.controllerReady())
  {
    abortTo(trace, "trajectory controller/action not ready");
    return trace;
  }

  push(ExecutorPhase::PLAN_CURRENT_TO_HOME);
  PlanResult plan;
  if (hooks.planCurrentToHome)
  {
    plan = hooks.planCurrentToHome();
  }
  else
  {
    plan.attempted = true;
    plan.success = true;
  }
  if (plan.attempted)
  {
    ++trace.current_to_home_plans;
  }
  if (plan.used_frozen_current_to_home)
  {
    trace.used_frozen_current_to_home = true;
    abortTo(trace, "Current→Home must be runtime replanned");
    return trace;
  }
  if ((trace.motion_enabled || cfg.plan_current_to_home) && !plan.success)
  {
    abortTo(trace, plan.error.empty() ? "Current→Home plan failed" : plan.error);
    return trace;
  }

  push(ExecutorPhase::WAIT_USER_EXECUTION_GATE);
  logLine(hooks, trace.motion_enabled ? "REAL ROBOT MOTION IS ENABLED" :
                                        "DRY RUN ONLY; motion gate closed: " + reason);
  if (!trace.motion_enabled)
  {
    trace.events.push_back("DRY_RUN_GRIPPER_OPEN");
    trace.events.push_back("DRY_RUN_GRIPPER_CLOSE");
    trace.events.push_back("DRY_RUN_ATTACH");
    trace.events.push_back("DRY_RUN_RESTORE_TABLE");
    trace.gripper_open_before_pregrasp = true;
    trace.gripper_before_attach = true;
    trace.gripper_close_before_lift = true;
    trace.attach_before_lift = true;
    push(ExecutorPhase::COMPLETE);
    trace.ok = true;
    return trace;
  }

  if (cfg.gripper_enabled && hooks.pingGripper)
  {
    push(ExecutorPhase::GRIPPER_PING_PREFLIGHT);
    trace.events.push_back("GRIPPER_PING_PREFLIGHT");
    auto ping = hooks.pingGripper();
    if (ping.sent)
    {
      ++trace.gripper_commands_sent;
    }
    if (!ping.success)
    {
      abortTo(trace, ping.error.empty() ? kErrorGripperPingFailed : ping.error);
      return trace;
    }
    trace.events.push_back("GRIPPER_PING_PASS");
  }

  auto refreshOk = [&]() -> bool {
    if (!hooks.readJoints)
    {
      abortTo(trace, "joint_states unavailable");
      return false;
    }
    snap = hooks.readJoints();
    if (!snap)
    {
      abortTo(trace, "joint_states unavailable");
      return false;
    }
    std::string local;
    if (!snapshotHasRequiredJoints(*snap, local) ||
        !snapshotFresh(*snap, cfg.joint_state_max_age_sec, local))
    {
      abortTo(trace, local);
      return false;
    }
    return true;
  };

  auto sendLogical = [&](const std::string& logical) -> bool {
    const auto* frozen = findLogical(deployable, logical);
    const auto* scaled_seg = findLogical(scaled, logical);
    if (!frozen || !scaled_seg)
    {
      abortTo(trace, "missing scaled/frozen segment " + logical);
      return false;
    }
    if (!refreshOk())
    {
      return false;
    }
    auto start = checkSegmentStart(*snap, *frozen, cfg.segment_start_tolerance_rad);
    logLine(hooks, "segment " + logical + " expected start " + formatJoints(start.expected) +
                       " actual " + formatJoints(start.actual) +
                       " max_error=" + std::to_string(start.max_error));
    if (!start.ok)
    {
      abortTo(trace, start.error + " segment=" + logical);
      return false;
    }
    if (!hooks.sendSegment)
    {
      abortTo(trace, "sendSegment hook missing");
      return false;
    }
    auto sent = hooks.sendSegment(logical, *scaled_seg);
    if (sent.sent)
    {
      ++trace.trajectories_sent;
      trace.segments_sent.push_back(logical);
    }
    if (!sent.success)
    {
      abortTo(trace, sent.error.empty() ? ("controller failure on " + logical) : sent.error);
      return false;
    }
    const auto end_names = frozen->joint_names.empty() ? armNames() : frozen->joint_names;
    const auto end_values = (frozen->end_joints.size() < kArmJoints.size() && !frozen->points.empty()) ?
                                frozen->points.back().positions :
                                frozen->end_joints;
    auto settle = waitForJointConvergence(logical, end_names, end_values,
                                          cfg.segment_end_tolerance_rad, cfg, hooks);
    logLine(hooks, "segment " + logical + " immediate_end_error_rad=" +
                       std::to_string(settle.immediate_end_error_rad) +
                       " settled_end_error_rad=" + std::to_string(settle.settled_end_error_rad) +
                       " settle_elapsed_sec=" + std::to_string(settle.settle_elapsed_sec));
    if (!settle.ok)
    {
      abortTo(trace, settle.error + " segment=" + logical);
      return false;
    }
    if (hooks.readJoints)
    {
      snap = hooks.readJoints();
    }
    return true;
  };

  push(ExecutorPhase::EXECUTE_CURRENT_TO_HOME);
  if (!hooks.sendSegment)
  {
    abortTo(trace, "sendSegment hook missing");
    return trace;
  }
  if (plan.planned.points.empty())
  {
    abortTo(trace, "Current→Home planned trajectory empty");
    return trace;
  }
  {
    auto home_send = hooks.sendSegment("Current_to_Home_runtime", plan.planned);
    if (home_send.sent)
    {
      ++trace.trajectories_sent;
      ++trace.current_to_home_executes;
    }
    if (!home_send.success)
    {
      abortTo(trace, home_send.error.empty() ? "Current→Home execute failed" : home_send.error);
      return trace;
    }
  }

  push(ExecutorPhase::VERIFY_HOME);
  {
    const auto home_vec =
        traj.winner.home_rad.empty() ? kStep12cHomeRad : jointsToVec(traj.winner.home_rad);
    auto settle = waitForJointConvergence("Home", armNames(), home_vec, cfg.home_tolerance_rad, cfg,
                                          hooks);
    logLine(hooks, "Home immediate_end_error_rad=" + std::to_string(settle.immediate_end_error_rad) +
                       " settled_end_error_rad=" + std::to_string(settle.settled_end_error_rad) +
                       " settle_elapsed_sec=" + std::to_string(settle.settle_elapsed_sec));
    if (!settle.ok)
    {
      abortTo(trace, settle.error);
      return trace;
    }
    if (hooks.readJoints)
    {
      snap = hooks.readJoints();
    }
  }

  auto requireGripperMotionDone = [&](SendResult grip, const GripperBridgeRequestFields& req,
                                      const char* prefix, const char* missing_hook) -> bool {
    if (grip.sent)
    {
      ++trace.gripper_commands_sent;
    }
    GripperCallOutcome call;
    call.ok = grip.success;
    call.error = grip.error;
    call.error_code = grip.error_code;
    call.message = grip.message;
    call.elapsed_sec = grip.elapsed_sec;
    call.response_received = grip.response_received;
    if (!grip.success && grip.error.find(kErrorGripperServiceUnavailable) != std::string::npos)
    {
      call.kind = GripperCallKind::ServiceUnavailable;
    }
    else if (!grip.success && grip.error.find(kErrorGripperServiceTimeout) != std::string::npos)
    {
      call.kind = GripperCallKind::Timeout;
    }
    else if (!grip.success && grip.error.find(kErrorGripperNullResponse) != std::string::npos)
    {
      call.kind = GripperCallKind::NullResponse;
    }
    else if (!grip.success)
    {
      call.kind = GripperCallKind::BridgeError;
    }
    GripperCompletionKind completion = grip.gripper_completion;
    if (completion == GripperCompletionKind::Unknown)
    {
      completion = classifyGripperCompletion(call, req);
    }
    trace.last_gripper_completion = completion;
    logLine(hooks, std::string("GRIPPER COMPLETION ") + gripperCompletionName(completion) +
                       " error_code=" + std::to_string(grip.error_code) + " message=\"" +
                       grip.message + "\" elapsed_sec=" + std::to_string(grip.elapsed_sec));
    if (!grip.success || !gripperBridgeMotionDone(completion))
    {
      std::string reason = grip.error;
      if (reason.empty())
      {
        reason = missing_hook;
      }
      if (!gripperBridgeMotionDone(completion))
      {
        reason = formatGripperCompletionFailure(call, completion, prefix);
      }
      abortTo(trace, reason);
      return false;
    }
    return true;
  };

  push(ExecutorPhase::OPEN_REAL_GRIPPER);
  trace.events.push_back("OPEN_REAL_GRIPPER");
  if (cfg.gripper_enabled)
  {
    if (!hooks.openGripper)
    {
      abortTo(trace, kErrorGripperOpenFailed);
      return trace;
    }
    logLine(hooks, "Home settled; opening real gripper to position=0. Confirm fingers are clear.");
    auto opened = hooks.openGripper();
    if (!requireGripperMotionDone(opened, makeGripperOpenRequest(cfg), kErrorGripperOpenFailed,
                                  kErrorGripperOpenFailed))
    {
      return trace;
    }
  }
  push(ExecutorPhase::WAIT_GRIPPER_OPEN_DONE);
  trace.events.push_back("WAIT_GRIPPER_OPEN_DONE");
  trace.gripper_open_before_pregrasp = true;

  push(ExecutorPhase::VERIFY_FROZEN_START);
  if (!refreshOk())
  {
    return trace;
  }
  {
    auto start = checkSegmentStart(*snap, deployable.front(), cfg.start_state_tolerance_rad);
    logLine(hooks, "frozen start expected " + formatJoints(start.expected) + " actual " +
                       formatJoints(start.actual) + " max_error=" + std::to_string(start.max_error));
    if (!start.ok)
    {
      abortTo(trace, start.error);
      return trace;
    }
  }

  push(ExecutorPhase::EXECUTE_HOME_TO_PREGRASP);
  if (!sendLogical(kLogicalHomeToPreGrasp))
  {
    return trace;
  }
  push(ExecutorPhase::VERIFY_PREGRASP);

  push(ExecutorPhase::EXECUTE_PREGRASP_TO_GRASP);
  if (!sendLogical(kLogicalPreGraspToGrasp))
  {
    return trace;
  }
  push(ExecutorPhase::VERIFY_GRASP);

  push(ExecutorPhase::CLOSE_REAL_GRIPPER);
  trace.events.push_back("CLOSE_REAL_GRIPPER");
  if (cfg.gripper_enabled)
  {
    if (!hooks.closeGripper)
    {
      abortTo(trace, kErrorGripperCloseFailed);
      return trace;
    }
    auto grip = hooks.closeGripper();
    if (!requireGripperMotionDone(grip, makeGripperCloseRequest(cfg), kErrorGripperCloseFailed,
                                  kErrorGripperCloseFailed))
    {
      return trace;
    }
  }
  push(ExecutorPhase::WAIT_GRIPPER_CLOSE_DONE);
  trace.events.push_back("WAIT_GRIPPER_CLOSE_DONE");
  push(ExecutorPhase::VERIFY_GRIPPER_CLOSED);
  trace.events.push_back("VERIFY_GRIPPER_CLOSED");
  trace.gripper_before_attach = true;
  trace.gripper_close_before_lift = true;

  push(ExecutorPhase::ATTACH_PLANNING_SCENE_OBJECT);
  trace.events.push_back("ATTACH_PLANNING_SCENE_OBJECT");
  if (!hooks.attachObject)
  {
    abortTo(trace, "attach hook missing");
    return trace;
  }
  {
    auto attached = hooks.attachObject();
    if (attached.sent || attached.success)
    {
      ++trace.attach_calls;
    }
    if (!attached.success)
    {
      abortTo(trace, attached.error.empty() ? "PlanningScene attach failed" : attached.error);
      return trace;
    }
  }
  trace.gripper_before_attach = true;
  trace.attach_before_lift = true;

  push(ExecutorPhase::EXECUTE_GRASP_TO_LIFT);
  if (!sendLogical(kLogicalGraspToLift))
  {
    return trace;
  }
  push(ExecutorPhase::VERIFY_LIFT);

  push(ExecutorPhase::RESTORE_PART_TABLE_COLLISION);
  trace.events.push_back("RESTORE_PART_TABLE_COLLISION");
  if (hooks.restoreTableCollision)
  {
    auto restored = hooks.restoreTableCollision();
    if (restored.success)
    {
      ++trace.restore_table_calls;
    }
    if (!restored.success)
    {
      abortTo(trace, restored.error.empty() ? "restore part-table collision failed" :
                                              restored.error);
      return trace;
    }
  }

  push(ExecutorPhase::EXECUTE_LIFT_TO_A);
  if (!sendLogical(kLogicalLiftToA))
  {
    return trace;
  }
  push(ExecutorPhase::VERIFY_A);

  push(ExecutorPhase::EXECUTE_A_TO_B);
  if (!sendLogical(kLogicalAToB))
  {
    return trace;
  }
  push(ExecutorPhase::VERIFY_B);

  push(ExecutorPhase::EXECUTE_B_TO_C);
  if (!sendLogical(kLogicalBToC))
  {
    return trace;
  }
  push(ExecutorPhase::VERIFY_C);
  push(ExecutorPhase::COMPLETE);
  trace.ok = true;
  return trace;
}
}  // namespace fr_task_planner
