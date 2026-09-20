#include "fr_task_planner/frozen_executor.hpp"
#include "fr_task_planner/inspection_endpoint_candidates.hpp"
#include "fr_task_planner/winner_trajectory_io.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace fr_task_planner;

namespace
{
int g_fails = 0;
int g_passes = 0;

void check(bool cond, const std::string& name)
{
  if (cond)
  {
    ++g_passes;
    std::cout << "PASS " << name << "\n";
  }
  else
  {
    ++g_fails;
    std::cerr << "FAIL " << name << "\n";
  }
}

JointSnapshot snapFrom(const std::vector<double>& values)
{
  JointSnapshot snap;
  snap.stamp_valid = true;
  snap.age_sec = 0.01;
  snap.joints = vecToJoints(values);
  return snap;
}

JointSnapshot snapNamed(const std::vector<std::string>& names, const std::vector<double>& values,
                        double age = 0.01)
{
  JointSnapshot snap;
  snap.stamp_valid = true;
  snap.age_sec = age;
  const size_t n = std::min(names.size(), values.size());
  for (size_t i = 0; i < n; ++i)
  {
    snap.joints[names[i]] = values[i];
  }
  return snap;
}

TrajectoryPointRecord pt(const std::vector<double>& pos, const std::vector<double>& vel,
                         const std::vector<double>& acc, double t)
{
  TrajectoryPointRecord rec;
  rec.positions = pos;
  rec.velocities = vel;
  rec.accelerations = acc;
  setTimeFromStart(rec, t);
  return rec;
}

TrajectorySegmentRecord makeSeg(const std::string& logical, const std::vector<double>& a,
                                const std::vector<double>& b, double duration)
{
  TrajectorySegmentRecord seg;
  seg.name = logical;
  seg.logical_segment = logical;
  seg.deployable = true;
  seg.runtime_replan_required = false;
  seg.joint_names.assign(kArmJoints.begin(), kArmJoints.end());
  seg.start_joints = a;
  seg.end_joints = b;
  seg.velocities_present = true;
  seg.accelerations_present = true;
  const std::vector<double> vel = { 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 };
  const std::vector<double> acc = { 0.8, 0.8, 0.8, 0.8, 0.8, 0.8 };
  seg.points.push_back(pt(a, { 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0 }, 0.0));
  seg.points.push_back(pt(b, vel, acc, duration));
  seg.duration = duration;
  return seg;
}

PersistedTrajectory makeFrozen()
{
  PersistedTrajectory traj;
  traj.trajectory_format_version = 1;
  traj.task_version = "STEP12C";
  traj.joint_names.assign(kArmJoints.begin(), kArmJoints.end());
  traj.winner.home_rad = vecToJoints(kStep12cHomeRad);
  traj.winner.a_rad = vecToJoints(kStep12cARad);
  traj.winner.b_rad = vecToJoints(kStep12cBRad);
  traj.winner.c_rad = vecToJoints(kStep12cCRad);
  traj.winner.a_roll_deg = -90.0;
  traj.winner.b_roll_deg = -130.0;
  traj.winner.c_roll_deg = -85.0;
  auto current = kStep12cHomeRad;
  current[0] += 0.1;
  TrajectorySegmentRecord cur = makeSeg(kLogicalCurrentToHome, current, kStep12cHomeRad, 0.2);
  cur.deployable = false;
  cur.runtime_replan_required = true;
  traj.segments.push_back(cur);
  const std::vector<double> pre = { -2.2, -1.6, -1.8, -2.8, 0.0, 0.8 };
  const std::vector<double> grasp = { -2.1, -1.5, -1.7, -2.7, 0.0, 0.8 };
  const std::vector<double> lift = { -2.1, -1.4, -1.6, -2.7, 0.0, 0.8 };
  traj.segments.push_back(makeSeg(kLogicalHomeToPreGrasp, kStep12cHomeRad, pre, 1.0));
  traj.segments.push_back(makeSeg(kLogicalPreGraspToGrasp, pre, grasp, 0.4));
  traj.segments.push_back(makeSeg(kLogicalGraspToLift, grasp, lift, 0.4));
  traj.segments.push_back(makeSeg(kLogicalLiftToA, lift, kStep12cARad, 1.0));
  traj.segments.push_back(makeSeg(kLogicalAToB, kStep12cARad, kStep12cBRad, 1.0));
  traj.segments.push_back(makeSeg(kLogicalBToC, kStep12cBRad, kStep12cCRad, 1.0));
  attachStandardEvents(traj);
  fillDerivedTotals(traj);
  return traj;
}

bool nearly(double a, double b, double tol = 1e-9)
{
  return std::abs(a - b) <= tol;
}

bool vecNear(const std::vector<double>& a, const std::vector<double>& b, double tol = 1e-12)
{
  if (a.size() != b.size())
  {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i)
  {
    if (std::abs(a[i] - b[i]) > tol)
    {
      return false;
    }
  }
  return true;
}

struct MockRobot
{
  JointSnapshot joints = snapFrom(kStep12cHomeRad);
  bool gripper_ok = true;
  bool open_ok = true;
  bool open_pending = false;
  bool close_pending = false;
  bool servo_resume_fail = false;
  bool ping_ok = true;
  bool ping_enabled = false;
  bool attach_ok = true;
  bool restore_ok = true;
  bool controller_ok = true;
  bool skip_joint_update = false;
  std::string fail_segment;
  int send_calls = 0;
  int gripper_calls = 0;
  int open_calls = 0;
  int ping_calls = 0;
  int attach_calls = 0;
  std::vector<std::string> sent;
  std::vector<std::string> events;
  bool used_frozen_home = false;
  double time_sec = 0.0;
  int open_error_code = 0;
  int close_error_code = 0;
  std::string open_message = kGripperBridgeMotionDoneMessage;
  std::string close_message = kGripperBridgeMotionDoneMessage;
  std::string open_error;
  std::string close_error;

  SendResult makeGripperMove(bool ok, bool pending, bool servo_fail, int error_code,
                             const std::string& message, const std::string& error,
                             const char* fallback)
  {
    SendResult r;
    r.attempted = true;
    r.sent = true;
    r.response_received = true;
    r.elapsed_sec = 1.2;
    r.error_code = error_code;
    r.message = message;
    if (pending)
    {
      r.success = true;
      r.error_code = 0;
      r.message = "motion still pending";
      r.gripper_completion = GripperCompletionKind::MotionPending;
      return r;
    }
    if (servo_fail)
    {
      r.success = false;
      r.error_code = error_code == 0 ? -1 : error_code;
      r.message = "ServoMoveStart failed";
      r.error = std::string(kErrorGripperServoJResumeFailed) + " " + r.message;
      r.gripper_completion = GripperCompletionKind::ServoJResumeFailure;
      return r;
    }
    if (!ok)
    {
      r.success = false;
      r.error = error.empty() ? fallback : error;
      r.error_code = error_code == 0 ? 123 : error_code;
      if (r.message == kGripperBridgeMotionDoneMessage)
      {
        r.message = "MoveGripper failed";
      }
      if (r.error.find(kErrorGripperServiceTimeout) != std::string::npos ||
          r.message.find("timeout") != std::string::npos)
      {
        r.gripper_completion = GripperCompletionKind::MotionDoneTimeout;
        r.response_received = false;
      }
      else
      {
        r.gripper_completion = GripperCompletionKind::Unknown;
      }
      return r;
    }
    r.success = true;
    r.error_code = 0;
    r.message = kGripperBridgeMotionDoneMessage;
    r.gripper_completion = GripperCompletionKind::BridgeMotionDone;
    return r;
  }

  ExecutorHooks hooks()
  {
    ExecutorHooks h;
    h.readJoints = [this]() { return std::optional<JointSnapshot>(joints); };
    h.planCurrentToHome = [this]() {
      PlanResult p;
      p.attempted = true;
      p.success = true;
      p.used_frozen_current_to_home = used_frozen_home;
      p.planned = makeSeg("Current_to_Home_runtime", jointsToVec(joints.joints), kStep12cHomeRad, 0.5);
      return p;
    };
    h.sendSegment = [this](const std::string& logical, const TrajectorySegmentRecord& seg) {
      SendResult r;
      r.attempted = true;
      ++send_calls;
      if (!fail_segment.empty() && (logical == fail_segment || seg.logical_segment == fail_segment))
      {
        r.error = "controller failure";
        return r;
      }
      r.sent = true;
      r.success = true;
      sent.push_back(logical);
      if (!skip_joint_update)
      {
        joints = snapFrom(seg.end_joints.empty() ? (seg.points.empty() ? kStep12cHomeRad :
                                                                          extractArmPositions(seg.joint_names,
                                                                                              seg.points.back().positions)) :
                                                   extractArmPositions(seg.joint_names.empty() ?
                                                                           std::vector<std::string>(kArmJoints.begin(),
                                                                                                    kArmJoints.end()) :
                                                                           seg.joint_names,
                                                                       seg.end_joints));
      }
      return r;
    };
    h.openGripper = [this]() {
      SendResult r = makeGripperMove(open_ok, open_pending, false, open_error_code, open_message,
                                     open_error, kErrorGripperOpenFailed);
      ++open_calls;
      events.push_back("OPEN_REAL_GRIPPER");
      return r;
    };
    h.closeGripper = [this]() {
      SendResult r = makeGripperMove(gripper_ok, close_pending, servo_resume_fail, close_error_code,
                                     close_message, close_error, kErrorGripperCloseFailed);
      ++gripper_calls;
      events.push_back("CLOSE_REAL_GRIPPER");
      return r;
    };
    if (ping_enabled)
    {
      h.pingGripper = [this]() {
        SendResult r;
        r.attempted = true;
        ++ping_calls;
        r.sent = true;
        events.push_back("GRIPPER_PING_PREFLIGHT");
        if (!ping_ok)
        {
          r.error = kErrorGripperPingFailed;
          return r;
        }
        r.success = true;
        return r;
      };
    }
    h.attachObject = [this]() {
      SendResult r;
      r.attempted = true;
      ++attach_calls;
      events.push_back("ATTACH");
      r.sent = true;
      r.success = attach_ok;
      if (!attach_ok)
      {
        r.error = "attach failed";
      }
      return r;
    };
    h.restoreTableCollision = [this]() {
      SendResult r;
      r.attempted = true;
      r.success = restore_ok;
      r.sent = restore_ok;
      return r;
    };
    h.useSimTimeInvalid = []() { return false; };
    h.gazeboDetected = []() { return false; };
    h.controllerReady = [this]() { return controller_ok; };
    h.nowSec = [this]() { return time_sec; };
    h.sleepSec = [this](double s) { time_sec += s; };
    return h;
  }
};

ExecutorConfig execCfg()
{
  ExecutorConfig cfg;
  cfg.execute = true;
  cfg.real_robot_confirmation = kRealRobotConfirmation;
  cfg.trajectory_speed_scale = 1.0;
  cfg.plan_current_to_home = true;
  return cfg;
}

struct SeqReader
{
  std::vector<JointSnapshot> seq;
  size_t index = 0;
  int reads = 0;
  int send_calls = 0;
  double time_sec = 0.0;

  std::optional<JointSnapshot> next()
  {
    ++reads;
    if (seq.empty())
    {
      return std::nullopt;
    }
    if (index < seq.size())
    {
      return seq[index++];
    }
    return seq.back();
  }

  ExecutorHooks hooks()
  {
    ExecutorHooks h;
    h.readJoints = [this]() { return next(); };
    h.nowSec = [this]() { return time_sec; };
    h.sleepSec = [this](double s) { time_sec += s; };
    h.sendSegment = [this](const std::string&, const TrajectorySegmentRecord&) {
      ++send_calls;
      SendResult r;
      r.attempted = true;
      r.error = "settle helper must not send motion";
      return r;
    };
    return h;
  }
};

ExecutorConfig settleCfg()
{
  ExecutorConfig cfg;
  cfg.joint_settle_timeout_sec = 10.0;
  cfg.joint_settle_poll_period_sec = 0.10;
  cfg.joint_settle_required_samples = 3;
  cfg.joint_state_max_age_sec = 0.5;
  cfg.segment_end_tolerance_rad = 0.02;
  cfg.home_tolerance_rad = 0.02;
  return cfg;
}
}  // namespace

int main(int argc, char** argv)
{
  const char* traj_path = argc > 1 ? argv[1] : FR_TASK_PLANNER_TRAJ_PATH;
  PersistedTrajectory frozen;
  std::string error;
  const bool loaded = readTrajectoryYaml(traj_path, frozen, error);
  check(loaded, "TEST1_frozen_trajectory_parse");
  if (loaded)
  {
    auto segs = collectDeployableSegments(frozen, error);
    check(segs.size() == 6, "TEST2_exact_6_deployable_segments");
    bool current_rejected = true;
    for (const auto& seg : segs)
    {
      if (seg.logical_segment == kLogicalCurrentToHome)
      {
        current_rejected = false;
      }
    }
    std::string reject_error;
    check(rejectCurrentToHomeAsDeployable(frozen, reject_error),
          "TEST3_current_to_home_not_deployable");
    check(current_rejected, "TEST3b_collected_skips_current_to_home");
    auto mutated = frozen;
    for (auto& seg : mutated.segments)
    {
      if (seg.logical_segment == kLogicalCurrentToHome)
      {
        seg.deployable = true;
      }
    }
    check(!rejectCurrentToHomeAsDeployable(mutated, reject_error),
          "TEST3c_current_to_home_marked_deployable_rejected");

    std::vector<TrajectorySegmentRecord> scaled1;
    check(scaleDeployableTrajectory(frozen, 1.0, scaled1, error), "TEST5_scale_1_ok");
    const auto original = collectDeployableSegments(frozen, error);
    bool identical = scaled1.size() == original.size();
    bool pos_same = true;
    for (size_t i = 0; i < original.size() && i < scaled1.size(); ++i)
    {
      if (original[i].points.size() != scaled1[i].points.size())
      {
        identical = false;
        break;
      }
      for (size_t k = 0; k < original[i].points.size(); ++k)
      {
        pos_same = pos_same && vecNear(original[i].points[k].positions, scaled1[i].points[k].positions);
        identical = identical && original[i].points[k].sec == scaled1[i].points[k].sec &&
                    original[i].points[k].nanosec == scaled1[i].points[k].nanosec &&
                    vecNear(original[i].points[k].velocities, scaled1[i].points[k].velocities) &&
                    vecNear(original[i].points[k].accelerations, scaled1[i].points[k].accelerations);
      }
    }
    check(pos_same, "TEST4_positions_unchanged_scale1");
    check(identical, "TEST5_scale_1_exact");

    auto checkScale = [&](double scale, const char* name) {
      std::vector<TrajectorySegmentRecord> scaled;
      bool ok = scaleDeployableTrajectory(frozen, scale, scaled, error);
      bool pass = ok && scaled.size() == original.size();
      for (size_t i = 0; i < original.size() && pass; ++i)
      {
        const double dur0 = segmentDuration(original[i]);
        const double dur1 = segmentDuration(scaled[i]);
        pass = pass && nearly(dur1, dur0 / scale, 1e-6);
        for (size_t k = 0; k < original[i].points.size(); ++k)
        {
          pass = pass && vecNear(original[i].points[k].positions, scaled[i].points[k].positions);
          pass = pass && nearly(timeFromStartSec(scaled[i].points[k]),
                                timeFromStartSec(original[i].points[k]) / scale, 1e-6);
          if (!original[i].points[k].velocities.empty())
          {
            std::vector<double> expect = original[i].points[k].velocities;
            for (double& v : expect)
            {
              v *= scale;
            }
            pass = pass && vecNear(expect, scaled[i].points[k].velocities, 1e-9);
          }
          if (!original[i].points[k].accelerations.empty())
          {
            std::vector<double> expect = original[i].points[k].accelerations;
            for (double& a : expect)
            {
              a *= scale * scale;
            }
            pass = pass && vecNear(expect, scaled[i].points[k].accelerations, 1e-9);
          }
        }
      }
      check(pass, name);
    };
    checkScale(0.5, "TEST6_scale_0_5");
    checkScale(0.05, "TEST7_scale_0_05");
    std::vector<TrajectorySegmentRecord> bad;
    check(!scaleDeployableTrajectory(frozen, 0.0, bad, error), "TEST8_scale_zero_rejected");
    check(!scaleDeployableTrajectory(frozen, -0.1, bad, error), "TEST8b_scale_negative_rejected");
    check(!scaleDeployableTrajectory(frozen, 1.1, bad, error), "TEST8c_scale_gt1_rejected");
  }
  else
  {
    std::cerr << "could not load " << traj_path << ": " << error << "\n";
  }

  auto dummy = makeFrozen();
  TrajectorySegmentRecord remapped;
  std::vector<std::string> reverse = { "j6", "j5", "j4", "j3", "j2", "j1" };
  check(remapSegmentByJointName(dummy.segments[1], reverse, remapped, error), "TEST9_remap_ok");
  check(remapped.joint_names == reverse, "TEST9b_remap_names");
  check(nearly(remapped.points.front().positions.back(), dummy.segments[1].points.front().positions.front()),
        "TEST9c_remap_values");

  auto home_snap = snapFrom(kStep12cHomeRad);
  auto home_ok = checkHome(home_snap, kStep12cHomeRad, 0.02);
  check(home_ok.ok, "TEST12_home_pass");
  auto home_bad_vals = kStep12cHomeRad;
  home_bad_vals[0] += 0.05;
  auto home_fail = checkHome(snapFrom(home_bad_vals), kStep12cHomeRad, 0.02);
  check(!home_fail.ok, "TEST12b_home_fail");

  auto start_ok = checkSegmentStart(home_snap, dummy.segments[1], 0.02);
  check(start_ok.ok, "TEST10_start_pass");
  auto start_fail = checkSegmentStart(snapFrom(home_bad_vals), dummy.segments[1], 0.02);
  check(!start_fail.ok && start_fail.error == kErrorFrozenStartMismatch, "TEST10b_start_fail");
  auto end_ok = checkSegmentEnd(snapFrom(dummy.segments[1].end_joints), dummy.segments[1], 0.02);
  check(end_ok.ok, "TEST11_end_pass");
  auto end_fail = checkSegmentEnd(home_snap, dummy.segments[1], 0.02);
  check(!end_fail.ok && end_fail.error == kErrorSegmentEndMismatch, "TEST11b_end_fail");

  JointSnapshot missing = home_snap;
  missing.joints.erase("j3");
  std::string missing_error;
  check(!snapshotHasRequiredJoints(missing, missing_error), "TEST13_missing_joint");

  auto nan_seg = dummy.segments[1];
  nan_seg.points[0].positions[0] = std::numeric_limits<double>::quiet_NaN();
  check(!trajectoryHasFiniteValues(nan_seg, error), "TEST14_nan_rejected");

  auto nonmono = dummy.segments[1];
  nonmono.points.push_back(pt(nonmono.end_joints, nonmono.points.back().velocities,
                              nonmono.points.back().accelerations, 0.1));
  check(!timesMonotonic(nonmono), "TEST15_nonmonotonic_time");

  ExecutorConfig dry;
  dry.trajectory_speed_scale = 1.0;
  MockRobot robot;
  auto dry_trace = runFrozenExecutor(dry, dummy, robot.hooks());
  check(dry_trace.dry_run && dry_trace.ok && robot.send_calls == 0 &&
            robot.gripper_calls == 0 && robot.open_calls == 0,
        "TEST16_execute_false_sends_nothing");

  ExecutorConfig exec_no_confirm;
  exec_no_confirm.execute = true;
  exec_no_confirm.trajectory_speed_scale = 1.0;
  MockRobot robot2;
  auto t17 = runFrozenExecutor(exec_no_confirm, dummy, robot2.hooks());
  check(t17.dry_run && robot2.send_calls == 0, "TEST17_execute_true_without_confirmation");

  ExecutorConfig exec_wrong;
  exec_wrong.execute = true;
  exec_wrong.real_robot_confirmation = "yes";
  exec_wrong.trajectory_speed_scale = 1.0;
  MockRobot robot3;
  auto t18 = runFrozenExecutor(exec_wrong, dummy, robot3.hooks());
  check(t18.dry_run && robot3.send_calls == 0, "TEST18_wrong_confirmation");

  MockRobot robot4;
  robot4.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1], kStep12cHomeRad[2],
                             kStep12cHomeRad[3], kStep12cHomeRad[4], kStep12cHomeRad[5] });
  auto t19 = runFrozenExecutor(execCfg(), dummy, robot4.hooks());
  bool grip_before_attach = false;
  for (size_t i = 0; i + 1 < t19.events.size(); ++i)
  {
    if (t19.events[i] == "CLOSE_REAL_GRIPPER")
    {
      for (size_t j = i + 1; j < t19.events.size(); ++j)
      {
        if (t19.events[j] == "ATTACH_PLANNING_SCENE_OBJECT")
        {
          grip_before_attach = true;
        }
      }
    }
  }
  check(t19.ok && t19.gripper_before_attach && grip_before_attach && robot4.gripper_calls == 1 &&
            robot4.open_calls == 1 && robot4.attach_calls == 1 && t19.gripper_open_before_pregrasp &&
            t19.gripper_close_before_lift,
        "TEST19_gripper_before_attach");
  bool lift_after_attach = false;
  for (const auto& s : t19.segments_sent)
  {
    if (s == kLogicalGraspToLift)
    {
      lift_after_attach = true;
    }
  }
  check(t19.attach_before_lift && lift_after_attach, "TEST19b_attach_before_lift");

  MockRobot robot5;
  robot5.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1], kStep12cHomeRad[2],
                             kStep12cHomeRad[3], kStep12cHomeRad[4], kStep12cHomeRad[5] });
  robot5.gripper_ok = false;
  auto t20 = runFrozenExecutor(execCfg(), dummy, robot5.hooks());
  bool lift_sent = false;
  for (const auto& s : t20.segments_sent)
  {
    lift_sent = lift_sent || s == kLogicalGraspToLift;
  }
  check(t20.aborted && t20.abort_reason.find(kErrorGripperCloseFailed) != std::string::npos &&
            !lift_sent && robot5.attach_calls == 0,
        "TEST20_gripper_failure_blocks_lift");
  check(!lift_sent && robot5.attach_calls == 0, "GRIPPER_G_close_failure_blocks_attach_and_lift");

  MockRobot robot6;
  robot6.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1], kStep12cHomeRad[2],
                             kStep12cHomeRad[3], kStep12cHomeRad[4], kStep12cHomeRad[5] });
  robot6.fail_segment = kLogicalHomeToPreGrasp;
  auto t21 = runFrozenExecutor(execCfg(), dummy, robot6.hooks());
  bool later = false;
  for (const auto& s : t21.segments_sent)
  {
    later = later || s == kLogicalPreGraspToGrasp || s == kLogicalGraspToLift ||
            s == kLogicalLiftToA || s == kLogicalAToB || s == kLogicalBToC;
  }
  check(t21.aborted && !later, "TEST21_segment_failure_blocks_later");

  MockRobot robot7;
  robot7.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1], kStep12cHomeRad[2],
                             kStep12cHomeRad[3], kStep12cHomeRad[4], kStep12cHomeRad[5] });
  auto t22 = runFrozenExecutor(execCfg(), dummy, robot7.hooks());
  bool sent_frozen_current = false;
  for (const auto& s : t22.segments_sent)
  {
    sent_frozen_current = sent_frozen_current || s == kLogicalCurrentToHome;
  }
  check(t22.ok && t22.current_to_home_plans >= 1 && t22.current_to_home_executes >= 1 &&
            !t22.used_frozen_current_to_home && !sent_frozen_current,
        "TEST22_current_to_home_runtime_planned");

  check(t22.frozen_replans == 0 && t22.ok, "TEST23_frozen_home_to_c_never_replanned");
  bool all_frozen_sent = true;
  for (const auto& logical : deployableLogicalOrder())
  {
    bool found = false;
    for (const auto& s : t22.segments_sent)
    {
      found = found || s == logical;
    }
    all_frozen_sent = all_frozen_sent && found;
  }
  check(all_frozen_sent, "TEST23b_all_frozen_segments_sent_without_replan");

  const std::vector<std::string> unordered = { "j1", "j2", "j4", "j5", "j3", "j6" };
  const std::vector<double> grasp_target = { 0.419211129,  0.145426114, 0.272553076,
                                            -0.417949142, 2.775405619, 0.785428936 };
  const std::vector<std::string> arm = { "j1", "j2", "j3", "j4", "j5", "j6" };
  auto unorderedFromArm = [&](const std::vector<double>& arm_vals) {
    std::map<std::string, double> m;
    for (size_t i = 0; i < arm.size(); ++i)
    {
      m[arm[i]] = arm_vals[i];
    }
    std::vector<double> ordered;
    for (const auto& n : unordered)
    {
      ordered.push_back(m[n]);
    }
    return snapNamed(unordered, ordered);
  };

  {
    SeqReader r;
    r.seq = {
      unorderedFromArm({ 0.447345250, 0.122331173, 0.325040642, -0.447409798, 2.803547230, 0.785390573 }),
      unorderedFromArm({ 0.434000000, 0.133000000, 0.302000000, -0.432000000, 2.790000000, 0.785410000 }),
      unorderedFromArm({ 0.425000000, 0.140000000, 0.287000000, -0.423000000, 2.780000000, 0.785420000 }),
      unorderedFromArm({ 0.422000000, 0.143000000, 0.280000000, -0.420000000, 2.777000000, 0.785425000 }),
      unorderedFromArm({ 0.419209618638, 0.145428219687, 0.272547373326, -0.417949011336,
                         2.775404008962, 0.785432318438 }),
    };
    auto s = waitForJointConvergence("PreGrasp_to_Grasp", arm, grasp_target, 0.02, settleCfg(),
                                     r.hooks());
    check(s.ok && s.consecutive_ok >= 3 && s.immediate_end_error_rad > 0.05 &&
              s.settled_end_error_rad < 1e-5 && r.send_calls == 0,
          "SETTLE_real_case_unordered_converges");
  }

  {
    SeqReader r;
    auto good = unorderedFromArm(grasp_target);
    r.seq = { good, good, good };
    auto s = waitForJointConvergence("A", arm, grasp_target, 0.02, settleCfg(), r.hooks());
    check(s.ok && s.samples_received == 3 && s.immediate_end_error_rad <= 0.02, "SETTLE_A_immediate_pass");
  }

  {
    SeqReader r;
    auto bad = unorderedFromArm({ 0.447345250, 0.122331173, 0.325040642, -0.447409798, 2.803547230,
                                  0.785390573 });
    auto mid = unorderedFromArm({ 0.425000000, 0.140000000, 0.287000000, -0.423000000, 2.780000000,
                                  0.785420000 });
    auto good = unorderedFromArm(grasp_target);
    r.seq = { bad, mid, good, good, good };
    auto s = waitForJointConvergence("B", arm, grasp_target, 0.02, settleCfg(), r.hooks());
    check(s.ok && s.immediate_end_error_rad > 0.02, "SETTLE_B_later_converge");
  }

  {
    SeqReader r;
    auto bad = unorderedFromArm({ 0.447345250, 0.122331173, 0.325040642, -0.447409798, 2.803547230,
                                  0.785390573 });
    r.seq = { bad };
    auto cfg = settleCfg();
    cfg.joint_settle_timeout_sec = 0.35;
    auto s = waitForJointConvergence("C", arm, grasp_target, 0.02, cfg, r.hooks());
    check(!s.ok && s.error.find(kErrorSegmentEndSettleTimeout) != std::string::npos,
          "SETTLE_C_timeout_fail");
  }

  {
    SeqReader r;
    auto good = unorderedFromArm(grasp_target);
    auto bad = unorderedFromArm({ 0.447345250, 0.122331173, 0.325040642, -0.447409798, 2.803547230,
                                  0.785390573 });
    r.seq = { good, bad, good, good, good };
    auto s = waitForJointConvergence("D", arm, grasp_target, 0.02, settleCfg(), r.hooks());
    check(s.ok && s.samples_received >= 5, "SETTLE_D_consecutive_reset");
  }

  {
    SeqReader r;
    auto stale = unorderedFromArm(grasp_target);
    stale.age_sec = 1.0;
    r.seq = { stale };
    auto cfg = settleCfg();
    cfg.joint_settle_timeout_sec = 0.35;
    auto s = waitForJointConvergence("E", arm, grasp_target, 0.02, cfg, r.hooks());
    check(!s.ok && s.samples_received == 0, "SETTLE_E_stale_not_success");
  }

  {
    SeqReader r;
    JointSnapshot missing;
    missing.stamp_valid = true;
    missing.age_sec = 0.01;
    missing.joints["j1"] = grasp_target[0];
    missing.joints["j2"] = grasp_target[1];
    missing.joints["j4"] = grasp_target[3];
    missing.joints["j5"] = grasp_target[4];
    missing.joints["j6"] = grasp_target[5];
    r.seq = { missing };
    auto cfg = settleCfg();
    cfg.joint_settle_timeout_sec = 0.35;
    auto s = waitForJointConvergence("F", arm, grasp_target, 0.02, cfg, r.hooks());
    check(!s.ok && s.samples_received == 0, "SETTLE_F_missing_j3");
  }

  {
    SeqReader r;
    auto good = unorderedFromArm(grasp_target);
    r.seq = { good, good, good };
    auto s = waitForJointConvergence("G", arm, grasp_target, 0.02, settleCfg(), r.hooks());
    check(s.ok, "SETTLE_G_unordered_namemap");
  }

  {
    SeqReader r;
    auto good = unorderedFromArm(grasp_target);
    for (const auto& n : arm)
    {
      good.velocities[n] = std::numeric_limits<double>::quiet_NaN();
    }
    r.seq = { good, good, good };
    auto s = waitForJointConvergence("H", arm, grasp_target, 0.02, settleCfg(), r.hooks());
    check(s.ok && s.velocity_unavailable, "SETTLE_H_velocity_nan_still_pass");
  }

  {
    SeqReader r;
    auto good = unorderedFromArm(grasp_target);
    r.seq = { good, good, good };
    auto s = waitForJointConvergence("J", arm, grasp_target, 0.02, settleCfg(), r.hooks());
    check(s.ok && s.motion_commands_sent == 0 && r.send_calls == 0, "SETTLE_J_helper_sends_no_motion");
  }

  {
    auto success = classifyGripperCall(true, false, false, 0, "ok", 1.2, kErrorGripperCloseFailed);
    check(success.ok && success.kind == GripperCallKind::Success, "GRIPPER_A_success");
  }
  {
    auto fail = classifyGripperCall(true, false, false, 123, "MoveGripper failed", 9.0,
                                    kErrorGripperCloseFailed);
    check(!fail.ok && fail.kind == GripperCallKind::BridgeError &&
              fail.error.find(kErrorGripperCloseFailed) != std::string::npos &&
              fail.error.find("123") != std::string::npos &&
              fail.error.find("MoveGripper failed") != std::string::npos &&
              fail.error.find(kErrorGripperBridgeError) != std::string::npos,
          "GRIPPER_B_bridge_error_keeps_code_and_message");
  }
  {
    auto timeout = classifyGripperCall(true, true, false, 0, "", 15.002, kErrorGripperCloseFailed);
    check(!timeout.ok && timeout.kind == GripperCallKind::Timeout &&
              timeout.error.find(kErrorGripperServiceTimeout) != std::string::npos &&
              timeout.error.find("15.002") != std::string::npos &&
              timeout.error.find(kErrorGripperCloseFailed) == std::string::npos,
          "GRIPPER_C_service_timeout");
  }
  {
    auto null_resp =
        classifyGripperCall(true, false, true, 0, "", 0.4, kErrorGripperCloseFailed);
    check(!null_resp.ok && null_resp.kind == GripperCallKind::NullResponse &&
              null_resp.error.find(kErrorGripperNullResponse) != std::string::npos,
          "GRIPPER_D_null_response");
  }
  {
    auto unavailable =
        classifyGripperCall(false, false, false, 0, "", 2.0, kErrorGripperCloseFailed);
    check(!unavailable.ok && unavailable.kind == GripperCallKind::ServiceUnavailable &&
              unavailable.error.find(kErrorGripperServiceUnavailable) != std::string::npos,
          "GRIPPER_UNAVAILABLE");
  }

  {
    MockRobot ping_ok_robot;
    ping_ok_robot.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1], kStep12cHomeRad[2],
                                      kStep12cHomeRad[3], kStep12cHomeRad[4], kStep12cHomeRad[5] });
    ping_ok_robot.ping_enabled = true;
    ping_ok_robot.ping_ok = true;
    auto t_ping = runFrozenExecutor(execCfg(), dummy, ping_ok_robot.hooks());
    bool ping_pass = false;
    for (const auto& e : t_ping.events)
    {
      ping_pass = ping_pass || e == "GRIPPER_PING_PASS";
    }
    check(t_ping.ok && ping_ok_robot.ping_calls == 1 && ping_pass, "GRIPPER_E_ping_preflight_pass");
  }
  {
    MockRobot ping_fail_robot;
    ping_fail_robot.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1],
                                        kStep12cHomeRad[2], kStep12cHomeRad[3], kStep12cHomeRad[4],
                                        kStep12cHomeRad[5] });
    ping_fail_robot.ping_enabled = true;
    ping_fail_robot.ping_ok = false;
    auto t_ping_fail = runFrozenExecutor(execCfg(), dummy, ping_fail_robot.hooks());
    check(t_ping_fail.aborted && ping_fail_robot.send_calls == 0 &&
              ping_fail_robot.gripper_calls == 0 && ping_fail_robot.open_calls == 0 &&
              ping_fail_robot.attach_calls == 0 &&
              t_ping_fail.abort_reason.find(kErrorGripperPingFailed) != std::string::npos,
          "GRIPPER_F_ping_failure_blocks_sequence");
  }
  {
    ExecutorConfig dry_g;
    dry_g.trajectory_speed_scale = 1.0;
    MockRobot dry_robot;
    auto dry_g_trace = runFrozenExecutor(dry_g, dummy, dry_robot.hooks());
    check(dry_g_trace.dry_run && dry_g_trace.ok && dry_robot.send_calls == 0 &&
              dry_robot.gripper_calls == 0 && dry_robot.open_calls == 0 &&
              dry_robot.ping_calls == 0 && dry_g_trace.gripper_commands_sent == 0,
          "GRIPPER_H_execute_false_zero_gripper_commands");
  }
  {
    ExecutorConfig cfg;
    cfg.gripper_id = 1;
    cfg.gripper_close_position = 85;
    cfg.gripper_velocity = 20;
    cfg.gripper_force = 20;
    cfg.gripper_max_time_ms = 5000;
    cfg.gripper_block = 1;
    cfg.gripper_type = 0;
    cfg.gripper_rot_num = 0.0;
    cfg.gripper_rot_vel = 0;
    cfg.gripper_rot_torque = 0;
    auto close_req = makeGripperCloseRequest(cfg);
    auto ping_req = makeGripperPingRequest(cfg);
    check(close_req.command == kGripperCommandMove && close_req.gripper_id == 1 &&
              close_req.position == 85 && close_req.velocity == 20 && close_req.force == 20 &&
              close_req.max_time_ms == 5000 && close_req.block == 1 && close_req.gripper_type == 0 &&
              close_req.rot_num == 0.0 && close_req.rot_vel == 0 && close_req.rot_torque == 0 &&
              ping_req.command == kGripperCommandPing && ping_req.position == 0 &&
              ping_req.gripper_id == 1 && ping_req.rot_num == 0.0 && ping_req.rot_vel == 0 &&
              ping_req.rot_torque == 0,
          "GRIPPER_I_request_fields_match_config");
    auto open_req = makeGripperOpenRequest(cfg);
    check(open_req.command == kGripperCommandMove && open_req.position == 0 &&
              open_req.velocity == close_req.velocity && open_req.force == close_req.force &&
              open_req.max_time_ms == close_req.max_time_ms && open_req.block == close_req.block,
          "GRIPPER_I_open_request_position_0");
    const std::string req_log = formatGripperRequestLog(close_req);
    check(req_log.find("command=move") != std::string::npos &&
              req_log.find("rot_num=") != std::string::npos &&
              req_log.find("rot_vel=") != std::string::npos &&
              req_log.find("rot_torque=") != std::string::npos,
          "GRIPPER_I_request_log_includes_rot_fields");
  }

  auto hasSent = [](const std::vector<std::string>& sent, const char* logical) {
    for (const auto& s : sent)
    {
      if (s == logical)
      {
        return true;
      }
    }
    return false;
  };
  auto hasPhase = [](const ExecutorTrace& t, const char* name) {
    for (const auto& p : t.phases)
    {
      if (p == name)
      {
        return true;
      }
    }
    return false;
  };

  {
    MockRobot r;
    r.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1], kStep12cHomeRad[2],
                          kStep12cHomeRad[3], kStep12cHomeRad[4], kStep12cHomeRad[5] });
    r.skip_joint_update = true;
    auto cfg = execCfg();
    cfg.joint_settle_timeout_sec = 0.35;
    auto t = runFrozenExecutor(cfg, dummy, r.hooks());
    check(t.aborted && r.open_calls == 0 && !hasSent(r.sent, kLogicalHomeToPreGrasp),
          "HOME_OPEN_A_home_not_settled_skips_open");
  }
  {
    MockRobot r;
    r.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1], kStep12cHomeRad[2],
                          kStep12cHomeRad[3], kStep12cHomeRad[4], kStep12cHomeRad[5] });
    auto t = runFrozenExecutor(execCfg(), dummy, r.hooks());
    check(t.ok && r.open_calls == 1 && r.gripper_calls == 1 && t.gripper_open_before_pregrasp &&
              hasPhase(t, "OPEN_REAL_GRIPPER") && hasPhase(t, "WAIT_GRIPPER_OPEN_DONE") &&
              hasSent(r.sent, kLogicalHomeToPreGrasp) &&
              t.last_gripper_completion == GripperCompletionKind::BridgeMotionDone,
          "HOME_OPEN_B_home_settled_open_then_pregrasp");
  }
  {
    MockRobot r;
    r.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1], kStep12cHomeRad[2],
                          kStep12cHomeRad[3], kStep12cHomeRad[4], kStep12cHomeRad[5] });
    r.open_ok = false;
    r.open_error = kErrorGripperServiceTimeout;
    r.open_message = "GetGripperMotionDone timeout";
    auto t = runFrozenExecutor(execCfg(), dummy, r.hooks());
    check(t.aborted && r.open_calls == 1 && !hasSent(r.sent, kLogicalHomeToPreGrasp) &&
              t.abort_reason.find(kErrorGripperMotionDoneTimeout) != std::string::npos,
          "HOME_OPEN_C_open_timeout_blocks_pregrasp");
  }
  {
    MockRobot r;
    r.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1], kStep12cHomeRad[2],
                          kStep12cHomeRad[3], kStep12cHomeRad[4], kStep12cHomeRad[5] });
    r.open_ok = false;
    r.open_error_code = 7;
    r.open_message = "MoveGripper failed";
    auto t = runFrozenExecutor(execCfg(), dummy, r.hooks());
    check(t.aborted && r.open_calls == 1 && !hasSent(r.sent, kLogicalHomeToPreGrasp) &&
              t.abort_reason.find("error_code=7") != std::string::npos &&
              t.abort_reason.find("MoveGripper failed") != std::string::npos,
          "HOME_OPEN_D_open_nonzero_error_blocks_pregrasp");
  }
  {
    MockRobot r;
    r.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1], kStep12cHomeRad[2],
                          kStep12cHomeRad[3], kStep12cHomeRad[4], kStep12cHomeRad[5] });
    r.open_pending = true;
    auto t = runFrozenExecutor(execCfg(), dummy, r.hooks());
    check(t.aborted && r.open_calls == 1 && !hasSent(r.sent, kLogicalHomeToPreGrasp) &&
              t.abort_reason.find(kErrorGripperMotionPending) != std::string::npos,
          "HOME_OPEN_E_open_pending_blocks_pregrasp");
  }
  {
    MockRobot r;
    r.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1], kStep12cHomeRad[2],
                          kStep12cHomeRad[3], kStep12cHomeRad[4], kStep12cHomeRad[5] });
    r.fail_segment = kLogicalPreGraspToGrasp;
    auto t = runFrozenExecutor(execCfg(), dummy, r.hooks());
    check(t.aborted && r.open_calls == 1 && r.gripper_calls == 0 && r.attach_calls == 0 &&
              !hasSent(r.sent, kLogicalGraspToLift),
          "GRASP_CLOSE_A_grasp_not_reached_skips_close");
  }
  {
    MockRobot r;
    r.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1], kStep12cHomeRad[2],
                          kStep12cHomeRad[3], kStep12cHomeRad[4], kStep12cHomeRad[5] });
    r.close_pending = true;
    auto t = runFrozenExecutor(execCfg(), dummy, r.hooks());
    check(t.aborted && r.gripper_calls == 1 && r.attach_calls == 0 &&
              !hasSent(r.sent, kLogicalGraspToLift) &&
              t.abort_reason.find(kErrorGripperMotionPending) != std::string::npos &&
              hasPhase(t, "CLOSE_REAL_GRIPPER") && !hasPhase(t, "WAIT_GRIPPER_CLOSE_DONE"),
          "GRASP_CLOSE_B_success_but_pending_blocks_attach_and_lift");
  }
  {
    MockRobot r;
    r.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1], kStep12cHomeRad[2],
                          kStep12cHomeRad[3], kStep12cHomeRad[4], kStep12cHomeRad[5] });
    r.gripper_ok = false;
    r.close_error = kErrorGripperServiceTimeout;
    r.close_message = "GetGripperMotionDone timeout";
    auto t = runFrozenExecutor(execCfg(), dummy, r.hooks());
    check(t.aborted && r.attach_calls == 0 && !hasSent(r.sent, kLogicalGraspToLift) &&
              t.abort_reason.find(kErrorGripperMotionDoneTimeout) != std::string::npos,
          "GRASP_CLOSE_C_close_timeout_blocks_attach_and_lift");
  }
  {
    MockRobot r;
    r.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1], kStep12cHomeRad[2],
                          kStep12cHomeRad[3], kStep12cHomeRad[4], kStep12cHomeRad[5] });
    r.gripper_ok = false;
    r.close_error_code = 9;
    r.close_message = "MoveGripper failed";
    auto t = runFrozenExecutor(execCfg(), dummy, r.hooks());
    check(t.aborted && r.attach_calls == 0 && !hasSent(r.sent, kLogicalGraspToLift) &&
              t.abort_reason.find("error_code=9") != std::string::npos,
          "GRASP_CLOSE_D_close_nonzero_error_blocks_attach_and_lift");
  }
  {
    MockRobot r;
    r.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1], kStep12cHomeRad[2],
                          kStep12cHomeRad[3], kStep12cHomeRad[4], kStep12cHomeRad[5] });
    r.servo_resume_fail = true;
    auto t = runFrozenExecutor(execCfg(), dummy, r.hooks());
    check(t.aborted && r.attach_calls == 0 && !hasSent(r.sent, kLogicalGraspToLift) &&
              t.abort_reason.find(kErrorGripperServoJResumeFailed) != std::string::npos,
          "GRASP_CLOSE_E_servoj_resume_failure_blocks_lift");
  }
  {
    MockRobot r;
    r.joints = snapFrom({ kStep12cHomeRad[0] + 0.1, kStep12cHomeRad[1], kStep12cHomeRad[2],
                          kStep12cHomeRad[3], kStep12cHomeRad[4], kStep12cHomeRad[5] });
    r.attach_ok = false;
    auto t = runFrozenExecutor(execCfg(), dummy, r.hooks());
    check(t.aborted && r.gripper_calls == 1 && r.attach_calls == 1 &&
              !hasSent(r.sent, kLogicalGraspToLift),
          "GRASP_CLOSE_F_attach_failure_blocks_lift");
  }
  {
    auto done = classifyGripperCall(true, false, false, 0, kGripperBridgeMotionDoneMessage, 2.0,
                                    kErrorGripperCloseFailed);
    auto kind = classifyGripperCompletion(done, makeGripperCloseRequest(ExecutorConfig{}));
    check(kind == GripperCompletionKind::BridgeMotionDone && gripperBridgeMotionDone(kind),
          "GRIPPER_COMPLETION_move_done");
    auto ping_call = classifyGripperCall(true, false, false, 0, "ok", 0.1, kErrorGripperPingFailed);
    auto ping_kind = classifyGripperCompletion(ping_call, makeGripperPingRequest(ExecutorConfig{}));
    check(ping_kind == GripperCompletionKind::CommandAccepted && !gripperBridgeMotionDone(ping_kind),
          "GRIPPER_COMPLETION_ping_is_not_closed");
  }

  {
    struct StartupMock
    {
      std::optional<JointSnapshot> joints;
      double joints_after = 0.0;
      double action_after = 0.0;
      double gripper_after = 0.0;
      double time_sec = 0.0;
      StartupReadyHooks hooks()
      {
        StartupReadyHooks h;
        h.readJoints = [this]() -> std::optional<JointSnapshot> {
          if (!joints || time_sec < joints_after)
          {
            return std::nullopt;
          }
          return joints;
        };
        h.actionReady = [this]() { return time_sec >= action_after; };
        h.gripperReady = [this]() { return time_sec >= gripper_after; };
        h.nowSec = [this]() { return time_sec; };
        h.sleepSec = [this](double s) { time_sec += s; };
        return h;
      }
    };
    ExecutorConfig ready_cfg;
    ready_cfg.joint_state_max_age_sec = 0.5;

    {
      StartupMock m;
      m.joints = snapFrom(kStep12cHomeRad);
      auto st = waitForStartupInterfaces(ready_cfg, m.hooks(), 15.0);
      check(st.allReady() && st.received_joint_msg && st.total_wait_sec < 1.0 &&
                st.timeout_reason.empty(),
            "STARTUP_READY_all_immediate");
      bool saw_first = false;
      bool saw_all = false;
      for (const auto& line : st.logs)
      {
        saw_first = saw_first || line.find("first /joint_states received") != std::string::npos;
        saw_all = saw_all || line.find("all interfaces ready") != std::string::npos;
      }
      check(saw_first && saw_all, "STARTUP_READY_logs_first_msg_and_all_ready");
    }
    {
      StartupMock m;
      m.joints = snapFrom(kStep12cHomeRad);
      m.joints_after = 0.25;
      m.action_after = 0.40;
      m.gripper_after = 0.10;
      auto st = waitForStartupInterfaces(ready_cfg, m.hooks(), 15.0);
      check(st.allReady() && st.total_wait_sec >= 0.40 && st.total_wait_sec < 1.0 &&
                st.joints_wait_sec >= 0.25 && st.action_wait_sec >= 0.40 &&
                st.gripper_wait_sec >= 0.10 && st.first_joint_msg_elapsed_sec >= 0.25,
            "STARTUP_READY_early_exit_before_15s");
    }
    {
      StartupMock m;
      m.action_after = 0.0;
      m.gripper_after = 0.0;
      auto st = waitForStartupInterfaces(ready_cfg, m.hooks(), 0.35);
      check(!st.allReady() && !st.joints_ready && st.action_ready && st.gripper_ready &&
                st.timeout_reason.find(kErrorStartupInterfacesNotReady) != std::string::npos &&
                st.timeout_reason.find("joint_states=") != std::string::npos &&
                st.timeout_reason.find("trajectory_action=") == std::string::npos &&
                st.timeout_reason.find("gripper_service=") == std::string::npos,
            "STARTUP_READY_timeout_joint_states");
    }
    {
      StartupMock m;
      m.joints = snapFrom(kStep12cHomeRad);
      m.action_after = 99.0;
      m.gripper_after = 0.0;
      auto st = waitForStartupInterfaces(ready_cfg, m.hooks(), 0.35);
      check(!st.allReady() && st.joints_ready && !st.action_ready && st.gripper_ready &&
                st.timeout_reason.find("trajectory_action=not online") != std::string::npos &&
                st.timeout_reason.find("joint_states=") == std::string::npos &&
                st.timeout_reason.find("gripper_service=") == std::string::npos,
            "STARTUP_READY_timeout_action");
    }
    {
      StartupMock m;
      m.joints = snapFrom(kStep12cHomeRad);
      m.action_after = 0.0;
      m.gripper_after = 99.0;
      auto st = waitForStartupInterfaces(ready_cfg, m.hooks(), 0.35);
      check(!st.allReady() && st.joints_ready && st.action_ready && !st.gripper_ready &&
                st.timeout_reason.find("gripper_service=not online") != std::string::npos &&
                st.timeout_reason.find("joint_states=") == std::string::npos &&
                st.timeout_reason.find("trajectory_action=") == std::string::npos,
            "STARTUP_READY_timeout_gripper");
    }
    {
      StartupMock m;
      auto stale = snapFrom(kStep12cHomeRad);
      stale.age_sec = 1.0;
      m.joints = stale;
      auto st = waitForStartupInterfaces(ready_cfg, m.hooks(), 0.35);
      check(!st.allReady() && st.received_joint_msg && !st.joints_ready &&
                st.joints_reason.find("stale") != std::string::npos,
            "STARTUP_READY_stale_not_accepted");
    }
    {
      StartupMock m;
      JointSnapshot missing = snapFrom(kStep12cHomeRad);
      missing.joints.erase("j3");
      m.joints = missing;
      auto st = waitForStartupInterfaces(ready_cfg, m.hooks(), 0.35);
      check(!st.allReady() && st.received_joint_msg && !st.joints_ready &&
                st.joints_reason.find(kErrorMissingJoint) != std::string::npos,
            "STARTUP_READY_missing_j3_not_accepted");
    }
    {
      std::string fresh_error;
      auto fresh = snapFrom(kStep12cHomeRad);
      check(snapshotFresh(fresh, 0.5, fresh_error), "STARTUP_READY_freshness_unchanged_pass");
      auto stale = snapFrom(kStep12cHomeRad);
      stale.age_sec = 0.51;
      check(!snapshotFresh(stale, 0.5, fresh_error) &&
                checkSegmentStart(fresh, dummy.segments[1], 0.02).ok &&
                !checkSegmentStart(snapFrom(home_bad_vals), dummy.segments[1], 0.02).ok,
            "STARTUP_READY_does_not_relax_freshness_or_start_tolerance");
    }
  }

  std::cout << "passed=" << g_passes << " failed=" << g_fails << "\n";
  return g_fails == 0 ? 0 : 1;
}
