#include "fr_task_planner/frozen_executor.hpp"
#include "fr_task_planner/inspection_endpoint_candidates.hpp"
#include "fr_task_planner/winner_trajectory_io.hpp"

#include <cmath>
#include <iostream>
#include <limits>
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
  bool attach_ok = true;
  bool restore_ok = true;
  bool controller_ok = true;
  std::string fail_segment;
  int send_calls = 0;
  int gripper_calls = 0;
  int attach_calls = 0;
  std::vector<std::string> sent;
  std::vector<std::string> events;
  bool used_frozen_home = false;

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
      joints = snapFrom(seg.end_joints.empty() ? (seg.points.empty() ? kStep12cHomeRad :
                                                                        extractArmPositions(seg.joint_names,
                                                                                            seg.points.back().positions)) :
                                                 extractArmPositions(seg.joint_names.empty() ?
                                                                         std::vector<std::string>(kArmJoints.begin(),
                                                                                                  kArmJoints.end()) :
                                                                         seg.joint_names,
                                                                     seg.end_joints));
      return r;
    };
    h.closeGripper = [this]() {
      SendResult r;
      r.attempted = true;
      ++gripper_calls;
      events.push_back("CLOSE_REAL_GRIPPER");
      if (!gripper_ok)
      {
        r.error = kErrorGripperCloseFailed;
        return r;
      }
      r.sent = true;
      r.success = true;
      return r;
    };
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
            robot.gripper_calls == 0,
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
            robot4.attach_calls == 1,
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
            !lift_sent,
        "TEST20_gripper_failure_blocks_lift");

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

  std::cout << "passed=" << g_passes << " failed=" << g_fails << "\n";
  return g_fails == 0 ? 0 : 1;
}
