#include "fr_task_planner/frozen_executor.hpp"
#include "fr_task_planner/winner_trajectory_io.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

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

void reportSegment(const char* name, const LogicalMetrics& m)
{
  const double j1_start = m.start_joints.empty() ? 0.0 : m.start_joints[0];
  const double j1_end = m.end_joints.empty() ? 0.0 : m.end_joints[0];
  std::cout << name << " j1_start=" << j1_start << " j1_end=" << j1_end
            << " j1_net=" << std::abs(j1_end - j1_start)
            << " j1_cum=" << m.j1_abs_travel << " all_joint=" << m.all_joint_abs_travel
            << " duration=" << m.duration << "\n";
}
}  // namespace

int main()
{
  std::string error;
  WinnerEndpoints winner12c;
  check(loadWinnerYaml(FR_TASK_PLANNER_STEP12C_WINNER_PATH, winner12c, error),
        "load STEP12C winner");
  check(winnerMatchesFrozenStep12c(winner12c, error), "STEP12C A/B/C/Home frozen");

  PersistedTrajectory step14;
  check(readTrajectoryYaml(FR_TASK_PLANNER_STEP14_TRAJ_PATH, step14, error), "load STEP14 trajectory");
  const FrozenTaskMetrics m14 = computeFrozenTaskMetrics(step14);
  reportSegment("STEP14 Home->PreGrasp", m14.home_to_pregrasp);
  reportSegment("STEP14 PreGrasp->Grasp", m14.pregrasp_to_grasp);
  reportSegment("STEP14 Grasp->Lift", m14.grasp_to_lift);
  reportSegment("STEP14 Lift->A", m14.lift_to_a);
  reportSegment("STEP14 A->B", m14.a_to_b);
  reportSegment("STEP14 B->C", m14.b_to_c);
  const double full14 = m14.home_to_pregrasp.j1_abs_travel + m14.pregrasp_to_grasp.j1_abs_travel +
                        m14.grasp_to_lift.j1_abs_travel + m14.lift_to_a.j1_abs_travel +
                        m14.a_to_b.j1_abs_travel + m14.b_to_c.j1_abs_travel;
  std::cout << "STEP14 full Home->C J1 cumulative=" << full14 << " duration="
            << m14.full_deployable_duration << "\n";
  check(m14.lift_to_a.j1_abs_travel > 2.8, "STEP14 Lift->A J1 still large before STEP15 search");
  check(maxAbsError(jointsToVec(step14.winner.b_rad), kStep12cBRad) < 1e-3, "STEP14 B joints frozen");
  check(maxAbsError(jointsToVec(step14.winner.c_rad), kStep12cCRad) < 1e-3, "STEP14 C joints frozen");

  auto open_req = makeGripperOpenRequest(ExecutorConfig{});
  auto close_req = makeGripperCloseRequest(ExecutorConfig{});
  check(open_req.position == 0 && close_req.position == 85, "open=0 close=85");
  auto ping = classifyGripperCall(true, false, false, 0, "ok", 0.1, kErrorGripperPingFailed);
  check(classifyGripperCompletion(ping, makeGripperPingRequest(ExecutorConfig{})) ==
            GripperCompletionKind::CommandAccepted,
        "ping is COMMAND_ACCEPTED not gripper closed");
  auto done = classifyGripperCall(true, false, false, 0, kGripperBridgeMotionDoneMessage, 1.0,
                                  kErrorGripperCloseFailed);
  check(classifyGripperCompletion(done, close_req) == GripperCompletionKind::BridgeMotionDone,
        "MoveGripper done is BRIDGE_MOTION_DONE not PHYSICAL_GRASP_CONFIRMED");

  std::ifstream step15_traj(FR_TASK_PLANNER_STEP15_TRAJ_PATH);
  if (!step15_traj)
  {
    std::cout << "SKIP STEP15 trajectory not generated yet\n";
    std::cout << "passed=" << g_passes << " failed=" << g_fails << "\n";
    return g_fails == 0 ? 0 : 1;
  }
  PersistedTrajectory neu;
  check(readTrajectoryYaml(FR_TASK_PLANNER_STEP15_TRAJ_PATH, neu, error), "load STEP15 trajectory");
  const FrozenTaskMetrics m15 = computeFrozenTaskMetrics(neu);
  reportSegment("STEP15 Home->PreGrasp", m15.home_to_pregrasp);
  reportSegment("STEP15 PreGrasp->Grasp", m15.pregrasp_to_grasp);
  reportSegment("STEP15 Grasp->Lift", m15.grasp_to_lift);
  reportSegment("STEP15 Lift->A", m15.lift_to_a);
  reportSegment("STEP15 A->B", m15.a_to_b);
  reportSegment("STEP15 B->C", m15.b_to_c);
  const double full15 = m15.home_to_pregrasp.j1_abs_travel + m15.pregrasp_to_grasp.j1_abs_travel +
                        m15.grasp_to_lift.j1_abs_travel + m15.lift_to_a.j1_abs_travel +
                        m15.a_to_b.j1_abs_travel + m15.b_to_c.j1_abs_travel;
  std::cout << "STEP15 full Home->C J1 cumulative=" << full15 << " duration="
            << m15.full_deployable_duration << "\n";
  check(m15.lift_to_a.j1_abs_travel <= 0.5 + 1e-6, "STEP15 Lift->A J1 <= 0.5 rad");
  check(m15.home_to_pregrasp.j1_abs_travel <= 0.5 + 1e-6, "STEP15 Home->PreGrasp J1 <= 0.5 rad");
  check(m15.pregrasp_to_grasp.j1_abs_travel <= 0.5 + 1e-6, "STEP15 PreGrasp->Grasp J1 <= 0.5 rad");
  check(m15.grasp_to_lift.j1_abs_travel <= 0.5 + 1e-6, "STEP15 Grasp->Lift J1 <= 0.5 rad");
  check(m15.a_to_b.j1_abs_travel <= 0.5 + 1e-6, "STEP15 A->B J1 <= 0.5 rad");
  check(m15.b_to_c.j1_abs_travel <= 0.5 + 1e-6, "STEP15 B->C J1 <= 0.5 rad");
  check(full15 < full14, "full Home->C J1 smaller than STEP14");
  WinnerEndpoints w15;
  if (loadWinnerYaml(FR_TASK_PLANNER_STEP15_WINNER_PATH, w15, error))
  {
    check(!w15.a_rad.empty() && !w15.b_rad.empty() && !w15.c_rad.empty(),
          "winner YAML has A/B/C joints");
  }
  PersistedTrajectory loaded;
  check(readTrajectoryYaml(FR_TASK_PLANNER_STEP15_TRAJ_PATH, loaded, error), "STEP15 reload");
  check(validateRoundTrip(neu, loaded).ok, "STEP15 scored/saved round-trip");
  check(validateContinuity(neu, 1e-3).ok, "STEP15 continuity");

  std::cout << "passed=" << g_passes << " failed=" << g_fails << "\n";
  return g_fails == 0 ? 0 : 1;
}
