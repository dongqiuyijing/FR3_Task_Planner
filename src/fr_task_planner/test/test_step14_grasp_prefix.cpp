#include "fr_task_planner/pregrasp_ik_candidates.hpp"
#include "fr_task_planner/winner_trajectory_io.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
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
}  // namespace

int main()
{
  std::string error;
  PersistedTrajectory frozen;
  check(readTrajectoryYaml(FR_TASK_PLANNER_TRAJ_PATH, frozen, error), "load frozen trajectory");
  const FrozenTaskMetrics old_m = computeFrozenTaskMetrics(frozen);
  check(old_m.home_to_pregrasp.present, "Home_to_PreGrasp present");
  check(old_m.pregrasp_to_grasp.present, "PreGrasp_to_Grasp present");
  check(old_m.grasp_to_lift.present, "Grasp_to_Lift present");
  check(old_m.lift_to_a.present && old_m.a_to_b.present && old_m.b_to_c.present, "A/B/C present");
  check(std::abs(old_m.home_to_pregrasp.duration - 5.902542553) < 1e-9,
        "baseline Home->PreGrasp duration from points");
  check(std::abs(old_m.full_deployable_duration - frozen.persisted_fixed_task_total_duration) < 1e-12,
        "full Home->C duration recomputed from points");
  check(old_m.home_to_pregrasp.j1_abs_travel > 2.8 && old_m.home_to_pregrasp.j1_abs_travel < 2.9,
        "baseline J1 travel ~163 deg");
  check(std::abs(old_m.scaled_full_duration_at_005 - old_m.full_deployable_duration / 0.05) < 1e-12,
        "0.05 scale is duration/0.05");

  WinnerEndpoints winner;
  check(loadWinnerYaml(FR_TASK_PLANNER_WINNER_PATH, winner, error), "load winner yaml");
  check(winnerMatchesFrozenStep12c(winner, error), "A/B/C/Home frozen values");
  check(std::abs(winner.a_roll_deg + 90.0) < 1e-9 && std::abs(winner.b_roll_deg + 130.0) < 1e-9 &&
            std::abs(winner.c_roll_deg + 85.0) < 1e-9,
        "A/B/C rolls unchanged");

  const auto home = vecToJoints(kStep12cHomeRad);
  const auto baseline = vecToJoints(kStep12cPreGraspRad);
  check(isSameIkBranch(baseline, baseline, 0.1), "identical joints are same branch");
  auto near = baseline;
  near["j6"] += 0.01;
  check(isSameIkBranch(baseline, near, 0.1), "0.01 rad is duplicate at 0.1 threshold");
  auto other = baseline;
  other["j1"] -= 0.5;
  check(!isSameIkBranch(baseline, other, 0.1), "0.5 rad J1 is a different branch");

  std::vector<PreGraspIkCandidate> pool;
  PreGraspIkCandidate a;
  a.candidate_id = "baseline_frozen_pregrasp";
  a.joints = baseline;
  a.is_baseline = true;
  pool.push_back(a);
  PreGraspIkCandidate b;
  b.joints = other;
  check(findDuplicateIndex(pool, baseline, 0.1) == 0, "baseline is in candidate pool");
  check(findDuplicateIndex(pool, other, 0.1) < 0, "other IK is unique");
  check(pool.front().is_baseline, "baseline flagged");

  const double j1_direct = std::abs(baseline.at("j1") - home.at("j1"));
  check(std::abs(j1_direct - old_m.home_to_pregrasp.j1_abs_travel) < 0.02,
        "endpoint J1 displacement matches path-integrated travel");
  const double wrap = shortestValidJointDisplacement(home.at("j1"), baseline.at("j1"), -3.0543,
                                                     3.0543);
  check(std::abs(wrap - j1_direct) < 1e-12, "FR3 j1 cannot wrap 2pi inside limits");
  check(std::abs(shortestValidJointDisplacement(0.0, 0.1, -3.0, 3.0) - 0.1) < 1e-12,
        "short displacement");

  PersistedTrajectory again;
  check(readTrajectoryYaml(FR_TASK_PLANNER_TRAJ_PATH, again, error), "reload frozen");
  const auto rt = validateRoundTrip(frozen, again);
  check(rt.ok && rt.time_identical && rt.joint_names_identical, "frozen round-trip identical");

  std::ifstream step14_traj(FR_TASK_PLANNER_STEP14_TRAJ_PATH);
  if (step14_traj)
  {
    PersistedTrajectory neu;
    check(readTrajectoryYaml(FR_TASK_PLANNER_STEP14_TRAJ_PATH, neu, error), "load step14 trajectory");
    const FrozenTaskMetrics neu_m = computeFrozenTaskMetrics(neu);
    check(neu_m.home_to_pregrasp.present && neu_m.grasp_to_lift.present, "step14 has prefix segments");
    check(neu_m.lift_to_a.present && neu_m.a_to_b.present && neu_m.b_to_c.present,
          "step14 completes A/B/C");
    const auto ep = validateEndpoints(neu, winner, 1e-3);
    check(ep.ok, "step14 A/B/C joints match frozen winner");
    const auto cont = validateContinuity(neu, 1e-3);
    check(cont.ok, "step14 continuity");
    PersistedTrajectory loaded;
    check(readTrajectoryYaml(FR_TASK_PLANNER_STEP14_TRAJ_PATH, loaded, error), "step14 reload");
    const auto rt14 = validateRoundTrip(neu, loaded);
    check(rt14.ok, "step14 scored/saved round-trip");
    check(std::abs(neu_m.full_deployable_duration - computeFrozenTaskMetrics(loaded).full_deployable_duration) <
              1e-12,
          "step14 score duration equals saved duration");
    WinnerEndpoints step14_winner;
    if (loadWinnerYaml(FR_TASK_PLANNER_STEP14_WINNER_PATH, step14_winner, error))
    {
      check(winnerMatchesFrozenStep12c(step14_winner, error), "step14 winner keeps frozen A/B/C");
    }
  }
  else
  {
    std::cout << "SKIP step14 output files not generated yet\n";
  }

  std::cout << "passed=" << g_passes << " failed=" << g_fails << "\n";
  return g_fails == 0 ? 0 : 1;
}
