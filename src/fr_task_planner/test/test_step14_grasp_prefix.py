#!/usr/bin/env python3
"""STEP 14 grasp-prefix optimization tests.

Offline: recompute frozen baseline metrics from actual trajectory points,
protect STEP12C/12D SHA256, and (when present) verify STEP14 saved trajectory
identity and frozen A/B/C joints.
No MoveIt execute. No real robot.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import unittest

import yaml

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WINNER = os.path.join(_PKG, "config", "step12c_tilted_camera_winner.yaml")
_TRAJ = os.path.join(_PKG, "config", "step12c_tilted_camera_winner_trajectory.yaml")
_STEP14_TRAJ = os.path.join(_PKG, "config", "step14_optimized_grasp_trajectory.yaml")
_STEP14_WINNER = os.path.join(_PKG, "config", "step14_optimized_grasp_winner.yaml")
_STEP14_DIAG = os.path.join(_PKG, "config", "step14_grasp_optimization_diagnostics.yaml")
_ARM = ("j1", "j2", "j3", "j4", "j5", "j6")
_HOME = [
    -2.271411675804,
    -1.642863047968,
    -1.869634009789,
    -2.871167788011,
    -0.003375517130,
    0.839900045153,
]
_A = [0.761386084, -0.642747728, 1.360379768, 0.853167960, 1.570792654, -0.024008243]
_B = [0.128839903, -0.765711154, 2.071148432, 0.157964462, 0.879595649, 2.523736939]
_C = [0.398510667, 0.059882338, 0.469679891, -0.529563111, -1.172285497, 2.443454499]
_FROZEN_WINNER_SHA = "4711d4b374e955d30a1154bf44a54ca520ee47a5f17158db93913fc9a85dc5bd"
_FROZEN_TRAJ_SHA = "2689d683b1f90f617834437defbf5aef237060589f05e4679a09f4c51fe1ea11"
_DEPLOYABLE = (
    "Home_to_PreGrasp",
    "PreGrasp_to_Grasp",
    "Grasp_to_Lift",
    "Lift_to_A",
    "A_to_B",
    "B_to_C_original_bottom",
)


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise AssertionError(f"{path} is not a YAML mapping")
    return data


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def _time_s(pt: dict) -> float:
    tfs = pt.get("time_from_start") or {}
    return float(tfs.get("sec", 0)) + 1e-9 * float(tfs.get("nanosec", 0))


def _arm(names, values):
    lookup = {str(n): float(v) for n, v in zip(names, values)}
    return [lookup[j] for j in _ARM]


def _max_abs(a, b) -> float:
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def _logical_metrics(segments, logical: str) -> dict:
    parts = [s for s in segments if s.get("logical_segment") == logical and s.get("points")]
    if not parts:
        return {"present": False}
    duration = 0.0
    path = 0.0
    j1 = 0.0
    l1 = 0.0
    points = 0
    for seg in parts:
        pts = seg["points"]
        duration += _time_s(pts[-1])
        points += len(pts)
        prev = None
        for pt in pts:
            cur = _arm(seg["joint_names"], pt["positions"])
            if prev is not None:
                path += math.sqrt(sum((a - b) ** 2 for a, b in zip(cur, prev)))
                l1 += sum(abs(a - b) for a, b in zip(cur, prev))
                j1 += abs(cur[0] - prev[0])
            prev = cur
    start = parts[0].get("start_joints") or _arm(parts[0]["joint_names"], parts[0]["points"][0]["positions"])
    end = parts[-1].get("end_joints") or _arm(parts[-1]["joint_names"], parts[-1]["points"][-1]["positions"])
    return {
        "present": True,
        "duration": duration,
        "path": path,
        "j1": j1,
        "l1": l1,
        "points": points,
        "start": [float(v) for v in start[:6]],
        "end": [float(v) for v in end[:6]],
    }


class Step14BaselineAndSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.winner = _load(_WINNER)
        cls.traj = _load(_TRAJ)
        cls.segments = list(cls.traj.get("segments") or [])

    def test_1_frozen_sha256_unchanged(self) -> None:
        self.assertEqual(_sha256(_WINNER), _FROZEN_WINNER_SHA)
        self.assertEqual(_sha256(_TRAJ), _FROZEN_TRAJ_SHA)

    def test_2_abc_definition_unchanged(self) -> None:
        self.assertAlmostEqual(float(self.winner["a_roll_deg"]), -90.0, places=6)
        self.assertAlmostEqual(float(self.winner["b_roll_deg"]), -130.0, places=6)
        self.assertAlmostEqual(float(self.winner["c_roll_deg"]), -85.0, places=6)
        self.assertLessEqual(_max_abs(self.winner["home_joints_rad"], _HOME), 1e-9)
        self.assertLessEqual(_max_abs(self.winner["a_joints_rad"], _A), 1e-9)
        self.assertLessEqual(_max_abs(self.winner["b_joints_rad"], _B), 1e-9)
        self.assertLessEqual(_max_abs(self.winner["c_joints_rad"], _C), 1e-9)

    def test_3_baseline_metrics_from_actual_points(self) -> None:
        home = _logical_metrics(self.segments, "Home_to_PreGrasp")
        grasp = _logical_metrics(self.segments, "PreGrasp_to_Grasp")
        lift = _logical_metrics(self.segments, "Grasp_to_Lift")
        a = _logical_metrics(self.segments, "Lift_to_A")
        b = _logical_metrics(self.segments, "A_to_B")
        c = _logical_metrics(self.segments, "B_to_C_original_bottom")
        self.assertTrue(home["present"])
        self.assertAlmostEqual(home["duration"], 5.902542553, places=9)
        self.assertGreater(home["j1"], 2.8)
        self.assertLess(home["j1"], 2.9)
        prefix = home["duration"] + grasp["duration"] + lift["duration"]
        full = prefix + a["duration"] + b["duration"] + c["duration"]
        self.assertAlmostEqual(full, float(self.traj["persisted_fixed_task_total_duration"]), places=9)
        self.assertAlmostEqual(full / 0.05, full / 0.05)
        self.assertLessEqual(_max_abs(home["start"], _HOME), 1e-3)

    def test_4_step13_default_still_points_at_old_trajectory(self) -> None:
        launch = os.path.join(_PKG, "launch", "fr3_real_frozen_trajectory_executor.launch.py")
        with open(launch, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("step12c_tilted_camera_winner_trajectory.yaml", text)
        self.assertNotIn("step14_optimized_grasp_trajectory.yaml", text)
        src = os.path.join(_PKG, "src", "fr3_real_frozen_trajectory_executor.cpp")
        with open(src, encoding="utf-8") as handle:
            executor = handle.read()
        self.assertNotIn("optimize_grasp_prefix", executor)

    def test_5_default_search_launch_is_opt_in(self) -> None:
        launch = os.path.join(_PKG, "launch", "mtc_fr3_complete_abc_search.launch.py")
        with open(launch, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn('DeclareLaunchArgument("optimize_grasp_prefix", default_value="false")', text)
        self.assertIn('if params["optimize_grasp_prefix"]:', text)
        self.assertIn("step14_optimized_grasp_winner.yaml", text)
        self.assertIn("step14_optimized_grasp_trajectory.yaml", text)
        persist = os.path.join(_PKG, "launch", "mtc_fr3_persist_step12c_winner.launch.py")
        with open(persist, encoding="utf-8") as handle:
            persist_text = handle.read()
        self.assertNotIn("optimize_grasp_prefix", persist_text)

        src = os.path.join(_PKG, "src", "mtc_fr3_complete_abc_search.cpp")
        with open(src, encoding="utf-8") as handle:
            cpp = handle.read()
        self.assertIn("targetsFrozenBaselineFile", cpp)
        self.assertIn("refusing to overwrite frozen winner YAML", cpp)
        self.assertIn("refusing to overwrite frozen trajectory YAML", cpp)

    def test_6_pilz_lin_still_used_in_prefix_stages(self) -> None:
        src = os.path.join(_PKG, "src", "mtc_fr3_complete_abc_search.cpp")
        with open(src, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn('const char* kPilzPlannerId = "LIN"', text)
        self.assertIn("MoveTo Grasp", text)
        self.assertIn("MoveTo Lift", text)
        self.assertIn("setPlannerId(kPilzPlannerId)", text)
        self.assertIn("pregrasp_joints", text)
        dedicated = os.path.join(_PKG, "launch", "mtc_fr3_step14_optimize_grasp_prefix.launch.py")
        with open(dedicated, encoding="utf-8") as handle:
            step14 = handle.read()
        self.assertIn('"optimize_grasp_prefix": "true"', step14)
        self.assertIn("step14_optimized_grasp_trajectory.yaml", step14)
        self.assertNotIn("execute:=true", step14)


@unittest.skipUnless(os.path.isfile(_STEP14_TRAJ), "STEP14 trajectory not generated yet")
class Step14WinnerOutputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.traj = _load(_STEP14_TRAJ)
        cls.old = _load(_TRAJ)
        cls.winner = _load(_WINNER)
        cls.step14_winner = _load(_STEP14_WINNER) if os.path.isfile(_STEP14_WINNER) else {}
        cls.diag = _load(_STEP14_DIAG) if os.path.isfile(_STEP14_DIAG) else {}
        cls.segments = list(cls.traj.get("segments") or [])

    def test_7_does_not_overwrite_frozen_files(self) -> None:
        self.assertEqual(_sha256(_WINNER), _FROZEN_WINNER_SHA)
        self.assertEqual(_sha256(_TRAJ), _FROZEN_TRAJ_SHA)

    def test_8_abc_joints_match_frozen_winner(self) -> None:
        a = _logical_metrics(self.segments, "Lift_to_A")
        b = _logical_metrics(self.segments, "A_to_B")
        c = _logical_metrics(self.segments, "B_to_C_original_bottom")
        self.assertLessEqual(_max_abs(a["end"], _A), 1e-3)
        self.assertLessEqual(_max_abs(b["end"], _B), 1e-3)
        self.assertLessEqual(_max_abs(c["end"], _C), 1e-3)

    def test_9_saved_duration_matches_diagnostics_score(self) -> None:
        home = _logical_metrics(self.segments, "Home_to_PreGrasp")
        grasp = _logical_metrics(self.segments, "PreGrasp_to_Grasp")
        lift = _logical_metrics(self.segments, "Grasp_to_Lift")
        a = _logical_metrics(self.segments, "Lift_to_A")
        b = _logical_metrics(self.segments, "A_to_B")
        c = _logical_metrics(self.segments, "B_to_C_original_bottom")
        full = (
            home["duration"]
            + grasp["duration"]
            + lift["duration"]
            + a["duration"]
            + b["duration"]
            + c["duration"]
        )
        self.assertAlmostEqual(full, float(self.traj["persisted_fixed_task_total_duration"]), places=9)
        if self.diag:
            self.assertAlmostEqual(full, float(self.diag["new_full_home_to_c_time"]), places=9)
            self.assertIn(self.diag.get("scored_trajectory_is_saved_trajectory"), ("YES", True))
            self.assertEqual(self.diag.get("pregrasp_to_grasp_planner_id"), "LIN")
            self.assertEqual(self.diag.get("grasp_to_lift_planner_id"), "LIN")
            self.assertEqual(int(self.diag.get("real_robot_commands", 1)), 0)
            self.assertEqual(int(self.diag.get("gripper_commands", 1)), 0)

    def test_10_round_trip_positions_times(self) -> None:
        again = _load(_STEP14_TRAJ)
        self.assertEqual(self.traj["joint_names"], again["joint_names"])
        self.assertEqual(len(self.traj["segments"]), len(again["segments"]))
        for a, b in zip(self.traj["segments"], again["segments"]):
            self.assertEqual(a["joint_names"], b["joint_names"])
            self.assertEqual(len(a["points"]), len(b["points"]))
            for pa, pb in zip(a["points"], b["points"]):
                self.assertLessEqual(_max_abs(pa["positions"], pb["positions"]), 1e-12)
                self.assertEqual(pa["time_from_start"], pb["time_from_start"])
                if pa.get("velocities"):
                    self.assertLessEqual(_max_abs(pa["velocities"], pb["velocities"]), 1e-12)
                if pa.get("accelerations"):
                    self.assertLessEqual(_max_abs(pa["accelerations"], pb["accelerations"]), 1e-12)

    def test_11_execution_false(self) -> None:
        self.assertFalse(self.traj.get("execution_performed"))
        self.assertEqual(self.traj.get("runtime_current_to_home"), "REPLAN_REQUIRED")


class Step14CppHelperTest(unittest.TestCase):
    def test_12_cpp_helper_executable(self) -> None:
        exe = os.path.expanduser(
            "~/fr_task_ws/install/fr_task_planner/lib/fr_task_planner/test_step14_grasp_prefix"
        )
        if not os.path.isfile(exe):
            self.skipTest("test_step14_grasp_prefix not built yet")
        proc = subprocess.run([exe], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
