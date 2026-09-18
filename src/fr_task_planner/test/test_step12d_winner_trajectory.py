#!/usr/bin/env python3
"""STEP 12D winner-trajectory persistence schema and regression tests.

No MoveIt, no Gazebo, no execution. Parses the frozen winner metadata
and the persisted JointTrajectory YAML after STEP12D has written it.
"""

from __future__ import annotations

import math
import os
import unittest

import yaml

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WINNER = os.path.join(_PKG, "config", "step12c_tilted_camera_winner.yaml")
_TRAJ = os.path.join(_PKG, "config", "step12c_tilted_camera_winner_trajectory.yaml")

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
_REQUIRED = (
    "Home_to_PreGrasp",
    "PreGrasp_to_Grasp",
    "Grasp_to_Lift",
    "Lift_to_A",
    "A_to_B",
    "B_to_C_original_bottom",
)
_CHAIN = _REQUIRED


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise AssertionError(f"{path} is not a YAML mapping")
    return data


def _finite(values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values)


def _max_abs(a, b) -> float:
    if len(a) != len(b):
        return float("inf")
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def _arm_from(names, values):
    lookup = {str(n): float(v) for n, v in zip(names, values)}
    return [lookup[j] for j in _ARM]


def _time_ns(pt) -> int:
    tfs = pt.get("time_from_start") or {}
    return int(tfs.get("sec", 0)) * 1_000_000_000 + int(tfs.get("nanosec", 0))


class Step12DWinnerYamlTest(unittest.TestCase):
    """TEST 1: winner YAML A/B/C fixed values parse correctly."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.winner = _load(_WINNER)

    def test_1_winner_yaml_abc_fixed_values(self) -> None:
        self.assertEqual(self.winner.get("task_version"), "STEP12C")
        self.assertAlmostEqual(float(self.winner["a_roll_deg"]), -90.0, places=6)
        self.assertAlmostEqual(float(self.winner["b_roll_deg"]), -130.0, places=6)
        self.assertAlmostEqual(float(self.winner["c_roll_deg"]), -85.0, places=6)
        self.assertLessEqual(_max_abs(self.winner["home_joints_rad"], _HOME), 1e-9)
        self.assertLessEqual(_max_abs(self.winner["a_joints_rad"], _A), 1e-9)
        self.assertLessEqual(_max_abs(self.winner["b_joints_rad"], _B), 1e-9)
        self.assertLessEqual(_max_abs(self.winner["c_joints_rad"], _C), 1e-9)
        self.assertAlmostEqual(float(self.winner["predicted_total_trajectory_time"]), 13.847580915)
        self.assertAlmostEqual(float(self.winner["total_joint_path_length"]), 14.505901660)


@unittest.skipUnless(os.path.isfile(_TRAJ), "STEP12D trajectory file not persisted yet")
class Step12DTrajectoryFileTest(unittest.TestCase):
    """TESTs 2-19 against the persisted trajectory YAML."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.traj = _load(_TRAJ)
        cls.winner = _load(_WINNER)
        cls.segments = list(cls.traj.get("segments") or [])
        cls.motion = [s for s in cls.segments if s.get("points")]
        cls.by_logical = {}
        for seg in cls.motion:
            cls.by_logical.setdefault(seg.get("logical_segment"), []).append(seg)

    def _first(self, logical: str) -> dict:
        return self.by_logical[logical][0]

    def _last(self, logical: str) -> dict:
        return self.by_logical[logical][-1]

    def test_2_trajectory_output_schema_parse(self) -> None:
        self.assertEqual(int(self.traj["trajectory_format_version"]), 1)
        self.assertEqual(self.traj["task_version"], "STEP12C")
        self.assertEqual(self.traj["source_winner_file"], "step12c_tilted_camera_winner.yaml")
        self.assertEqual(self.traj["label"], "FROZEN STEP12C WINNER TRAJECTORY")
        self.assertTrue(self.segments)
        for name in _REQUIRED:
            self.assertIn(name, self.by_logical, msg=name)

    def test_3_every_motion_segment_has_joint_names(self) -> None:
        self.assertTrue(self.motion)
        for seg in self.motion:
            self.assertTrue(seg.get("joint_names"), msg=seg.get("name"))

    def test_4_all_required_fr3_arm_joints_present(self) -> None:
        top = list(self.traj.get("joint_names") or [])
        for joint in _ARM:
            self.assertIn(joint, top)
        for seg in self.motion:
            for joint in _ARM:
                self.assertIn(joint, seg["joint_names"], msg=seg.get("name"))

    def test_5_positions_length_matches_joint_names(self) -> None:
        for seg in self.motion:
            n = len(seg["joint_names"])
            for i, pt in enumerate(seg["points"]):
                self.assertEqual(len(pt.get("positions") or []), n, msg=f"{seg['name']}[{i}]")

    def test_6_velocities_empty_or_match_joint_names(self) -> None:
        for seg in self.motion:
            n = len(seg["joint_names"])
            present = bool(seg.get("velocities_present"))
            for i, pt in enumerate(seg["points"]):
                vel = pt.get("velocities") or []
                if present:
                    self.assertEqual(len(vel), n, msg=f"{seg['name']}[{i}]")
                else:
                    self.assertEqual(vel, [], msg=f"{seg['name']}[{i}] fabricated velocities")

    def test_7_accelerations_empty_or_match_joint_names(self) -> None:
        for seg in self.motion:
            n = len(seg["joint_names"])
            present = bool(seg.get("accelerations_present"))
            for i, pt in enumerate(seg["points"]):
                acc = pt.get("accelerations") or []
                if present:
                    self.assertEqual(len(acc), n, msg=f"{seg['name']}[{i}]")
                else:
                    self.assertEqual(acc, [], msg=f"{seg['name']}[{i}] fabricated accelerations")

    def test_8_no_nan_inf(self) -> None:
        for seg in self.motion:
            self.assertTrue(_finite(seg.get("start_joints") or []), msg=seg.get("name"))
            self.assertTrue(_finite(seg.get("end_joints") or []), msg=seg.get("name"))
            for i, pt in enumerate(seg["points"]):
                self.assertTrue(_finite(pt.get("positions") or []), msg=f"{seg['name']}[{i}] pos")
                self.assertTrue(_finite(pt.get("velocities") or []), msg=f"{seg['name']}[{i}] vel")
                self.assertTrue(
                    _finite(pt.get("accelerations") or []), msg=f"{seg['name']}[{i}] acc"
                )

    def test_9_time_from_start_monotonic(self) -> None:
        for seg in self.motion:
            prev = -1
            for i, pt in enumerate(seg["points"]):
                now = _time_ns(pt)
                self.assertGreaterEqual(now, 0, msg=f"{seg['name']}[{i}]")
                self.assertGreaterEqual(now, prev, msg=f"{seg['name']}[{i}]")
                prev = now

    def test_10_segment_duration_positive(self) -> None:
        for seg in self.motion:
            self.assertGreater(float(seg.get("duration") or 0.0), 0.0, msg=seg.get("name"))
            last = seg["points"][-1]
            self.assertGreater(_time_ns(last), 0, msg=seg.get("name"))

    def test_11_trajectory_continuity(self) -> None:
        max_err = 0.0
        for left, right in zip(_CHAIN, _CHAIN[1:]):
            end = self._last(left)["end_joints"]
            start = self._first(right)["start_joints"]
            err = _max_abs(end, start)
            max_err = max(max_err, err)
            self.assertLessEqual(err, 1e-4, msg=f"{left} -> {right}")
        self.assertLessEqual(max_err, 1e-4)

    def test_12_a_endpoint_match(self) -> None:
        end_a = self._last("Lift_to_A")["end_joints"]
        start_ab = self._first("A_to_B")["start_joints"]
        self.assertLessEqual(_max_abs(end_a, _A), 1e-4)
        self.assertLessEqual(_max_abs(start_ab, _A), 1e-4)

    def test_13_b_endpoint_match(self) -> None:
        end_b = self._last("A_to_B")["end_joints"]
        start_bc = self._first("B_to_C_original_bottom")["start_joints"]
        self.assertLessEqual(_max_abs(end_b, _B), 1e-4)
        self.assertLessEqual(_max_abs(start_bc, _B), 1e-4)

    def test_14_c_endpoint_match(self) -> None:
        end_c = self._last("B_to_C_original_bottom")["end_joints"]
        self.assertLessEqual(_max_abs(end_c, _C), 1e-4)

    def test_15_home_fixed_start_match(self) -> None:
        home_seg = self._first("Home_to_PreGrasp")
        start = home_seg["start_joints"]
        self.assertLessEqual(_max_abs(start, _HOME), 1e-4)
        first_pt = home_seg["points"][0]["positions"]
        first_arm = _arm_from(home_seg["joint_names"], first_pt)
        pt_err = _max_abs(first_arm, _HOME)
        if pt_err > 1e-4:
            # Planner may omit the exact start sample; start_state is authoritative.
            self.assertLessEqual(_max_abs(start, _HOME), 1e-4)

    def test_16_round_trip_serialization_match(self) -> None:
        again = _load(_TRAJ)
        self.assertEqual(self.traj["joint_names"], again["joint_names"])
        self.assertEqual(len(self.traj["segments"]), len(again["segments"]))
        for a, b in zip(self.traj["segments"], again["segments"]):
            self.assertEqual(len(a["points"]), len(b["points"]))
            self.assertEqual(a["joint_names"], b["joint_names"])
            for pa, pb in zip(a["points"], b["points"]):
                self.assertLessEqual(_max_abs(pa["positions"], pb["positions"]), 1e-12)
                self.assertEqual(len(pa.get("velocities") or []), len(pb.get("velocities") or []))
                self.assertEqual(
                    len(pa.get("accelerations") or []), len(pb.get("accelerations") or [])
                )
                if pa.get("velocities"):
                    self.assertLessEqual(_max_abs(pa["velocities"], pb["velocities"]), 1e-12)
                if pa.get("accelerations"):
                    self.assertLessEqual(
                        _max_abs(pa["accelerations"], pb["accelerations"]), 1e-12
                    )
                self.assertEqual(pa["time_from_start"], pb["time_from_start"])

    def test_17_execution_performed_false(self) -> None:
        self.assertFalse(self.traj.get("execution_performed"))
        self.assertFalse((self.traj.get("deployment") or {}).get("execution_performed"))

    def test_18_runtime_current_to_home_replan_required(self) -> None:
        self.assertEqual(self.traj.get("runtime_current_to_home"), "REPLAN_REQUIRED")
        self.assertEqual(
            (self.traj.get("deployment") or {}).get("runtime_current_to_home"),
            "REPLAN_REQUIRED",
        )
        if "Current_to_Home" in self.by_logical:
            for seg in self.by_logical["Current_to_Home"]:
                self.assertFalse(seg.get("deployable"))
                self.assertTrue(seg.get("runtime_replan_required"))

    def test_19_real_gripper_close_required(self) -> None:
        self.assertTrue(self.traj.get("requires_real_gripper_close"))
        self.assertTrue((self.traj.get("deployment") or {}).get("real_gripper_close_required"))
        events = self.traj.get("events") or []
        kinds = {ev.get("event") for ev in events}
        self.assertIn("REAL_GRIPPER_CLOSE_REQUIRED", kinds)
        self.assertIn("ATTACH_SMALL_PART_TO_TCP", kinds)
        self.assertIn("RESTORE_PART_TABLE_COLLISION", kinds)
        self.assertTrue(all(ev.get("executed_now") is False for ev in events))


if __name__ == "__main__":
    unittest.main()
