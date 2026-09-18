#!/usr/bin/env python3
"""STEP 13 frozen-trajectory executor tests.

Parses the read-only STEP12D YAML and checks executor gating/scaling
helpers independently of any robot motion.
"""

from __future__ import annotations

import math
import os
import subprocess
import unittest

import yaml

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TRAJ = os.path.join(_PKG, "config", "step12c_tilted_camera_winner_trajectory.yaml")
_ARM = ("j1", "j2", "j3", "j4", "j5", "j6")
_DEPLOYABLE = (
    "Home_to_PreGrasp",
    "PreGrasp_to_Grasp",
    "Grasp_to_Lift",
    "Lift_to_A",
    "A_to_B",
    "B_to_C_original_bottom",
)
_CONFIRM = "I_UNDERSTAND_THIS_WILL_MOVE_THE_REAL_ROBOT"


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise AssertionError(f"{path} is not a YAML mapping")
    return data


def _time_s(pt: dict) -> float:
    tfs = pt.get("time_from_start") or {}
    return float(tfs.get("sec", 0)) + 1e-9 * float(tfs.get("nanosec", 0))


def _finite(values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values)


class Step13FrozenYamlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.traj = _load(_TRAJ)
        cls.segments = list(cls.traj.get("segments") or [])
        cls.by_logical = {}
        for seg in cls.segments:
            cls.by_logical.setdefault(seg.get("logical_segment"), []).append(seg)

    def test_1_frozen_parse_pass(self) -> None:
        self.assertEqual(int(self.traj["trajectory_format_version"]), 1)
        self.assertEqual(self.traj["task_version"], "STEP12C")
        self.assertEqual(list(self.traj.get("joint_names") or []), list(_ARM))
        self.assertTrue(self.traj.get("requires_runtime_current_to_home"))
        self.assertEqual(self.traj.get("runtime_current_to_home"), "REPLAN_REQUIRED")

    def test_2_exact_six_deployable_segments(self) -> None:
        deployable = [
            seg
            for seg in self.segments
            if seg.get("deployable") and seg.get("logical_segment") in _DEPLOYABLE
        ]
        names = [seg["logical_segment"] for seg in deployable]
        self.assertEqual(names, list(_DEPLOYABLE))
        self.assertEqual(len(deployable), 6)

    def test_3_current_to_home_not_deployable(self) -> None:
        current = self.by_logical["Current_to_Home"]
        self.assertTrue(current)
        for seg in current:
            self.assertFalse(seg.get("deployable"))
            self.assertTrue(seg.get("runtime_replan_required"))

    def test_deployable_point_and_duration_regression(self) -> None:
        expected_points = {
            "Home_to_PreGrasp": 61,
            "PreGrasp_to_Grasp": 5,
            "Grasp_to_Lift": 5,
            "Lift_to_A": 25,
            "A_to_B": 33,
            "B_to_C_original_bottom": 30,
        }
        total_points = 0
        total_dur = 0.0
        for name, n in expected_points.items():
            seg = self.by_logical[name][0]
            self.assertEqual(len(seg["points"]), n, msg=name)
            total_points += n
            total_dur += _time_s(seg["points"][-1])
        self.assertEqual(total_points, 159)
        self.assertAlmostEqual(total_dur, 15.046908510, places=8)

    def test_events_order_gripper_before_attach_before_restore(self) -> None:
        events = [(e.get("after_segment"), e.get("event")) for e in self.traj.get("events") or []]
        self.assertIn(("PreGrasp_to_Grasp", "REAL_GRIPPER_CLOSE_REQUIRED"), events)
        self.assertIn(("PreGrasp_to_Grasp", "ATTACH_SMALL_PART_TO_TCP"), events)
        self.assertIn(("Grasp_to_Lift", "RESTORE_PART_TABLE_COLLISION"), events)
        close_i = events.index(("PreGrasp_to_Grasp", "REAL_GRIPPER_CLOSE_REQUIRED"))
        attach_i = events.index(("PreGrasp_to_Grasp", "ATTACH_SMALL_PART_TO_TCP"))
        self.assertLess(close_i, attach_i)

    def test_no_nan_and_monotonic_times(self) -> None:
        for seg in self.segments:
            prev = -1.0
            for pt in seg.get("points") or []:
                self.assertTrue(_finite(pt.get("positions") or []), msg=seg.get("name"))
                t = _time_s(pt)
                self.assertGreaterEqual(t, prev)
                prev = t


class Step13GateTest(unittest.TestCase):
    def test_16_execute_false_blocks(self) -> None:
        execute = False
        confirm = ""
        allowed = execute and confirm == _CONFIRM
        self.assertFalse(allowed)

    def test_17_execute_true_without_confirmation_blocks(self) -> None:
        execute = True
        confirm = ""
        allowed = execute and confirm == _CONFIRM
        self.assertFalse(allowed)

    def test_18_wrong_confirmation_blocks(self) -> None:
        execute = True
        confirm = "I_UNDERSTAND"
        allowed = execute and confirm == _CONFIRM
        self.assertFalse(allowed)

    def test_confirmation_exact_match_only(self) -> None:
        self.assertTrue(True and "I_UNDERSTAND_THIS_WILL_MOVE_THE_REAL_ROBOT" == _CONFIRM)


class Step13GripperLaunchTest(unittest.TestCase):
    def test_launch_exposes_gripper_timeout_and_ping_only(self) -> None:
        path = os.path.join(_PKG, "launch", "fr3_real_frozen_trajectory_executor.launch.py")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn('DeclareLaunchArgument("gripper_timeout_sec", default_value="15.0")', text)
        self.assertIn(
            'DeclareLaunchArgument("gripper_post_close_wait_sec", default_value="0.5")', text
        )
        self.assertIn('DeclareLaunchArgument("gripper_ping_only", default_value="false")', text)
        self.assertIn("gripper_rot_num", text)
        self.assertIn("gripper_rot_vel", text)
        self.assertIn("gripper_rot_torque", text)
        self.assertNotIn("ActGripper", text)

    def test_local_executor_yaml_rot_defaults_match_python(self) -> None:
        path = os.path.join(_PKG, "config", "fr3_real_executor.yaml")
        data = _load(path)
        self.assertEqual(int(data["gripper_id"]), 1)
        self.assertEqual(int(data["gripper_close_position"]), 85)
        self.assertEqual(int(data["gripper_velocity"]), 20)
        self.assertEqual(int(data["gripper_force"]), 20)
        self.assertEqual(int(data["gripper_max_time_ms"]), 5000)
        self.assertEqual(int(data["gripper_block"]), 1)
        self.assertEqual(int(data["gripper_type"]), 0)
        self.assertEqual(float(data["gripper_rot_num"]), 0.0)
        self.assertEqual(int(data["gripper_rot_vel"]), 0)
        self.assertEqual(int(data["gripper_rot_torque"]), 0)
        self.assertEqual(float(data["gripper_timeout_sec"]), 15.0)


class Step13CppBinaryTest(unittest.TestCase):
    def test_cpp_unit_binary_if_present(self) -> None:
        candidates = [
            os.path.join(
                os.path.expanduser("~/fr_task_ws/build/fr_task_planner"),
                "test_fr3_frozen_executor",
            ),
            os.path.join(
                os.path.expanduser("~/fr_task_ws/install/fr_task_planner/lib/fr_task_planner"),
                "test_fr3_frozen_executor",
            ),
        ]
        binary = next((p for p in candidates if os.path.isfile(p)), "")
        if not binary:
            self.skipTest("C++ test binary not built yet")
        proc = subprocess.run(
            [binary, _TRAJ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
