#!/usr/bin/env python3
"""STEP 15 sequencing and baseline-protection tests. No robot motion."""

from __future__ import annotations

import hashlib
import os
import subprocess
import unittest

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_STEP12C_W = os.path.join(_PKG, "config", "step12c_tilted_camera_winner.yaml")
_STEP12C_T = os.path.join(_PKG, "config", "step12c_tilted_camera_winner_trajectory.yaml")
_STEP14_W = os.path.join(_PKG, "config", "step14_optimized_grasp_winner.yaml")
_STEP14_T = os.path.join(_PKG, "config", "step14_optimized_grasp_trajectory.yaml")
_STEP14_D = os.path.join(_PKG, "config", "step14_grasp_optimization_diagnostics.yaml")
_BASELINE = {
    _STEP12C_W: "4711d4b374e955d30a1154bf44a54ca520ee47a5f17158db93913fc9a85dc5bd",
    _STEP12C_T: "2689d683b1f90f617834437defbf5aef237060589f05e4679a09f4c51fe1ea11",
    _STEP14_W: "5475647e9581a5268399a61ec3e8fe248a0f8049ce1584b8a112c59143c54497",
    _STEP14_T: "d4e3ba36a9473fef247aaf001d336c077ce314389037d26f8e0ed1d0e2575003",
    _STEP14_D: "40793752e3c50650a0e47636bab9b4c441b6053d2736b4ea4b3a134d4a5fa1ae",
}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


class Step15BaselineTest(unittest.TestCase):
    def test_baseline_sha256_unchanged(self) -> None:
        for path, expected in _BASELINE.items():
            self.assertTrue(os.path.isfile(path), msg=path)
            self.assertEqual(_sha256(path), expected, msg=path)


class Step15LaunchTest(unittest.TestCase):
    def test_step15_launch_does_not_overwrite_baselines(self) -> None:
        path = os.path.join(_PKG, "launch", "mtc_fr3_step15_optimize_lift_to_a.launch.py")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("step15_optimized_lift_to_a_winner.yaml", text)
        self.assertIn("step15_optimized_lift_to_a_trajectory.yaml", text)
        self.assertIn("optimize_lift_to_a", text)
        self.assertIn("max_lift_to_a_j1_travel_rad", text)
        self.assertNotIn("execute:=true", text)

    def test_step13_launch_not_switched_to_step15_traj(self) -> None:
        path = os.path.join(_PKG, "launch", "fr3_real_frozen_trajectory_executor.launch.py")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("step12c_tilted_camera_winner_trajectory.yaml", text)
        self.assertNotIn("step15_optimized_lift_to_a_trajectory.yaml", text)
        self.assertIn("open(position=0)", text)
        self.assertIn("GetGripperMotionDone", text)

    def test_open_position_in_executor_yaml(self) -> None:
        path = os.path.join(_PKG, "config", "fr3_real_executor.yaml")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("gripper_open_position: 0", text)
        self.assertIn("gripper_close_position: 85", text)


class Step15CppBinaryTest(unittest.TestCase):
    def test_cpp_unit_binaries_if_present(self) -> None:
        root = os.path.expanduser("~/fr_task_ws")
        binaries = [
            os.path.join(root, "build/fr_task_planner/test_fr3_frozen_executor"),
            os.path.join(root, "install/fr_task_planner/lib/fr_task_planner/test_fr3_frozen_executor"),
        ]
        binary = next((p for p in binaries if os.path.isfile(p)), "")
        if not binary:
            self.skipTest("C++ executor test binary not built yet")
        proc = subprocess.run(
            [binary],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
