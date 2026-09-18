#!/usr/bin/env python3
"""STEP 10 roll-grid geometry tests. No MoveIt, no Gazebo, no order search."""

from __future__ import annotations

import math
import os
import sys
import unittest

_LAUNCH_DIR = os.path.join(os.path.dirname(__file__), "..", "launch")
sys.path.insert(0, os.path.abspath(_LAUNCH_DIR))

from inspection_view_geometry import (  # noqa: E402
    actual_view_center,
    actual_view_normal,
    actual_view_up,
    canonicalize_roll_deg,
    generate_roll_candidates,
    load_stage6_geometry,
    quat_angle_deg,
    quat_xyzw,
    roll_sample_degrees,
    tcp_target_from_object,
    validate_stage6_geometry,
    vec_norm,
    vec_sub,
)


class InspectionRollCandidateTest(unittest.TestCase):
    """Deterministic D1-roll grid must preserve P1 and D1."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_stage6_geometry(
            os.path.expanduser(
                "~/fairino_ws/src/fr_control/config/stage4_config.yaml"
            )
        )
        cls.step6_failures = validate_stage6_geometry(cls.data)
        cls.rolls = {
            name: generate_roll_candidates(
                target, cls.data["tcp_t_object"], roll_step_deg=30.0
            )
            for name, target in cls.data["targets"].items()
        }

    def test_step6_canonical_api_still_passes(self) -> None:
        self.assertEqual(self.step6_failures, [])

    def test_invalid_step_aborts(self) -> None:
        with self.assertRaises(Exception):
            roll_sample_degrees(0.0)
        with self.assertRaises(Exception):
            roll_sample_degrees(35.0)
        with self.assertRaises(Exception):
            roll_sample_degrees(181.0)

    def test_thirty_degree_grid_has_twelve_unique_angles(self) -> None:
        angles = roll_sample_degrees(30.0)
        self.assertEqual(angles[0], 0.0)
        self.assertEqual(len(angles), 12)
        self.assertEqual(len(set(angles)), 12)
        self.assertIn(180.0, angles)
        self.assertNotIn(-180.0, angles)
        self.assertEqual(
            angles,
            [0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 180.0],
        )

    def test_five_degree_grid_has_seventy_two_unique_angles(self) -> None:
        angles = roll_sample_degrees(5.0)
        self.assertEqual(angles[0], 0.0)
        self.assertEqual(len(angles), 72)
        self.assertEqual(len(set(angles)), 72)
        self.assertEqual(angles[1], 5.0)
        self.assertEqual(angles[2], -5.0)
        self.assertIn(180.0, angles)
        self.assertNotIn(-180.0, angles)
        five_deg = generate_roll_candidates(
            self.data["targets"]["top_circle"], self.data["tcp_t_object"], 5.0
        )
        self.assertEqual(len(five_deg), 72)

    def test_canonicalize_plus_minus_180(self) -> None:
        self.assertEqual(canonicalize_roll_deg(180.0), 180.0)
        self.assertEqual(canonicalize_roll_deg(-180.0), 180.0)
        self.assertEqual(canonicalize_roll_deg(330.0), -30.0)
        self.assertEqual(canonicalize_roll_deg(270.0), -90.0)

    def test_each_view_has_twelve_poses(self) -> None:
        for name, rolls in self.rolls.items():
            self.assertEqual(len(rolls), 12, msg=name)

    def test_roll_zero_matches_canonical(self) -> None:
        for name, target in self.data["targets"].items():
            zero = self.rolls[name][0]
            self.assertEqual(zero.roll_deg, 0.0, msg=name)
            self.assertLess(abs(zero.object_pose.position.x - target.object_pose.position.x), 1e-9)
            self.assertLess(abs(zero.object_pose.position.y - target.object_pose.position.y), 1e-9)
            self.assertLess(abs(zero.object_pose.position.z - target.object_pose.position.z), 1e-9)
            ori = quat_angle_deg(zero.object_pose.orientation, target.object_pose.orientation)
            self.assertLess(ori, 1e-6, msg=name)
            tcp_ori = quat_angle_deg(zero.tcp_pose.orientation, target.tcp_pose.orientation)
            self.assertLess(tcp_ori, 1e-6, msg=name)
            self.assertLess(zero.up_error_deg, 1e-6, msg=name)

    def test_source_frames_are_world(self) -> None:
        self.assertEqual(self.data["p1_source_frame"], "world")
        self.assertEqual(self.data["d1_source_frame"], "world")
        self.assertEqual(self.data["up_source_frame"], "world")

    def test_every_roll_keeps_p1_and_d1(self) -> None:
        p1 = self.data["p1_world"]
        d1 = self.data["d1_world"]
        for name, rolls in self.rolls.items():
            for cand in rolls:
                center = actual_view_center(cand.object_pose, cand.view.center_in_object)
                normal = actual_view_normal(cand.object_pose, cand.view.normal_in_object)
                self.assertLessEqual(
                    vec_norm(vec_sub(center, p1)), 1e-6, msg=f"{name} {cand.roll_deg}"
                )
                self.assertLessEqual(
                    cand.view_center_error_m, 1e-6, msg=f"{name} {cand.roll_deg}"
                )
                self.assertLessEqual(
                    cand.normal_angle_error_deg, 1e-6, msg=f"{name} {cand.roll_deg}"
                )
                self.assertAlmostEqual(normal[0], d1[0], places=9)
                self.assertAlmostEqual(normal[1], d1[1], places=9)
                self.assertAlmostEqual(normal[2], d1[2], places=9)

    def test_roll_zero_up_is_world_plus_z(self) -> None:
        for name, rolls in self.rolls.items():
            zero = rolls[0]
            up = actual_view_up(zero.object_pose, zero.view.up_in_object)
            self.assertAlmostEqual(up[0], 0.0, places=9, msg=name)
            self.assertAlmostEqual(up[1], 0.0, places=9, msg=name)
            self.assertAlmostEqual(up[2], 1.0, places=9, msg=name)

    def test_nonzero_roll_can_have_up_error(self) -> None:
        nonzero = [c for c in self.rolls["side_pos_y"] if abs(c.roll_deg) > 1e-9]
        self.assertTrue(any(c.up_error_deg > 1.0 for c in nonzero))

    def test_tcp_object_preserved(self) -> None:
        tcp = self.data["tcp_t_object"]
        for name, rolls in self.rolls.items():
            for cand in rolls:
                recomputed = tcp_target_from_object(cand.object_pose, tcp)
                self.assertAlmostEqual(recomputed.position.x, cand.tcp_pose.position.x, places=9)
                self.assertAlmostEqual(recomputed.position.y, cand.tcp_pose.position.y, places=9)
                self.assertAlmostEqual(recomputed.position.z, cand.tcp_pose.position.z, places=9)
                self.assertLess(
                    quat_angle_deg(recomputed.orientation, cand.tcp_pose.orientation),
                    1e-6,
                    msg=f"{name} {cand.roll_deg}",
                )

    def test_no_quaternion_sign_duplicates(self) -> None:
        for name, rolls in self.rolls.items():
            keys = []
            for cand in rolls:
                q = quat_xyzw(cand.object_pose.orientation)
                if q[3] < 0.0:
                    q = tuple(-v for v in q)
                key = (
                    round(cand.object_pose.position.x, 8),
                    round(cand.object_pose.position.y, 8),
                    round(cand.object_pose.position.z, 8),
                    tuple(round(v, 8) for v in q),
                )
                keys.append(key)
            self.assertEqual(len(keys), len(set(keys)), msg=name)


if __name__ == "__main__":
    unittest.main()
