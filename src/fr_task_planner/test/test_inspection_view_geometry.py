#!/usr/bin/env python3
"""STEP 6 geometry unit tests. No MoveIt, no Gazebo, no view order."""

from __future__ import annotations

import os
import sys
import unittest

_LAUNCH_DIR = os.path.join(os.path.dirname(__file__), "..", "launch")
sys.path.insert(0, os.path.abspath(_LAUNCH_DIR))

from inspection_view_geometry import (  # noqa: E402
    cylinder_center_in_object,
    interpret_tcp_object_rotation,
    load_stage6_geometry,
    object_pose_for_inspection_view,
    tcp_target_from_object,
    validate_stage6_geometry,
    vec_norm,
    vec_sub,
)
from fr_control.inspection_poses import object_pose_for_face  # noqa: E402


class InspectionViewGeometryTest(unittest.TestCase):
    """Assert corrected P1 = view-center semantics."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_stage6_geometry(
            os.path.expanduser(
                "~/fairino_ws/src/fr_control/config/stage4_config.yaml"
            )
        )
        cls.failures = validate_stage6_geometry(cls.data)

    def test_side_pos_y_offset_is_radius(self) -> None:
        self.assertAlmostEqual(
            self.data["targets"]["side_pos_y"].object_center_offset_m,
            self.data["radius"],
            places=9,
        )

    def test_side_neg_y_offset_is_radius(self) -> None:
        self.assertAlmostEqual(
            self.data["targets"]["side_neg_y"].object_center_offset_m,
            self.data["radius"],
            places=9,
        )

    def test_top_offset_is_half_height(self) -> None:
        self.assertAlmostEqual(
            self.data["targets"]["top_circle"].object_center_offset_m,
            0.5 * self.data["height"],
            places=9,
        )

    def test_every_view_center_is_p1(self) -> None:
        for name, target in self.data["targets"].items():
            self.assertLessEqual(
                target.view_center_error_m, 1e-6, msg=name
            )

    def test_every_view_normal_is_d1(self) -> None:
        for name, target in self.data["targets"].items():
            self.assertLessEqual(
                target.normal_angle_error_deg, 1e-6, msg=name
            )

    def test_object_centers_are_not_p1(self) -> None:
        p1 = self.data["p1"]
        centers = {}
        for name, target in self.data["targets"].items():
            center = (
                target.object_pose.position.x,
                target.object_pose.position.y,
                target.object_pose.position.z,
            )
            self.assertGreater(vec_norm(vec_sub(center, p1)), 1e-6, msg=name)
            centers[name] = tuple(round(v, 9) for v in center)
        self.assertNotEqual(centers["top_circle"], centers["side_pos_y"])
        q_pos = self.data["targets"]["side_pos_y"].object_pose.orientation
        q_neg = self.data["targets"]["side_neg_y"].object_pose.orientation
        same_ori = abs(
            q_pos.x * q_neg.x + q_pos.y * q_neg.y + q_pos.z * q_neg.z + q_pos.w * q_neg.w
        )
        self.assertLess(same_ori, 0.999)

    def test_tcp_object_is_rx_pi(self) -> None:
        tcp = self.data["tcp_t_object"]
        self.assertEqual(interpret_tcp_object_rotation(tcp), "Rx(180 deg)")
        recomputed = tcp_target_from_object(
            self.data["targets"]["top_circle"].object_pose, tcp
        )
        actual = self.data["targets"]["top_circle"].tcp_pose
        self.assertAlmostEqual(recomputed.position.x, actual.position.x, places=12)
        self.assertAlmostEqual(recomputed.position.y, actual.position.y, places=12)
        self.assertAlmostEqual(recomputed.position.z, actual.position.z, places=12)

    def test_center_derives_from_normal_not_view_name(self) -> None:
        radius = self.data["radius"]
        height = self.data["height"]
        self.assertEqual(
            cylinder_center_in_object((0.0, 1.0, 0.0), radius, height),
            (0.0, radius, 0.0),
        )
        self.assertEqual(
            cylinder_center_in_object((0.0, -1.0, 0.0), radius, height),
            (0.0, -radius, 0.0),
        )
        self.assertEqual(
            cylinder_center_in_object((0.0, 0.0, 1.0), radius, height),
            (0.0, 0.0, 0.5 * height),
        )
        with self.assertRaises(Exception):
            cylinder_center_in_object((1.0, 1.0, 1.0), radius, height)

    def test_legacy_object_pose_for_face_still_puts_origin_at_p1(self) -> None:
        """Old API is unchanged: object origin == P1. Do not use it for STEP 6."""
        p1 = (0.40, 0.05, 0.28)
        legacy = object_pose_for_face(
            p1,
            face_normal=(0.0, 1.0, 0.0),
            face_up=(0.0, 0.0, 1.0),
            inspection_direction=(0.0, 1.0, 0.0),
            inspection_up=(0.0, 0.0, 1.0),
        )
        self.assertAlmostEqual(legacy.position.x, p1[0])
        self.assertAlmostEqual(legacy.position.y, p1[1])
        self.assertAlmostEqual(legacy.position.z, p1[2])
        corrected = object_pose_for_inspection_view(
            p1,
            center_in_object=(0.0, self.data["radius"], 0.0),
            normal_in_object=(0.0, 1.0, 0.0),
            up_in_object=(0.0, 0.0, 1.0),
            inspection_direction=(0.0, 1.0, 0.0),
            inspection_up=(0.0, 0.0, 1.0),
        )
        self.assertGreater(
            abs(corrected.position.y - legacy.position.y), 1e-6
        )

    def test_validate_all(self) -> None:
        self.assertEqual(self.failures, [])

    def test_orientation_hook_accepts_explicit_quaternion(self) -> None:
        view = self.data["views"]["side_pos_y"]
        target = self.data["targets"]["side_pos_y"]
        q = target.object_pose.orientation
        again = object_pose_for_inspection_view(
            self.data["p1"],
            center_in_object=view.center_in_object,
            normal_in_object=view.normal_in_object,
            up_in_object=view.up_in_object,
            inspection_direction=self.data["direction"],
            inspection_up=self.data["up"],
            orientation_xyzw=(q.x, q.y, q.z, q.w),
        )
        self.assertAlmostEqual(again.position.x, target.object_pose.position.x)
        self.assertAlmostEqual(again.position.y, target.object_pose.position.y)
        self.assertAlmostEqual(again.position.z, target.object_pose.position.z)


if __name__ == "__main__":
    unittest.main()
