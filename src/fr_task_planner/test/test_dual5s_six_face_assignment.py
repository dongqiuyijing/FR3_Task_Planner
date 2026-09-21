#!/usr/bin/env python3
"""DUAL-5-S six-face assignment and +X/-X geometry tests. No MoveIt."""

from __future__ import annotations

import os
import sys
import unittest

_LAUNCH_DIR = os.path.join(os.path.dirname(__file__), "..", "launch")
sys.path.insert(0, os.path.abspath(_LAUNCH_DIR))

from dual5s_side_x_geometry import (  # noqa: E402
    assert_side_x_not_y,
    generate_side_x_rolls,
    load_step12c_geometry_with_side_x,
    make_side_neg_x_view,
    make_side_pos_x_view,
    physical_face_id_from_normal,
    validateSixFaceAssignment,
)
from inspection_view_geometry import (  # noqa: E402
    actual_view_center,
    actual_view_normal,
    angle_between_deg,
    vec_norm,
    vec_sub,
)
from fr_control.inspection_poses import quat_to_matrix, quat_xyzw  # noqa: E402


class Dual5sSixFaceAssignmentTest(unittest.TestCase):
    def test_previous_y_y_z_assignment_fails(self) -> None:
        result = validateSixFaceAssignment(
            [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)],
            [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)],
            arm_a_names=("side_pos_y", "side_neg_y", "bottom_circle"),
            arm_b_names=("side_pos_y", "side_neg_y", "top_circle"),
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "DUPLICATED PHYSICAL FACE")
        self.assertEqual(result["duplicates"], ["+Y", "-Y"])
        self.assertEqual(result["missing"], ["+X", "-X"])

    def test_corrected_x_x_z_assignment_passes(self) -> None:
        result = validateSixFaceAssignment(
            [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)],
            [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
            arm_a_names=("side_pos_y", "side_neg_y", "bottom_circle"),
            arm_b_names=("side_pos_x", "side_neg_x", "top_circle"),
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["arm_a"], ["+Y", "-Y", "-Z"])
        self.assertEqual(result["arm_b"], ["+X", "-X", "+Z"])
        self.assertEqual(result["duplicates"], [])
        self.assertEqual(result["missing"], [])

    def test_rename_only_strings_with_old_normals_fail(self) -> None:
        result = validateSixFaceAssignment(
            [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)],
            [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)],
            arm_a_names=("side_pos_y", "side_neg_y", "bottom_circle"),
            arm_b_names=("side_pos_x", "side_neg_x", "top_circle"),
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "WRONG PHYSICAL NORMAL")
        self.assertEqual(result["arm_b"], ["+Y", "-Y", "+Z"])


class Dual5sSideXGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_step12c_geometry_with_side_x()
        cls.x = cls.data["side_x_targets"]

    def test_p1_d1_unchanged(self) -> None:
        self.assertAlmostEqual(self.data["p1_world"][0], 0.0, places=9)
        self.assertAlmostEqual(self.data["p1_world"][1], 0.3, places=9)
        self.assertAlmostEqual(self.data["p1_world"][2], 1.2, places=9)
        self.assertAlmostEqual(self.data["d1_world"][0], 0.0, places=6)
        self.assertAlmostEqual(self.data["d1_world"][1], -0.707107, places=5)
        self.assertAlmostEqual(self.data["d1_world"][2], 0.707107, places=5)

    def test_side_views_use_existing_radius_center(self) -> None:
        pos = make_side_pos_x_view(self.data["radius"])
        neg = make_side_neg_x_view(self.data["radius"])
        self.assertEqual(physical_face_id_from_normal(pos.normal_in_object), "+X")
        self.assertEqual(physical_face_id_from_normal(neg.normal_in_object), "-X")
        self.assertAlmostEqual(pos.center_in_object[0], self.data["radius"], places=12)
        self.assertAlmostEqual(neg.center_in_object[0], -self.data["radius"], places=12)
        self.assertEqual(pos.up_in_object, (0.0, 0.0, 1.0))
        self.assertEqual(neg.up_in_object, (0.0, 0.0, 1.0))

    def test_x_targets_align_p1_d1_directed(self) -> None:
        for name, expected in (("side_pos_x", "+X"), ("side_neg_x", "-X")):
            target = self.x[name]
            self.assertEqual(
                physical_face_id_from_normal(target.view.normal_in_object), expected
            )
            self.assertLessEqual(target.view_center_error_m, 1e-9, msg=name)
            self.assertLessEqual(target.normal_angle_error_deg, 1e-9, msg=name)
            world_n = actual_view_normal(
                target.object_pose, target.view.normal_in_object
            )
            self.assertLessEqual(
                angle_between_deg(world_n, self.data["d1_world"]), 1e-9, msg=name
            )
            flipped = (-world_n[0], -world_n[1], -world_n[2])
            self.assertGreater(
                angle_between_deg(flipped, self.data["d1_world"]), 170.0, msg=name
            )

    def test_x_orientation_is_not_y(self) -> None:
        assert_side_x_not_y(
            self.x,
            {
                "side_pos_y": self.data["targets"]["side_pos_y"],
                "side_neg_y": self.data["targets"]["side_neg_y"],
            },
        )
        r = quat_to_matrix(quat_xyzw(self.x["side_pos_x"].object_pose.orientation))
        mapped_x = (
            r[0][0] * 1.0 + r[0][1] * 0.0 + r[0][2] * 0.0,
            r[1][0] * 1.0 + r[1][1] * 0.0 + r[1][2] * 0.0,
            r[2][0] * 1.0 + r[2][1] * 0.0 + r[2][2] * 0.0,
        )
        mapped_y = (
            r[0][0] * 0.0 + r[0][1] * 1.0 + r[0][2] * 0.0,
            r[1][0] * 0.0 + r[1][1] * 1.0 + r[1][2] * 0.0,
            r[2][0] * 0.0 + r[2][1] * 1.0 + r[2][2] * 0.0,
        )
        self.assertLessEqual(angle_between_deg(mapped_x, self.data["d1_world"]), 1e-9)
        self.assertGreater(angle_between_deg(mapped_y, self.data["d1_world"]), 80.0)

    def test_radial_object_centers_may_coincide(self) -> None:
        pos_x = (
            self.x["side_pos_x"].object_pose.position.x,
            self.x["side_pos_x"].object_pose.position.y,
            self.x["side_pos_x"].object_pose.position.z,
        )
        pos_y = (
            self.data["targets"]["side_pos_y"].object_pose.position.x,
            self.data["targets"]["side_pos_y"].object_pose.position.y,
            self.data["targets"]["side_pos_y"].object_pose.position.z,
        )
        self.assertLessEqual(vec_norm(vec_sub(pos_x, pos_y)), 1e-9)
        q_x = quat_xyzw(self.x["side_pos_x"].object_pose.orientation)
        q_y = quat_xyzw(self.data["targets"]["side_pos_y"].object_pose.orientation)
        dot = abs(
            q_x[0] * q_y[0] + q_x[1] * q_y[1] + q_x[2] * q_y[2] + q_x[3] * q_y[3]
        )
        self.assertLess(dot, 0.999)

    def test_d1_roll_keeps_physical_face_and_p1(self) -> None:
        rolls = generate_side_x_rolls(self.data, 30.0)
        self.assertGreaterEqual(len(rolls["side_pos_x"]), 12)
        self.assertAlmostEqual(rolls["side_pos_x"][0].roll_deg, 0.0, places=9)
        for name, expected in (("side_pos_x", "+X"), ("side_neg_x", "-X")):
            for cand in rolls[name]:
                self.assertEqual(
                    physical_face_id_from_normal(cand.view.normal_in_object), expected
                )
                self.assertLessEqual(cand.view_center_error_m, 1e-6)
                self.assertLessEqual(cand.normal_angle_error_deg, 1e-6)

    def test_top_circle_identity_unchanged(self) -> None:
        top = self.data["targets"]["top_circle"]
        self.assertEqual(physical_face_id_from_normal(top.view.normal_in_object), "+Z")
        self.assertAlmostEqual(top.view.center_in_object[2], 0.0175, places=12)
        self.assertLessEqual(
            vec_norm(
                vec_sub(
                    actual_view_center(top.object_pose, top.view.center_in_object),
                    self.data["p1_world"],
                )
            ),
            1e-9,
        )


if __name__ == "__main__":
    unittest.main()
