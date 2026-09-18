#!/usr/bin/env python3
"""STEP 12 / 12B fixed-P1 geometry regressions. No MoveIt, no Gazebo, no execution."""

from __future__ import annotations

import math
import os
import unittest

import yaml
from geometry_msgs.msg import Pose

_LAUNCH_DIR = os.path.join(os.path.dirname(__file__), "..", "launch")
import sys

sys.path.insert(0, os.path.abspath(_LAUNCH_DIR))

from inspection_view_geometry import (  # noqa: E402
    actual_view_center,
    actual_view_normal,
    attach_bottom_circle_view,
    cylinder_center_in_object,
    generate_roll_candidates,
    interpret_tcp_object_rotation,
    load_stage6_geometry,
    pose_multiply,
    retarget_inspection_p1,
    sample_circular_cap_local,
    tcp_target_from_object,
    validate_stage6_geometry,
    vec_dot,
    vec_norm,
    vec_sub,
    view_name_from_normal,
)
from fr_control.inspection_poses import vec_normalize  # noqa: E402

FIXED_P1 = (0.0, 0.4, 1.20)
_YAML_PATH = os.path.expanduser("~/fairino_ws/src/fr_control/config/stage4_config.yaml")


def _pose_xyz(pose: Pose) -> tuple[float, float, float]:
    return (pose.position.x, pose.position.y, pose.position.z)


class Step12FixedGeometryTest(unittest.TestCase):
    """P1.z=1.20 surface-center / normal / grasp-transform regressions."""

    @classmethod
    def setUpClass(cls) -> None:
        yaml_geo = load_stage6_geometry(
            os.path.expanduser("~/fairino_ws/src/fr_control/config/stage4_config.yaml")
        )
        cls.yaml_p1z = float(yaml_geo["p1_world"][2])
        cls.data = retarget_inspection_p1(yaml_geo, FIXED_P1)
        cls.failures = validate_stage6_geometry(cls.data)

    def test_retarget_does_not_change_yaml_authority_copy(self) -> None:
        self.assertAlmostEqual(self.yaml_p1z, 1.1, places=9)

    def test_p1_is_exactly_fixed(self) -> None:
        p1 = self.data["p1_world"]
        self.assertAlmostEqual(p1[0], 0.0, places=9)
        self.assertAlmostEqual(p1[1], 0.4, places=9)
        self.assertAlmostEqual(p1[2], 1.20, places=9)

    def test_validate_geometry_pass(self) -> None:
        self.assertEqual(self.failures, [])

    def test_surface_centers_equal_p1(self) -> None:
        for name, target in self.data["targets"].items():
            center = actual_view_center(target.object_pose, target.view.center_in_object)
            err = vec_norm(vec_sub(center, FIXED_P1))
            self.assertLessEqual(err, 1e-9, msg=f"{name} center {center} err={err}")
            self.assertLessEqual(target.view_center_error_m, 1e-9, msg=name)

    def test_face_normals_equal_minus_y(self) -> None:
        for name, target in self.data["targets"].items():
            normal = vec_normalize(
                actual_view_normal(target.object_pose, target.view.normal_in_object),
                name,
            )
            self.assertAlmostEqual(normal[0], 0.0, places=9, msg=name)
            self.assertAlmostEqual(normal[1], -1.0, places=9, msg=name)
            self.assertAlmostEqual(normal[2], 0.0, places=9, msg=name)
            self.assertLessEqual(target.normal_angle_error_deg, 1e-9, msg=name)

    def test_object_center_sanity(self) -> None:
        radius = float(self.data["radius"])
        height = float(self.data["height"])
        side = _pose_xyz(self.data["targets"]["side_pos_y"].object_pose)
        side_b = _pose_xyz(self.data["targets"]["side_neg_y"].object_pose)
        top = _pose_xyz(self.data["targets"]["top_circle"].object_pose)
        self.assertAlmostEqual(side[0], 0.0, places=6)
        self.assertAlmostEqual(side[1], 0.4 + radius, places=6)
        self.assertAlmostEqual(side[2], 1.20, places=6)
        self.assertAlmostEqual(side_b[0], 0.0, places=6)
        self.assertAlmostEqual(side_b[1], 0.4 + radius, places=6)
        self.assertAlmostEqual(side_b[2], 1.20, places=6)
        self.assertAlmostEqual(top[0], 0.0, places=6)
        self.assertAlmostEqual(top[1], 0.4 + 0.5 * height, places=6)
        self.assertAlmostEqual(top[2], 1.20, places=6)

    def test_grasp_transform_tcp_origin_is_object_center(self) -> None:
        tcp = self.data["tcp_t_object"]
        self.assertEqual(interpret_tcp_object_rotation(tcp), "Rx(180 deg)")
        self.assertLessEqual(vec_norm((tcp.position.x, tcp.position.y, tcp.position.z)), 1e-9)

    def test_world_tcp_times_tcp_object_equals_object(self) -> None:
        tcp_t_object = self.data["tcp_t_object"]
        for name, target in self.data["targets"].items():
            recomputed_tcp = tcp_target_from_object(target.object_pose, tcp_t_object)
            self.assertAlmostEqual(
                recomputed_tcp.position.x, target.tcp_pose.position.x, places=12, msg=name
            )
            self.assertAlmostEqual(
                recomputed_tcp.position.y, target.tcp_pose.position.y, places=12, msg=name
            )
            self.assertAlmostEqual(
                recomputed_tcp.position.z, target.tcp_pose.position.z, places=12, msg=name
            )
            reconstructed = pose_multiply(target.tcp_pose, tcp_t_object)
            obj = target.object_pose
            self.assertAlmostEqual(reconstructed.position.x, obj.position.x, places=12, msg=name)
            self.assertAlmostEqual(reconstructed.position.y, obj.position.y, places=12, msg=name)
            self.assertAlmostEqual(reconstructed.position.z, obj.position.z, places=12, msg=name)

    def test_rolled_poses_keep_p1_and_normal(self) -> None:
        tcp = self.data["tcp_t_object"]
        for name, target in self.data["targets"].items():
            rolls = generate_roll_candidates(target, tcp, 5.0)
            self.assertEqual(len(rolls), 72, msg=name)
            for cand in rolls:
                self.assertLessEqual(cand.view_center_error_m, 1e-8, msg=f"{name} r={cand.roll_deg}")
                self.assertLessEqual(
                    cand.normal_angle_error_deg, 1e-6, msg=f"{name} r={cand.roll_deg}"
                )
                center = actual_view_center(cand.object_pose, target.view.center_in_object)
                self.assertLessEqual(vec_norm(vec_sub(center, FIXED_P1)), 1e-8)


class Step12BBottomFaceTest(unittest.TestCase):
    """ARM1 C is original table-contact bottom (object -Z). top_circle stays +Z."""

    @classmethod
    def setUpClass(cls) -> None:
        with open(_YAML_PATH, encoding="utf-8") as handle:
            cls.cfg = yaml.safe_load(handle)
        yaml_geo = load_stage6_geometry(_YAML_PATH)
        cls.data = attach_bottom_circle_view(retarget_inspection_p1(yaml_geo, FIXED_P1))
        cls.failures = validate_stage6_geometry(cls.data)
        cls.height = float(cls.data["height"])
        cls.radius = float(cls.data["radius"])
        cls.half = 0.5 * cls.height
        cls.bottom = cls.data["targets"]["bottom_circle"]
        cls.top = cls.data["targets"]["top_circle"]
        cls.rolls_c = generate_roll_candidates(
            cls.bottom, cls.data["tcp_t_object"], 5.0
        )

    def test_validate_with_bottom_attached(self) -> None:
        self.assertEqual(self.failures, [])

    def test_1_initial_bottom_identity(self) -> None:
        obj = self.cfg["object"]["initial_pose"]["position"]
        table_z = float(self.cfg["grasp"]["table_top_z"])
        center_z = float(obj["z"])
        self.assertAlmostEqual(center_z, 0.7675, places=9)
        self.assertAlmostEqual(self.half, 0.0175, places=9)
        self.assertAlmostEqual(table_z, 0.750, places=9)
        bottom_z = center_z - self.half
        self.assertAlmostEqual(bottom_z, table_z, places=9)
        self.assertAlmostEqual(bottom_z, 0.750, places=9)

    def test_2_initial_top_identity(self) -> None:
        center_z = float(self.cfg["object"]["initial_pose"]["position"]["z"])
        top_z = center_z + self.half
        self.assertAlmostEqual(top_z, 0.785, places=9)

    def test_3_arm1_c_local_face_is_minus_z(self) -> None:
        view = self.data["views"]["bottom_circle"]
        self.assertEqual(view.name, "bottom_circle")
        self.assertEqual(view_name_from_normal((0.0, 0.0, -1.0)), "bottom_circle")
        self.assertAlmostEqual(view.normal_in_object[0], 0.0, places=12)
        self.assertAlmostEqual(view.normal_in_object[1], 0.0, places=12)
        self.assertAlmostEqual(view.normal_in_object[2], -1.0, places=12)
        self.assertNotEqual(self.data["arm1_views"][2], "top_circle")
        self.assertEqual(self.data["arm1_views"], ("side_pos_y", "side_neg_y", "bottom_circle"))

    def test_4_c_surface_center_is_p1_for_every_roll(self) -> None:
        self.assertEqual(len(self.rolls_c), 72)
        local_c = self.bottom.view.center_in_object
        self.assertAlmostEqual(local_c[0], 0.0, places=12)
        self.assertAlmostEqual(local_c[1], 0.0, places=12)
        self.assertAlmostEqual(local_c[2], -self.half, places=12)
        self.assertEqual(
            cylinder_center_in_object((0.0, 0.0, -1.0), self.radius, self.height),
            (0.0, 0.0, -self.half),
        )
        for cand in self.rolls_c:
            center = actual_view_center(cand.object_pose, local_c)
            err = vec_norm(vec_sub(center, FIXED_P1))
            self.assertLessEqual(err, 1e-8, msg=f"roll={cand.roll_deg} center={center}")
            self.assertLessEqual(cand.view_center_error_m, 1e-8, msg=cand.roll_deg)

    def test_5_c_transformed_normal_is_world_minus_y(self) -> None:
        local_n = (0.0, 0.0, -1.0)
        for cand in self.rolls_c:
            normal = vec_normalize(
                actual_view_normal(cand.object_pose, local_n),
                f"c roll {cand.roll_deg}",
            )
            self.assertAlmostEqual(normal[0], 0.0, places=8, msg=cand.roll_deg)
            self.assertAlmostEqual(normal[1], -1.0, places=8, msg=cand.roll_deg)
            self.assertAlmostEqual(normal[2], 0.0, places=8, msg=cand.roll_deg)
            self.assertLessEqual(cand.normal_angle_error_deg, 1e-6, msg=cand.roll_deg)
            top_world = vec_normalize(
                actual_view_normal(cand.object_pose, (0.0, 0.0, 1.0)),
                f"top-on-c {cand.roll_deg}",
            )
            self.assertAlmostEqual(top_world[1], 1.0, places=8, msg=cand.roll_deg)
            self.assertGreater(vec_dot(normal, (0.0, -1.0, 0.0)), 0.999)

    def test_6_future_top_face_preserved_as_plus_z(self) -> None:
        top_view = self.data["views"]["top_circle"]
        self.assertEqual(view_name_from_normal((0.0, 0.0, 1.0)), "top_circle")
        self.assertEqual(self.data["arm2_future_view"], "top_circle")
        self.assertAlmostEqual(top_view.normal_in_object[0], 0.0, places=12)
        self.assertAlmostEqual(top_view.normal_in_object[1], 0.0, places=12)
        self.assertAlmostEqual(top_view.normal_in_object[2], 1.0, places=12)
        self.assertAlmostEqual(top_view.center_in_object[2], self.half, places=12)
        self.assertNotEqual(top_view.normal_in_object, self.data["views"]["bottom_circle"].normal_in_object)

    def test_7_transform_closure_tcp_object(self) -> None:
        tcp_t_object = self.data["tcp_t_object"]
        self.assertEqual(interpret_tcp_object_rotation(tcp_t_object), "Rx(180 deg)")
        self.assertLessEqual(
            vec_norm((tcp_t_object.position.x, tcp_t_object.position.y, tcp_t_object.position.z)),
            1e-9,
        )
        for name, target in self.data["targets"].items():
            reconstructed = pose_multiply(target.tcp_pose, tcp_t_object)
            obj = target.object_pose
            self.assertAlmostEqual(reconstructed.position.x, obj.position.x, places=12, msg=name)
            self.assertAlmostEqual(reconstructed.position.y, obj.position.y, places=12, msg=name)
            self.assertAlmostEqual(reconstructed.position.z, obj.position.z, places=12, msg=name)
            recomputed = tcp_target_from_object(target.object_pose, tcp_t_object)
            self.assertAlmostEqual(recomputed.position.x, target.tcp_pose.position.x, places=12, msg=name)

    def test_bottom_roi_local_z_is_minus_half_height(self) -> None:
        pts = sample_circular_cap_local(-self.half, self.radius)
        self.assertEqual(len(pts), 1 + 8 + 8 + 12 + 16)
        for pt in pts:
            self.assertAlmostEqual(pt[2], -self.half, places=12)
            self.assertLessEqual(math.hypot(pt[0], pt[1]), self.radius + 1e-12)
        top_pts = sample_circular_cap_local(self.half, self.radius)
        for pt in top_pts:
            self.assertAlmostEqual(pt[2], self.half, places=12)
            self.assertNotAlmostEqual(pt[2], -self.half, places=6)

    def test_bottom_object_center_sanity_not_hardcoded_as_authority(self) -> None:
        obj = _pose_xyz(self.bottom.object_pose)
        expected_y = FIXED_P1[1] + self.half
        self.assertAlmostEqual(obj[0], 0.0, places=6)
        self.assertAlmostEqual(obj[1], expected_y, places=6)
        self.assertAlmostEqual(obj[2], 1.20, places=6)
        top_obj = _pose_xyz(self.top.object_pose)
        self.assertAlmostEqual(top_obj[1], expected_y, places=6)
        q_b = self.bottom.object_pose.orientation
        q_t = self.top.object_pose.orientation
        same = abs(q_b.x * q_t.x + q_b.y * q_t.y + q_b.z * q_t.z + q_b.w * q_t.w)
        self.assertLess(same, 0.999)

    def test_arm1_abc_names(self) -> None:
        self.assertIn("side_pos_y", self.data["views"])
        self.assertIn("side_neg_y", self.data["views"])
        self.assertIn("bottom_circle", self.data["views"])
        self.assertIn("top_circle", self.data["views"])
        self.assertEqual(self.data["views"]["side_pos_y"].normal_in_object[1], 1.0)
        self.assertEqual(self.data["views"]["side_neg_y"].normal_in_object[1], -1.0)


if __name__ == "__main__":
    unittest.main()
