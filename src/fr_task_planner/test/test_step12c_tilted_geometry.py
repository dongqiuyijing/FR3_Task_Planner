#!/usr/bin/env python3
"""STEP 12C tilted-camera / new-home / new-P1 geometry regressions.

No MoveIt, no Gazebo, no execution. Does not rewrite stage4_config.yaml
inspection fields; Home is read from the authoritative YAML source.
"""

from __future__ import annotations

import math
import os
import unittest

from geometry_msgs.msg import Pose

_LAUNCH_DIR = os.path.join(os.path.dirname(__file__), "..", "launch")
import sys

sys.path.insert(0, os.path.abspath(_LAUNCH_DIR))

from inspection_view_geometry import (  # noqa: E402
    STEP12C_CAMERA_POSITION,
    STEP12C_CAMERA_RPY,
    STEP12C_HOME_DEG,
    STEP12C_P1,
    STEP12C_SURFACE_RPY,
    actual_view_center,
    actual_view_normal,
    apply_step12c_tilted_geometry,
    camera_optical_axes_from_rpy,
    interpret_tcp_object_rotation,
    load_stage6_geometry,
    pose_multiply,
    rpy_rotation_matrix,
    surface_frame_from_rpy,
    tcp_target_from_object,
    validate_stage6_geometry,
    vec_dot,
    vec_norm,
    vec_sub,
    generate_roll_candidates,
    view_name_from_normal,
)
from fr_control.inspection_poses import vec_normalize  # noqa: E402
from fr_control.stage4_config import joint_positions, joint_positions_rad, load_yaml  # noqa: E402

_YAML_PATH = os.path.expanduser("~/fairino_ws/src/fr_control/config/stage4_config.yaml")
_EXPECTED_FORWARD = (0.0, 0.70710678, -0.70710678)
_EXPECTED_NORMAL = (0.0, -0.70710678, 0.70710678)


def _pose_xyz(pose: Pose) -> tuple[float, float, float]:
    return (pose.position.x, pose.position.y, pose.position.z)


class Step12CTiltedGeometryTest(unittest.TestCase):
    """Tests 1-13: Home, camera, P1, tilted n_target, A/B/C identities."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_yaml(_YAML_PATH)
        cls.home_deg = joint_positions(cls.cfg)
        cls.home_rad = joint_positions_rad(cls.cfg)
        yaml_geo = load_stage6_geometry(_YAML_PATH)
        cls.yaml_p1 = tuple(yaml_geo["p1_world"])
        cls.data = apply_step12c_tilted_geometry(yaml_geo)
        cls.failures = validate_stage6_geometry(cls.data)
        cls.radius = float(cls.data["radius"])
        cls.height = float(cls.data["height"])
        cls.half = 0.5 * cls.height
        cls.n_target = vec_normalize(cls.data["d1_world"], "n_target")
        cls.cam_fwd, cls.cam_x, cls.cam_y = camera_optical_axes_from_rpy(STEP12C_CAMERA_RPY)
        _, cls.surface_up, cls.surface_n = surface_frame_from_rpy(STEP12C_SURFACE_RPY)

    def test_1_new_home_config_matches_user_values(self) -> None:
        self.assertEqual(len(self.home_deg), 6)
        for got, expected in zip(self.home_deg, STEP12C_HOME_DEG):
            self.assertAlmostEqual(got, expected, places=12)

    def test_2_degree_radian_conversion(self) -> None:
        for deg, rad in zip(self.home_deg, self.home_rad):
            self.assertAlmostEqual(rad, math.radians(deg), places=12)
            self.assertAlmostEqual(deg, math.degrees(rad), places=12)
        self.assertGreater(max(abs(v) for v in self.home_deg), 90.0)
        self.assertLess(max(abs(v) for v in self.home_rad), 3.2)

    def test_3_camera_forward_from_rpy(self) -> None:
        self.assertEqual(STEP12C_CAMERA_POSITION, (0.0, 0.0, 1.40))
        self.assertAlmostEqual(STEP12C_CAMERA_RPY[0], -2.35619449, places=8)
        self.assertAlmostEqual(STEP12C_CAMERA_RPY[1], 0.0, places=12)
        self.assertAlmostEqual(STEP12C_CAMERA_RPY[2], 0.0, places=12)
        for i in range(3):
            self.assertAlmostEqual(self.cam_fwd[i], _EXPECTED_FORWARD[i], places=7)
        rot = rpy_rotation_matrix(*STEP12C_CAMERA_RPY)
        numeric = (
            rot[0][2],
            rot[1][2],
            rot[2][2],
        )
        for i in range(3):
            self.assertAlmostEqual(numeric[i], _EXPECTED_FORWARD[i], places=7)

    def test_4_canonical_surface_normal_from_rpy(self) -> None:
        self.assertAlmostEqual(STEP12C_SURFACE_RPY[0], 0.785398, places=6)
        for i in range(3):
            self.assertAlmostEqual(self.surface_n[i], _EXPECTED_NORMAL[i], places=5)
            self.assertAlmostEqual(self.n_target[i], _EXPECTED_NORMAL[i], places=5)

    def test_5_camera_forward_dot_surface_normal(self) -> None:
        dot = vec_dot(self.cam_fwd, self.surface_n)
        self.assertAlmostEqual(dot, -1.0, places=6)

    def test_6_p1_exactly_user_value(self) -> None:
        p1 = self.data["p1_world"]
        self.assertAlmostEqual(p1[0], 0.0, places=12)
        self.assertAlmostEqual(p1[1], 0.30, places=12)
        self.assertAlmostEqual(p1[2], 1.20, places=12)
        self.assertEqual(STEP12C_P1, (0.0, 0.30, 1.20))
        self.assertNotEqual(self.yaml_p1[1], 0.30)

    def test_7_a_surface_center_is_p1(self) -> None:
        target = self.data["targets"]["side_pos_y"]
        center = actual_view_center(target.object_pose, target.view.center_in_object)
        self.assertLessEqual(vec_norm(vec_sub(center, STEP12C_P1)), 1e-9)

    def test_8_b_surface_center_is_p1(self) -> None:
        target = self.data["targets"]["side_neg_y"]
        center = actual_view_center(target.object_pose, target.view.center_in_object)
        self.assertLessEqual(vec_norm(vec_sub(center, STEP12C_P1)), 1e-9)

    def test_9_c_bottom_surface_center_is_p1(self) -> None:
        target = self.data["targets"]["bottom_circle"]
        center = actual_view_center(target.object_pose, target.view.center_in_object)
        self.assertLessEqual(vec_norm(vec_sub(center, STEP12C_P1)), 1e-9)

    def test_10_abc_actual_normal_is_tilted_target(self) -> None:
        self.assertEqual(self.failures, [])
        for name in ("side_pos_y", "side_neg_y", "bottom_circle"):
            target = self.data["targets"][name]
            rolls = generate_roll_candidates(target, self.data["tcp_t_object"], 5.0)
            self.assertEqual(len(rolls), 72, msg=name)
            for cand in rolls:
                normal = vec_normalize(
                    actual_view_normal(cand.object_pose, target.view.normal_in_object),
                    f"{name} {cand.roll_deg}",
                )
                self.assertLessEqual(
                    cand.normal_angle_error_deg, 1e-6, msg=f"{name} r={cand.roll_deg}"
                )
                self.assertLessEqual(
                    cand.view_center_error_m, 1e-8, msg=f"{name} r={cand.roll_deg}"
                )
                for i in range(3):
                    self.assertAlmostEqual(
                        normal[i], _EXPECTED_NORMAL[i], places=6, msg=f"{name} r={cand.roll_deg}"
                    )

    def test_11_c_local_face_is_object_minus_z(self) -> None:
        view = self.data["views"]["bottom_circle"]
        self.assertEqual(view_name_from_normal((0.0, 0.0, -1.0)), "bottom_circle")
        self.assertAlmostEqual(view.normal_in_object[0], 0.0, places=12)
        self.assertAlmostEqual(view.normal_in_object[1], 0.0, places=12)
        self.assertAlmostEqual(view.normal_in_object[2], -1.0, places=12)
        self.assertAlmostEqual(view.center_in_object[2], -self.half, places=12)
        self.assertEqual(self.data["arm1_views"], ("side_pos_y", "side_neg_y", "bottom_circle"))

    def test_12_top_face_object_plus_z_preserved(self) -> None:
        top = self.data["views"]["top_circle"]
        self.assertEqual(view_name_from_normal((0.0, 0.0, 1.0)), "top_circle")
        self.assertEqual(self.data["arm2_future_view"], "top_circle")
        self.assertAlmostEqual(top.normal_in_object[2], 1.0, places=12)
        self.assertAlmostEqual(top.center_in_object[2], self.half, places=12)
        self.assertNotEqual(
            top.normal_in_object, self.data["views"]["bottom_circle"].normal_in_object
        )

    def test_13_tcp_object_closure(self) -> None:
        tcp = self.data["tcp_t_object"]
        self.assertEqual(interpret_tcp_object_rotation(tcp), "Rx(180 deg)")
        self.assertLessEqual(vec_norm((tcp.position.x, tcp.position.y, tcp.position.z)), 1e-9)
        for name, target in self.data["targets"].items():
            reconstructed = pose_multiply(target.tcp_pose, tcp)
            obj = target.object_pose
            self.assertAlmostEqual(reconstructed.position.x, obj.position.x, places=12, msg=name)
            self.assertAlmostEqual(reconstructed.position.y, obj.position.y, places=12, msg=name)
            self.assertAlmostEqual(reconstructed.position.z, obj.position.z, places=12, msg=name)
            recomputed = tcp_target_from_object(target.object_pose, tcp)
            self.assertAlmostEqual(recomputed.position.x, target.tcp_pose.position.x, places=12, msg=name)

    def test_ab_object_center_sanity_not_authority(self) -> None:
        n = self.n_target
        r = self.radius
        expected = (
            STEP12C_P1[0] - r * n[0],
            STEP12C_P1[1] - r * n[1],
            STEP12C_P1[2] - r * n[2],
        )
        for name in ("side_pos_y", "side_neg_y"):
            obj = _pose_xyz(self.data["targets"][name].object_pose)
            for i in range(3):
                self.assertAlmostEqual(obj[i], expected[i], places=6, msg=name)

    def test_c_object_center_sanity_not_authority(self) -> None:
        n = self.n_target
        offset = self.half
        expected = (
            STEP12C_P1[0] - offset * n[0],
            STEP12C_P1[1] - offset * n[1],
            STEP12C_P1[2] - offset * n[2],
        )
        obj = _pose_xyz(self.data["targets"]["bottom_circle"].object_pose)
        for i in range(3):
            self.assertAlmostEqual(obj[i], expected[i], places=6)

    def test_p1_is_not_forced_onto_optical_axis(self) -> None:
        cam = STEP12C_CAMERA_POSITION
        p1 = STEP12C_P1
        cam_to_p1 = vec_normalize((p1[0] - cam[0], p1[1] - cam[1], p1[2] - cam[2]), "cam->p1")
        dot = vec_dot(cam_to_p1, self.cam_fwd)
        self.assertLess(dot, 0.999)
        self.assertGreater(dot, 0.9)


if __name__ == "__main__":
    unittest.main()
