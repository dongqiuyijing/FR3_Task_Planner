#!/usr/bin/env python3
"""STEP 12 fixed-P1 geometry regressions. No MoveIt, no Gazebo, no execution."""

from __future__ import annotations

import os
import unittest

from geometry_msgs.msg import Pose

_LAUNCH_DIR = os.path.join(os.path.dirname(__file__), "..", "launch")
import sys

sys.path.insert(0, os.path.abspath(_LAUNCH_DIR))

from inspection_view_geometry import (  # noqa: E402
    actual_view_center,
    actual_view_normal,
    generate_roll_candidates,
    interpret_tcp_object_rotation,
    load_stage6_geometry,
    pose_multiply,
    retarget_inspection_p1,
    tcp_target_from_object,
    validate_stage6_geometry,
    vec_norm,
    vec_sub,
)
from fr_control.inspection_poses import vec_normalize  # noqa: E402

FIXED_P1 = (0.0, 0.4, 1.20)


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


if __name__ == "__main__":
    unittest.main()
