"""Load dual-arm robot_description for keypose tools.

Read-only: does not start bringup, controllers, or hardware.
"""

from __future__ import annotations

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue

from fr_control.grasp_poses import xyzw_to_rpy
from fr_control.inspection_poses import as_vec3
from fr_control.stage4_config import (
    as_xyzw,
    load_yaml,
    robot_base_rpy,
    robot_base_xyz,
    workcell_config_path,
)


def _arm_b_xyz_rpy(backends):
    pose = backends["arm_b"]["base_pose"]
    x, y, z = as_vec3(pose["position"])
    qx, qy, qz, qw = as_xyzw(pose["orientation_xyzw"])
    roll, pitch, yaw = xyzw_to_rpy(qx, qy, qz, qw)
    return (x, y, z), (roll, pitch, yaw)


def dual_xacro_mappings():
    pkg_share = get_package_share_directory("fairino3_dual_moveit_config")
    backends = load_yaml(os.path.join(pkg_share, "config", "dual_backends.yaml"))
    stage4 = load_yaml(workcell_config_path())
    ax, ay, az = robot_base_xyz(stage4)
    a_roll, a_pitch, a_yaw = robot_base_rpy(stage4)
    (bx, by, bz), (b_roll, b_pitch, b_yaw) = _arm_b_xyz_rpy(backends)
    initial_positions = os.path.join(pkg_share, "config", "initial_positions.yaml")
    return {
        "initial_positions_file": initial_positions,
        "arm_a_x": str(ax),
        "arm_a_y": str(ay),
        "arm_a_z": str(az),
        "arm_a_roll": str(a_roll),
        "arm_a_pitch": str(a_pitch),
        "arm_a_yaw": str(a_yaw),
        "arm_b_x": str(bx),
        "arm_b_y": str(by),
        "arm_b_z": str(bz),
        "arm_b_roll": str(b_roll),
        "arm_b_pitch": str(b_pitch),
        "arm_b_yaw": str(b_yaw),
    }


def dual_moveit_params():
    """Return robot_description params. Does not start move_group.

    Avoid MoveItConfigsBuilder here: it requires pilz_cartesian_limits.yaml
    which this dual package does not ship. Search only needs URDF/SRDF/KDL.
    """
    return {
        "robot_description": dual_robot_description_command(),
        "robot_description_semantic": dual_srdf_text(),
        "robot_description_kinematics": {
            "arm_a": {
                "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
                "kinematics_solver_search_resolution": 0.005,
                "kinematics_solver_timeout": 0.05,
            },
            "arm_b": {
                "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
                "kinematics_solver_search_resolution": 0.005,
                "kinematics_solver_timeout": 0.05,
            },
        },
        "use_sim_time": False,
    }


def dual_robot_description_command():
    pkg_share = get_package_share_directory("fairino3_dual_moveit_config")
    dual_xacro = os.path.join(pkg_share, "urdf", "dual_fr3.urdf.xacro")
    mappings = dual_xacro_mappings()
    cmd = ["xacro ", dual_xacro]
    for key, value in mappings.items():
        cmd.extend([" ", f"{key}:={value}"])
    return ParameterValue(Command(cmd), value_type=str)


def dual_srdf_text():
    pkg_share = get_package_share_directory("fairino3_dual_moveit_config")
    path = os.path.join(pkg_share, "config", "dual_fr3.srdf")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def load_yaml_file(path: str):
    with open(os.path.expanduser(path), encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
