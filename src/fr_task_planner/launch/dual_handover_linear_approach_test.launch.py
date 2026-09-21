#!/usr/bin/env python3

"""DUAL-4C: dual-arm handover linear approach test.

Does not start dual_bringup, move_group, hardware, or grippers.
Connects to an already running dual-arm /move_group if present.
Does not call /apply_planning_scene, execute(), or send motion.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _pkg_file(package, *parts):
    try:
        return os.path.join(get_package_share_directory(package), *parts)
    except Exception:
        return ""


def _existing(*candidates):
    for path in candidates:
        if path and os.path.isfile(os.path.expanduser(path)):
            return os.path.expanduser(path)
    return candidates[0] if candidates else ""


def _launch_setup(context, *args, **kwargs):
    planner_share = get_package_share_directory("fr_task_planner")
    home = os.path.expanduser("~")

    handover_yaml = _existing(
        LaunchConfiguration("handover_yaml").perform(context),
        os.path.join(planner_share, "config", "dual_handover.yaml"),
        os.path.join(home, "fr_task_ws/src/fr_task_planner/config/dual_handover.yaml"),
    )
    linear_yaml = _existing(
        LaunchConfiguration("linear_approach_yaml").perform(context),
        os.path.join(planner_share, "config", "dual_handover_linear_approach.yaml"),
        os.path.join(
            home, "fr_task_ws/src/fr_task_planner/config/dual_handover_linear_approach.yaml"
        ),
    )
    kinematics_yaml = _existing(
        LaunchConfiguration("kinematics_yaml").perform(context),
        _pkg_file("fairino3_dual_moveit_config", "config", "kinematics.yaml"),
        os.path.join(
            home,
            "fairino_ws/src/fairino3_dual_moveit_config/config/kinematics.yaml",
        ),
        os.path.join(
            home,
            "fairino_ws/install/fairino3_dual_moveit_config/share/"
            "fairino3_dual_moveit_config/config/kinematics.yaml",
        ),
    )
    step14_yaml = _existing(
        LaunchConfiguration("step14_winner_yaml").perform(context),
        os.path.join(planner_share, "config", "step14_optimized_grasp_winner.yaml"),
        os.path.join(
            home,
            "fr_task_ws/src/fr_task_planner/config/step14_optimized_grasp_winner.yaml",
        ),
    )

    kinematics_params = {}
    if os.path.isfile(kinematics_yaml):
        with open(kinematics_yaml, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, dict):
            kinematics_params = {"robot_description_kinematics": loaded}

    linear_params = {}
    if os.path.isfile(linear_yaml):
        with open(linear_yaml, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, dict):
            linear_params = loaded

    params = [
        kinematics_params,
        linear_params,
        {
            "handover_yaml": handover_yaml,
            "kinematics_yaml": kinematics_yaml,
            "step14_winner_yaml": step14_yaml,
            "max_ik_attempts": 48,
            "max_unique_candidates": 8,
            "ik_timeout_exact": 0.25,
            "ik_timeout_nearby": 0.05,
            "nearby_radius": 2.0,
            "nearby_radius_local": 0.8,
            "min_ik_solution_distance": linear_params.get("min_ik_solution_distance", 0.1),
            "tcp_position_tol_m": linear_params.get("tcp_position_tol_m", 0.001),
            "tcp_orientation_tol_deg": linear_params.get("tcp_orientation_tol_deg", 1.0),
            "joint_state_timeout_sec": 10.0,
            "joint_state_max_age_sec": 2.0,
            "ik_random_seed": 42,
            "max_contacts": linear_params.get("max_contacts", 1000),
            "max_contacts_per_pair": linear_params.get("max_contacts_per_pair", 20),
            "nominal_opening_seed": linear_params.get("nominal_opening_seed", 0.083),
            "opening_search_min": 0.070,
            "opening_search_max": 0.099,
            "opening_search_step": 0.001,
            "cartesian_step_m": linear_params.get("cartesian_step_m", 0.005),
            "densify_min_cartesian_m": linear_params.get("densify_min_cartesian_m", 0.001),
            "max_adjacent_joint_jump_rad": linear_params.get(
                "max_adjacent_joint_jump_rad", 0.35
            ),
            "ik_timeout_path": linear_params.get("ik_timeout_path", 0.25),
        },
    ]

    node = Node(
        package="fr_task_planner",
        executable="dual_handover_linear_approach_test",
        name="dual_handover_linear_approach_test",
        output="screen",
        emulate_tty=True,
        parameters=params,
    )

    return [
        LogInfo(
            msg=[
                "DUAL-4C LINEAR APPROACH ONLY. "
                "Will not start dual_bringup, apply the scene, or send motion."
            ]
        ),
        LogInfo(msg=[f"handover_yaml={handover_yaml}"]),
        LogInfo(msg=[f"linear_approach_yaml={linear_yaml}"]),
        LogInfo(msg=[f"kinematics_yaml={kinematics_yaml}"]),
        node,
    ]


def generate_launch_description():
    home = os.path.expanduser("~")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "handover_yaml",
                default_value=os.path.join(
                    home,
                    "fr_task_ws/src/fr_task_planner/config/dual_handover.yaml",
                ),
            ),
            DeclareLaunchArgument(
                "linear_approach_yaml",
                default_value=os.path.join(
                    home,
                    "fr_task_ws/src/fr_task_planner/config/"
                    "dual_handover_linear_approach.yaml",
                ),
            ),
            DeclareLaunchArgument(
                "kinematics_yaml",
                default_value=os.path.join(
                    home,
                    "fairino_ws/src/fairino3_dual_moveit_config/config/"
                    "kinematics.yaml",
                ),
            ),
            DeclareLaunchArgument(
                "step14_winner_yaml",
                default_value=os.path.join(
                    home,
                    "fr_task_ws/src/fr_task_planner/config/"
                    "step14_optimized_grasp_winner.yaml",
                ),
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
