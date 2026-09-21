#!/usr/bin/env python3

"""DUAL-4E: Arm A return-home model test with Arm B fixed holding small_part.

Does not start dual_bringup, move_group, hardware, or grippers.
Connects to an already running dual-arm /move_group if present.
Does not call /apply_planning_scene, execute(), or send motion.
PLAN ONLY.
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
    return_home_yaml = _existing(
        LaunchConfiguration("return_home_yaml").perform(context),
        os.path.join(planner_share, "config", "dual_arm_a_return_home.yaml"),
        os.path.join(home, "fr_task_ws/src/fr_task_planner/config/dual_arm_a_return_home.yaml"),
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
    step15_yaml = _existing(
        LaunchConfiguration("step15_winner_yaml").perform(context),
        os.path.join(planner_share, "config", "step15_optimized_lift_to_a_winner.yaml"),
        os.path.join(
            home,
            "fr_task_ws/src/fr_task_planner/config/step15_optimized_lift_to_a_winner.yaml",
        ),
    )

    kinematics_params = {}
    if os.path.isfile(kinematics_yaml):
        with open(kinematics_yaml, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, dict):
            kinematics_params = {"robot_description_kinematics": loaded}

    return_home_params = {}
    if os.path.isfile(return_home_yaml):
        with open(return_home_yaml, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, dict):
            return_home_params = loaded

    params = [
        kinematics_params,
        return_home_params,
        {
            "handover_yaml": handover_yaml,
            "kinematics_yaml": kinematics_yaml,
            "step14_winner_yaml": step14_yaml,
            "step15_winner_yaml": step15_yaml,
            "max_ik_attempts": 48,
            "max_unique_candidates": 8,
            "ik_timeout_exact": 0.25,
            "ik_timeout_nearby": 0.05,
            "nearby_radius": 2.0,
            "nearby_radius_local": 0.8,
            "min_ik_solution_distance": return_home_params.get("min_ik_solution_distance", 0.1),
            "tcp_position_tol_m": return_home_params.get("tcp_position_tol_m", 0.001),
            "tcp_orientation_tol_deg": return_home_params.get("tcp_orientation_tol_deg", 1.0),
            "joint_state_timeout_sec": 10.0,
            "joint_state_max_age_sec": 2.0,
            "ik_random_seed": 42,
            "max_contacts": return_home_params.get("max_contacts", 1000),
            "max_contacts_per_pair": return_home_params.get("max_contacts_per_pair", 20),
            "nominal_opening_seed": return_home_params.get("nominal_opening_seed", 0.083),
            "opening_search_min": 0.070,
            "opening_search_max": 0.099,
            "opening_search_step": 0.001,
            "cartesian_step_m": return_home_params.get("cartesian_step_m", 0.005),
            "densify_min_cartesian_m": return_home_params.get("densify_min_cartesian_m", 0.001),
            "max_adjacent_joint_jump_rad": return_home_params.get(
                "max_adjacent_joint_jump_rad", 0.35
            ),
            "ik_timeout_path": return_home_params.get("ik_timeout_path", 0.25),
            "gripper_step_q": return_home_params.get("gripper_step_q", 0.002),
            "arm_a_grasp_q": return_home_params.get("arm_a_grasp_q", 0.083),
            "arm_b_open_q": return_home_params.get("arm_b_open_q", 0.0),
            "arm_b_receive_q": return_home_params.get("arm_b_receive_q", 0.083),
            "object_position_tol_m": return_home_params.get("object_position_tol_m", 0.001),
            "object_orientation_tol_deg": return_home_params.get(
                "object_orientation_tol_deg", 1.0
            ),
            "max_planning_attempts": return_home_params.get("max_planning_attempts", 5),
            "planning_time_sec": return_home_params.get("planning_time_sec", 10.0),
            "max_validation_joint_step": return_home_params.get(
                "max_validation_joint_step", 0.02
            ),
            "home_arrival_tol_rad": return_home_params.get("home_arrival_tol_rad", 0.0001),
        },
    ]

    node = Node(
        package="fr_task_planner",
        executable="dual_arm_a_return_home_test",
        name="dual_arm_a_return_home_test",
        output="screen",
        emulate_tty=True,
        parameters=params,
    )

    return [
        LogInfo(
            msg=[
                "DUAL-4E ARM A RETURN HOME ONLY. "
                "Will not start dual_bringup, apply the scene, or send motion."
            ]
        ),
        LogInfo(msg=[f"handover_yaml={handover_yaml}"]),
        LogInfo(msg=[f"return_home_yaml={return_home_yaml}"]),
        LogInfo(msg=[f"kinematics_yaml={kinematics_yaml}"]),
        LogInfo(msg=[f"step14_winner_yaml={step14_yaml}"]),
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
                "return_home_yaml",
                default_value=os.path.join(
                    home,
                    "fr_task_ws/src/fr_task_planner/config/dual_arm_a_return_home.yaml",
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
            DeclareLaunchArgument(
                "step15_winner_yaml",
                default_value=os.path.join(
                    home,
                    "fr_task_ws/src/fr_task_planner/config/"
                    "step15_optimized_lift_to_a_winner.yaml",
                ),
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
