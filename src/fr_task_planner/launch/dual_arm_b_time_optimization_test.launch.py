#!/usr/bin/env python3

"""DUAL-6: Arm B H→I1→I2→I3 time-first optimization. PLAN ONLY / OFFLINE TOTG.

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
    sequence_yaml = _existing(
        LaunchConfiguration("sequence_yaml").perform(context),
        os.path.join(planner_share, "config", "dual_arm_b_inspection_sequence.yaml"),
        os.path.join(
            home, "fr_task_ws/src/fr_task_planner/config/dual_arm_b_inspection_sequence.yaml"
        ),
    )
    optimization_yaml = _existing(
        LaunchConfiguration("optimization_yaml").perform(context),
        os.path.join(planner_share, "config", "dual_arm_b_time_optimization.yaml"),
        os.path.join(
            home, "fr_task_ws/src/fr_task_planner/config/dual_arm_b_time_optimization.yaml"
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
    step12c_yaml = _existing(
        LaunchConfiguration("step12c_winner_yaml").perform(context),
        os.path.join(planner_share, "config", "step12c_tilted_camera_winner.yaml"),
        os.path.join(
            home,
            "fr_task_ws/src/fr_task_planner/config/step12c_tilted_camera_winner.yaml",
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

    sequence_params = {}
    if os.path.isfile(sequence_yaml):
        with open(sequence_yaml, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, dict):
            # Nested maps are loaded by C++ from the YAML path, not flattened here.
            sequence_params = {
                key: value
                for key, value in loaded.items()
                if not isinstance(value, (dict, list))
            }

    optimization_params = {}
    if os.path.isfile(optimization_yaml):
        with open(optimization_yaml, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, dict):
            optimization_params = {
                key: value
                for key, value in loaded.items()
                if not isinstance(value, (dict, list))
            }
            nested = loaded.get("optimization")
            if isinstance(nested, dict):
                for key, value in nested.items():
                    if not isinstance(value, (dict, list)):
                        optimization_params.setdefault(key, value)

    params = [
        kinematics_params,
        sequence_params,
        optimization_params,
        {
            "handover_yaml": handover_yaml,
            "sequence_yaml": sequence_yaml,
            "optimization_yaml": optimization_yaml,
            "kinematics_yaml": kinematics_yaml,
            "step12c_winner_yaml": step12c_yaml,
            "step14_winner_yaml": step14_yaml,
            "step15_winner_yaml": step15_yaml,
            "max_ik_attempts": 48,
            "max_unique_candidates": 8,
            "ik_timeout_exact": 0.25,
            "ik_timeout_nearby": 0.05,
            "nearby_radius": 2.0,
            "nearby_radius_local": 0.8,
            "min_ik_solution_distance": sequence_params.get("min_ik_solution_distance", 0.1),
            "tcp_position_tol_m": sequence_params.get("tcp_position_tol_m", 0.001),
            "tcp_orientation_tol_deg": sequence_params.get("tcp_orientation_tol_deg", 1.0),
            "joint_state_timeout_sec": 10.0,
            "joint_state_max_age_sec": 2.0,
            "ik_random_seed": 42,
            "max_contacts": sequence_params.get("max_contacts", 1000),
            "max_contacts_per_pair": sequence_params.get("max_contacts_per_pair", 20),
            "nominal_opening_seed": sequence_params.get("nominal_opening_seed", 0.083),
            "opening_search_min": 0.070,
            "opening_search_max": 0.099,
            "opening_search_step": 0.001,
            "cartesian_step_m": sequence_params.get("cartesian_step_m", 0.005),
            "densify_min_cartesian_m": sequence_params.get("densify_min_cartesian_m", 0.001),
            "max_adjacent_joint_jump_rad": sequence_params.get(
                "max_adjacent_joint_jump_rad", 0.35
            ),
            "ik_timeout_path": sequence_params.get("ik_timeout_path", 0.25),
            "gripper_step_q": sequence_params.get("gripper_step_q", 0.002),
            "arm_a_grasp_q": sequence_params.get("arm_a_grasp_q", 0.083),
            "arm_b_open_q": sequence_params.get("arm_b_open_q", 0.0),
            "arm_b_receive_q": sequence_params.get("arm_b_receive_q", 0.083),
            "object_position_tol_m": sequence_params.get("object_position_tol_m", 0.001),
            "object_orientation_tol_deg": sequence_params.get(
                "object_orientation_tol_deg", 1.0
            ),
            "max_planning_attempts": optimization_params.get(
                "ompl_attempts_per_edge", sequence_params.get("max_planning_attempts", 5)
            ),
            "planning_time_sec": sequence_params.get("planning_time_sec", 10.0),
            "max_validation_joint_step": optimization_params.get(
                "max_validation_joint_step_rad",
                sequence_params.get("max_validation_joint_step", 0.02),
            ),
            "home_arrival_tol_rad": sequence_params.get("home_arrival_tol_rad", 0.0001),
            "chain_joint_continuity_tol_rad": sequence_params.get(
                "chain_joint_continuity_tol_rad", 0.0001
            ),
            "face_center_tol_m": sequence_params.get("face_center_tol_m", 0.002),
            "face_normal_tol_deg": sequence_params.get("face_normal_tol_deg", 3.0),
            "max_ik_candidates_per_target": sequence_params.get(
                "max_ik_candidates_per_target", 8
            ),
        },
    ]

    node = Node(
        package="fr_task_planner",
        executable="dual_arm_b_time_optimization_test",
        name="dual_arm_b_time_optimization_test",
        output="screen",
        emulate_tty=True,
        parameters=params,
    )

    return [
        LogInfo(
            msg=[
                "DUAL-6 ARM B TIME-FIRST OPTIMIZATION ONLY. "
                "Will not start dual_bringup, apply the scene, or send motion."
            ]
        ),
        LogInfo(msg=[f"handover_yaml={handover_yaml}"]),
        LogInfo(msg=[f"sequence_yaml={sequence_yaml}"]),
        LogInfo(msg=[f"optimization_yaml={optimization_yaml}"]),
        LogInfo(msg=[f"step12c_winner_yaml={step12c_yaml}"]),
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
                "sequence_yaml",
                default_value=os.path.join(
                    home,
                    "fr_task_ws/src/fr_task_planner/config/"
                    "dual_arm_b_inspection_sequence.yaml",
                ),
            ),
            DeclareLaunchArgument(
                "optimization_yaml",
                default_value=os.path.join(
                    home,
                    "fr_task_ws/src/fr_task_planner/config/"
                    "dual_arm_b_time_optimization.yaml",
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
                "step12c_winner_yaml",
                default_value=os.path.join(
                    home,
                    "fr_task_ws/src/fr_task_planner/config/"
                    "step12c_tilted_camera_winner.yaml",
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
