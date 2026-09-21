#!/usr/bin/env python3
"""DUAL-7 dual-arm real executor. Default is DRY RUN. Never sets execute:=true."""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load(path):
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _setup(context, *args, **kwargs):
    pkg = get_package_share_directory("fr_task_planner")
    cfg_path = os.path.join(pkg, "config", "dual7_real_executor.yaml")
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.expanduser(
            "~/fr_task_ws/src/fr_task_planner/config/dual7_real_executor.yaml"
        )
    local = _load(cfg_path)
    arm_a = dict(local.get("arm_a") or {})
    arm_b = dict(local.get("arm_b") or {})
    execute = LaunchConfiguration("execute").perform(context).lower() == "true"
    params = {
        "use_sim_time": LaunchConfiguration("use_sim_time").perform(context).lower() == "true",
        "execute": execute,
        "auto_continue": LaunchConfiguration("auto_continue").perform(context).lower() == "true",
        "plan_current_to_home": LaunchConfiguration("plan_current_to_home")
        .perform(context)
        .lower()
        == "true",
        "continue_after_home": LaunchConfiguration("continue_after_home")
        .perform(context)
        .lower()
        == "true",
        "simultaneous_home": LaunchConfiguration("simultaneous_home")
        .perform(context)
        .lower()
        == "true",
        "confirm_handover_approach": LaunchConfiguration("confirm_handover_approach")
        .perform(context)
        .lower()
        == "true",
        "handover_b_grasp_confirmed": LaunchConfiguration("handover_b_grasp_confirmed")
        .perform(context)
        .lower()
        == "true",
        "auto_handover_release": LaunchConfiguration("auto_handover_release")
        .perform(context)
        .lower()
        == "true",
        "handover_release_delay_sec": float(
            LaunchConfiguration("handover_release_delay_sec").perform(context)
        ),
        "segment": LaunchConfiguration("segment").perform(context),
        "resume_from": LaunchConfiguration("resume_from").perform(context),
        "task_mode": LaunchConfiguration("task_mode").perform(context),
        "inject_failure": LaunchConfiguration("inject_failure").perform(context),
        "resume_object_owner": LaunchConfiguration("resume_object_owner").perform(context),
        "b_grasp_confirm_timeout_sec": float(
            LaunchConfiguration("b_grasp_confirm_timeout_sec").perform(context)
        ),
        "max_joint_age_sec": float(LaunchConfiguration("max_joint_age_sec").perform(context)),
        "real_robot_confirmation": LaunchConfiguration("real_robot_confirmation").perform(
            context
        ),
        "trajectory_speed_scale": float(
            LaunchConfiguration("trajectory_speed_scale").perform(context)
        ),
        "step15_trajectory": str(
            local.get(
                "step15_trajectory",
                os.path.expanduser(
                    "~/fr_task_ws/src/fr_task_planner/config/"
                    "step15_optimized_lift_to_a_trajectory.yaml"
                ),
            )
        ),
        "segments_file": str(
            local.get(
                "segments_file",
                os.path.expanduser(
                    "~/fr_task_ws/src/fr_task_planner/config/dual7_real_task_segments.yaml"
                ),
            )
        ),
        "home_yaml": str(
            local.get(
                "home_yaml",
                os.path.expanduser("~/fr_task_ws/src/fr_task_planner/config/dual8_home.yaml"),
            )
        ),
        "b_home_to_pre_file": str(
            local.get(
                "b_home_to_pre_file",
                os.path.expanduser(
                    "~/fr_task_ws/src/fr_task_planner/config/dual8_b_home_to_pre_handover.yaml"
                ),
            )
        ),
        "arm_a_trajectory_action_name": str(
            arm_a.get(
                "trajectory_action_name",
                "/arm_a_controller/follow_joint_trajectory",
            )
        ),
        "arm_b_trajectory_action_name": str(
            arm_b.get(
                "trajectory_action_name",
                "/arm_b_controller/follow_joint_trajectory",
            )
        ),
        "arm_a_gripper_service_name": str(
            arm_a.get("gripper_service_name", "/arm_a/fairino_gripper/command")
        ),
        "arm_b_gripper_service_name": str(
            arm_b.get("gripper_service_name", "/arm_b/fairino_gripper/command")
        ),
        "arm_a_planning_group": str(arm_a.get("planning_group", "arm_a")),
        "arm_b_planning_group": str(arm_b.get("planning_group", "arm_b")),
        "gripper_open_position": int(local.get("gripper_open_position", 0)),
        "gripper_close_position": int(local.get("gripper_close_position", 85)),
        "gripper_velocity": int(local.get("gripper_velocity", 20)),
        "gripper_force": int(local.get("gripper_force", 20)),
        "gripper_id": int(local.get("gripper_id", 1)),
        "gripper_max_time_ms": int(local.get("gripper_max_time_ms", 5000)),
        "gripper_timeout_sec": float(local.get("gripper_timeout_sec", 15.0)),
        "hardware_servo_restart_margin_ms": int(
            local.get("hardware_servo_restart_margin_ms", 4000)
        ),
        "gripper_a_already_activated": LaunchConfiguration("gripper_a_already_activated")
        .perform(context)
        .lower()
        == "true",
        "gripper_b_already_activated": LaunchConfiguration("gripper_b_already_activated")
        .perform(context)
        .lower()
        == "true",
        "home_tolerance_rad": float(local.get("home_tolerance_rad", 0.02)),
        "segment_start_tolerance_rad": float(local.get("segment_start_tolerance_rad", 0.02)),
        "segment_end_tolerance_rad": float(local.get("segment_end_tolerance_rad", 0.02)),
        "startup_ready_timeout_sec": float(local.get("startup_ready_timeout_sec", 15.0)),
        "conservative_joint_vmax_rad_s": float(local.get("conservative_joint_vmax_rad_s", 0.20)),
        "config_file": cfg_path,
    }
    node = Node(
        package="fr_task_planner",
        executable="dual7_real_task_executor",
        name="dual7_real_task_executor",
        output="screen",
        emulate_tty=True,
        parameters=[params],
    )
    return [
        LogInfo(
            msg=[
                "DUAL-7 REAL TASK EXECUTOR. Default DRY RUN. execute=",
                LaunchConfiguration("execute"),
                " auto_continue=",
                LaunchConfiguration("auto_continue"),
                " A=",
                params["arm_a_trajectory_action_name"],
                " B=",
                params["arm_b_trajectory_action_name"],
            ]
        ),
        node,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("execute", default_value="false"),
            DeclareLaunchArgument("auto_continue", default_value="false"),
            DeclareLaunchArgument("plan_current_to_home", default_value="false"),
            DeclareLaunchArgument("continue_after_home", default_value="false"),
            DeclareLaunchArgument("simultaneous_home", default_value="false"),
            DeclareLaunchArgument("confirm_handover_approach", default_value="false"),
            DeclareLaunchArgument("handover_b_grasp_confirmed", default_value="false"),
            DeclareLaunchArgument(
                "auto_handover_release",
                default_value="false",
                description="If true, after gripper_close_b SUCCESS wait then allow Arm A open "
                "(skip /confirm_b_grasp). Default false keeps operator confirm.",
            ),
            DeclareLaunchArgument(
                "handover_release_delay_sec",
                default_value="2.0",
                description="Seconds to wait after B CLOSE SUCCESS before auto Arm A release.",
            ),
            DeclareLaunchArgument("segment", default_value=""),
            DeclareLaunchArgument("resume_from", default_value=""),
            DeclareLaunchArgument("task_mode", default_value="home_only"),
            DeclareLaunchArgument("inject_failure", default_value=""),
            DeclareLaunchArgument("resume_object_owner", default_value="unknown"),
            DeclareLaunchArgument("b_grasp_confirm_timeout_sec", default_value="600.0"),
            DeclareLaunchArgument("max_joint_age_sec", default_value="2.0"),
            DeclareLaunchArgument("real_robot_confirmation", default_value=""),
            DeclareLaunchArgument("trajectory_speed_scale", default_value="0.3"),
            DeclareLaunchArgument("gripper_a_already_activated", default_value="false"),
            DeclareLaunchArgument("gripper_b_already_activated", default_value="false"),
            LogInfo(
                msg=[
                    "DUAL-7 real executor: default does not auto-run the full task. ",
                    "Physical B grasp is NOT implied by Attachment. ",
                    "OMPL handover approach is NOT a verified Cartesian line. ",
                    "Activate is explicit ActGripper(id,1); reset is never sent. ",
                    "Open/close never sent unless execute:=true and confirmation matches. ",
                    "dual_current_to_home sends ZERO gripper commands. ",
                    "task_mode:=home_only stops after Home. ",
                    "task_mode:=full_task continues the six-face task after real Home. ",
                    "segment:=ID is one stage; resume_from:=ID continues from that stage. ",
                    "B grasp confirm is a runtime service, not a launch-time flag.",
                ]
            ),
            OpaqueFunction(function=_setup),
        ]
    )
