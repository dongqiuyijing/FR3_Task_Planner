#!/usr/bin/env python3
"""Manual KEYPOSE_V1 executor entry. Default is dry-run. Does not set execute:=true."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    execute = LaunchConfiguration("execute").perform(context).lower() == "true"
    home = os.path.expanduser("~")
    traj = LaunchConfiguration("trajectory").perform(context)
    trajectory_speed_scale = float(
        LaunchConfiguration("trajectory_speed_scale").perform(context)
    )
    return [
        LogInfo(
            msg=[
                "KEYPOSE_V1 executor. execute is false unless you pass execute:=true ",
                "after checking RViz. Set trajectory_speed_scale:=0.4 (or 0.5) "
                "to override the default 0.2. Old DUAL-7 files are not used.",
            ]
        ),
        Node(
            package="fr_task_planner",
            executable="dual7_real_task_executor",
            name="keypose_v1_executor",
            output="screen",
            parameters=[
                {
                    "execute": execute,
                    "task_mode": "keypose_v1",
                    "trajectory_speed_scale": trajectory_speed_scale,
                    "grasp_post_close_wait_sec": 1.5,
                    "handover_post_close_wait_sec": 1.5,
                    "place_post_open_wait_sec": 1.5,
                    "auto_handover_release": LaunchConfiguration("auto_handover_release").perform(context).lower() ==
                        "true",
                    "keypose_v1_trajectory": traj,
                    "real_robot_confirmation": LaunchConfiguration("real_robot_confirmation").perform(
                        context
                    ),
                    "empty_gripper_confirmation": LaunchConfiguration(
                        "empty_gripper_confirmation"
                    ).perform(context),
                }
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("execute", default_value="false"),
            DeclareLaunchArgument("real_robot_confirmation", default_value=""),
            DeclareLaunchArgument("empty_gripper_confirmation", default_value=""),
            DeclareLaunchArgument("auto_handover_release", default_value="false"),
            DeclareLaunchArgument("trajectory_speed_scale", default_value="0.2"),
            DeclareLaunchArgument(
                "trajectory",
                default_value=os.path.expanduser(
                    "~/fr_task_ws/src/fr_task_planner/config/keypose_optimization_v1/"
                    "keypose_v1_six_face_with_place_candidate.yaml"
                ),
            ),
            OpaqueFunction(function=_setup),
        ]
    )
