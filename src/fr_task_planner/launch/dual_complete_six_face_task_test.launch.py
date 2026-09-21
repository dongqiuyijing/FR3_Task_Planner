#!/usr/bin/env python3
"""DUAL-7 complete six-face model demo. PLAN ONLY. Does not start bringup."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    home = os.path.expanduser("~")
    task_yaml = LaunchConfiguration("task_yaml").perform(context)
    node = Node(
        package="fr_task_planner",
        executable="dual_complete_six_face_task_test",
        name="dual_complete_six_face_task_test",
        output="screen",
        emulate_tty=True,
        parameters=[{"task_yaml": task_yaml}],
    )
    return [
        LogInfo(
            msg=[
                "DUAL-7 COMPLETE SIX-FACE MODEL DEMO. PLAN ONLY. "
                "Will not start dual_bringup, apply the scene, or send motion."
            ]
        ),
        node,
    ]


def generate_launch_description():
    home = os.path.expanduser("~")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "task_yaml",
                default_value=os.path.join(
                    home,
                    "fr_task_ws/src/fr_task_planner/config/"
                    "dual_complete_six_face_task.yaml",
                ),
            ),
            OpaqueFunction(function=_setup),
        ]
    )
