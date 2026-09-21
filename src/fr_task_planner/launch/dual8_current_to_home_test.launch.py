#!/usr/bin/env python3
"""DUAL-8-HOME Current→Home planner. PLAN ONLY. Does not start bringup or send motion."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    home_yaml = LaunchConfiguration("home_yaml").perform(context)
    node = Node(
        package="fr_task_planner",
        executable="dual8_current_to_home_test",
        name="dual8_current_to_home_test",
        output="screen",
        emulate_tty=True,
        parameters=[{"home_yaml": home_yaml}],
    )
    return [
        LogInfo(
            msg=[
                "DUAL-8-HOME PLAN ONLY. Reads live /joint_states by name. "
                "Does not send FollowJointTrajectory or gripper commands. "
                "Does not auto-continue into DUAL-7 grasp."
            ]
        ),
        node,
    ]


def generate_launch_description():
    home = os.path.expanduser("~")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "home_yaml",
                default_value=os.path.join(
                    home,
                    "fr_task_ws/src/fr_task_planner/config/dual8_home.yaml",
                ),
            ),
            OpaqueFunction(function=_setup),
        ]
    )
