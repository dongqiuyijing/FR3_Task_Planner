#!/usr/bin/env python3
"""Connect confirmed KEYPOSE_V1 poses. Plan only."""

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from keypose_dual_model import dual_moveit_params  # noqa: E402


def _setup(context, *args, **kwargs):
    only = LaunchConfiguration("only_a_return").perform(context).lower() == "true"
    return [
        LogInfo(msg="KEYPOSE_V1 connect only. Does not move the robot or rewrite keyposes."),
        Node(
            package="fr_task_planner",
            executable="keypose_v1_connect",
            name="keypose_v1_connect",
            output="screen",
            parameters=[
                dual_moveit_params(),
                {"only_a_return": only},
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("only_a_return", default_value="false"),
            OpaqueFunction(function=_setup),
        ]
    )
