#!/usr/bin/env python3
"""Solve the placement hover B_HOME. Plan only."""

import os
import sys

from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from keypose_dual_model import dual_moveit_params  # noqa: E402


def generate_launch_description():
    return LaunchDescription(
        [
            LogInfo(msg="B_HOME IK only. Does not move the robot."),
            Node(
                package="fr_task_planner",
                executable="keypose_v1_b_home",
                name="keypose_v1_b_home",
                output="screen",
                parameters=[dual_moveit_params()],
            ),
        ]
    )
