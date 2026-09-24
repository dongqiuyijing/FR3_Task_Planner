#!/usr/bin/env python3
"""Offline FK/IK and collision validation for B_HANDOVER world compensation."""

import os
import sys

from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from keypose_dual_model import dual_moveit_params  # noqa: E402


def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg="B_HANDOVER OFFSET VALIDATION ONLY. No hardware, motion, or gripper commands."),
        Node(package="fr_task_planner", executable="keypose_v1_b_handover_offset",
             name="keypose_v1_b_handover_offset", output="screen",
             parameters=[dual_moveit_params()]),
    ])
