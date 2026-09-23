#!/usr/bin/env python3
"""Offline-only B_FACE6 table-place candidate generator."""
import os
import sys

from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from keypose_dual_model import dual_moveit_params  # noqa: E402


def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg='B PLACE CANDIDATE: offline FK/IK/collision generation only; no robot or gripper commands.'),
        Node(package='fr_task_planner', executable='keypose_v1_b_place_candidate',
             name='keypose_v1_b_place_candidate', output='screen',
             parameters=[dual_moveit_params()]),
    ])
