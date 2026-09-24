#!/usr/bin/env python3
"""RViz-only frozen-versus-offset B_HANDOVER comparison."""

import os
import sys

from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from keypose_dual_model import dual_moveit_params  # noqa: E402


def generate_launch_description():
    params = dual_moveit_params()
    robot = {"robot_description": params["robot_description"],
             "robot_description_semantic": params["robot_description_semantic"],
             "use_sim_time": False}
    rviz = os.path.expanduser(
        "~/fr_task_ws/src/fr_task_planner/config/keypose_v1_b_handover_offset.rviz")
    return LaunchDescription([
        LogInfo(msg="B_HANDOVER OFFSET RViz PREVIEW ONLY. Does not move robot or grippers."),
        Node(package="fr_task_planner", executable="keypose_v1_b_handover_offset_preview.py",
             name="keypose_v1_b_handover_offset_preview", output="screen"),
        Node(package="rviz2", executable="rviz2", name="keypose_v1_b_handover_offset_rviz",
             output="screen", arguments=["-d", rviz], parameters=[robot]),
    ])
