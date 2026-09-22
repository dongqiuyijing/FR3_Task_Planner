#!/usr/bin/env python3
"""RViz playback of the KEYPOSE_V1 connected trajectory. Not executable."""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from keypose_dual_model import dual_moveit_params  # noqa: E402


def generate_launch_description():
    params = dual_moveit_params()
    pkg = get_package_share_directory("fr_task_planner")
    rviz = os.path.join(pkg, "config", "keypose_b_face45.rviz")
    if not os.path.isfile(rviz):
        rviz = os.path.expanduser(
            "~/fr_task_ws/src/fr_task_planner/config/keypose_b_face45.rviz"
        )
    robot = {
        "robot_description": params["robot_description"],
        "robot_description_semantic": params["robot_description_semantic"],
        "use_sim_time": False,
    }
    return LaunchDescription(
        [
            LogInfo(msg="KEYPOSE V1 TRAJECTORY PREVIEW ONLY. Does not move the robot."),
            Node(
                package="fr_task_planner",
                executable="keypose_v1_tour_preview.py",
                name="keypose_v1_tour_preview",
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="keypose_v1_rviz",
                output="screen",
                arguments=["-d", rviz],
                parameters=[robot],
            ),
        ]
    )
