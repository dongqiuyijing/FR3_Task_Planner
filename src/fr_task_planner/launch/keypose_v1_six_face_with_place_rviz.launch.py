#!/usr/bin/env python3
"""Offline complete-six-face-plus-place RViz playback; never starts hardware."""
import os
import sys
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Launch files are installed independently; add this package's launch directory
# so the adjacent shared model helper is importable in both source/install use.
sys.path.insert(0, os.path.dirname(__file__))
from keypose_dual_model import dual_moveit_params


def generate_launch_description():
    default = os.path.expanduser(
        "~/fr_task_ws/src/fr_task_planner/config/keypose_optimization_v1/"
        "keypose_v1_six_face_with_place_candidate.yaml"
    )
    return LaunchDescription([
        DeclareLaunchArgument("trajectory", default_value=default),
        LogInfo(msg="OFFLINE RViz playback only: no controllers, robot, or gripper commands."),
        Node(package="fr_task_planner", executable="dual_complete_six_face_rviz_preview.py",
             name="keypose_six_face_place_preview", output="screen",
             arguments=[LaunchConfiguration("trajectory")]),
        Node(package="rviz2", executable="rviz2", name="keypose_six_face_place_rviz",
             output="screen", arguments=["-d", os.path.expanduser(
                 "~/fr_task_ws/src/fr_task_planner/config/dual7_complete_task.rviz")],
             parameters=[dual_moveit_params()]),
    ])
