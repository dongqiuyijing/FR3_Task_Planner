#!/usr/bin/env python3
"""RViz: play every saved keypose for 1 second. Not executable."""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from keypose_dual_model import dual_moveit_params  # noqa: E402


def _setup(context, *args, **kwargs):
    output_dir = LaunchConfiguration("output_dir").perform(context)
    params = dual_moveit_params()
    pkg = get_package_share_directory("fr_task_planner")
    rviz_config = os.path.join(pkg, "config", "keypose_b_face45.rviz")
    if not os.path.isfile(rviz_config):
        rviz_config = os.path.expanduser(
            "~/fr_task_ws/src/fr_task_planner/config/keypose_b_face45.rviz"
        )
    robot_params = {
        "robot_description": params["robot_description"],
        "robot_description_semantic": params["robot_description_semantic"],
        "use_sim_time": False,
    }
    return [
        LogInfo(
            msg=[
                "KEYPOSE TOUR ONLY / NOT EXECUTABLE. ",
                "Each saved keypose holds for 1 second, then the next.",
            ]
        ),
        Node(
            package="fr_task_planner",
            executable="keypose_candidate_rviz_preview.py",
            name="keypose_candidate_rviz_preview",
            output="screen",
            parameters=[
                {
                    "output_dir": output_dir,
                    "cycle": True,
                    "dwell_sec": 1.0,
                    "candidate": 1,
                }
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="keypose_b_face4_rviz",
            output="screen",
            arguments=["-d", rviz_config],
            parameters=[robot_params],
        ),
    ]


def generate_launch_description():
    home = os.path.expanduser("~")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "output_dir",
                default_value=os.path.join(
                    home,
                    "fr_task_ws/src/fr_task_planner/config/keypose_optimization_v1",
                ),
            ),
            OpaqueFunction(function=_setup),
        ]
    )
