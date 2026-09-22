#!/usr/bin/env python3
"""B_FACE4/5 correct-mirror IK search. Local PlanningScene only. No execute."""

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from keypose_dual_model import dual_moveit_params  # noqa: E402


def _setup(context, *args, **kwargs):
    search_yaml = LaunchConfiguration("search_yaml").perform(context)
    output_dir = LaunchConfiguration("output_dir").perform(context)
    params = dual_moveit_params()
    return [
        LogInfo(
            msg=[
                "B_FACE4/5 MIRROR IK ONLY. No path planning, no hardware, ",
                "no other keyposes, collision not relaxed. Max 3 min per face.",
            ]
        ),
        Node(
            package="fr_task_planner",
            executable="keypose_b_face_mirror_search",
            name="keypose_b_face_mirror_search",
            output="screen",
            emulate_tty=True,
            parameters=[
                params,
                {"search_yaml": search_yaml, "output_dir": output_dir},
            ],
        ),
    ]


def generate_launch_description():
    home = os.path.expanduser("~")
    default_dir = os.path.join(
        home, "fr_task_ws/src/fr_task_planner/config/keypose_optimization_v1"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "search_yaml",
                default_value=os.path.join(default_dir, "search.yaml"),
            ),
            DeclareLaunchArgument("output_dir", default_value=default_dir),
            OpaqueFunction(function=_setup),
        ]
    )
