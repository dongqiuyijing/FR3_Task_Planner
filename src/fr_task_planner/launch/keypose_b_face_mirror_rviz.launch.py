#!/usr/bin/env python3
"""RViz display of A_FACE1/2 vs B_FACE4/5 mirror vs DUAL-7 part poses. Not executable."""

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
    show_b = LaunchConfiguration("show_b").perform(context)
    params = dual_moveit_params()
    pkg = get_package_share_directory("fr_task_planner")
    rviz_config = os.path.join(pkg, "config", "keypose_candidate.rviz")
    if not os.path.isfile(rviz_config):
        rviz_config = os.path.expanduser(
            "~/fr_task_ws/src/fr_task_planner/config/keypose_candidate.rviz"
        )
    robot_params = {
        "robot_description": params["robot_description"],
        "robot_description_semantic": params["robot_description_semantic"],
        "use_sim_time": False,
    }
    return [
        LogInfo(
            msg=[
                "B_FACE4/5 MIRROR TARGET PREVIEW ONLY / NOT EXECUTABLE. ",
                "orange=A_FACE1/2  cyan/blue=B mirror  red/magenta=DUAL-7. ",
                f"robot B joints from {show_b}",
            ]
        ),
        Node(
            package="fr_task_planner",
            executable="keypose_b_face_mirror_preview.py",
            name="keypose_b_face_mirror_preview",
            output="screen",
            parameters=[{"output_dir": output_dir, "show_b": show_b}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="keypose_b_face_mirror_rviz",
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
            DeclareLaunchArgument("show_b", default_value="DUAL7_I1"),
            OpaqueFunction(function=_setup),
        ]
    )
