#!/usr/bin/env python3
"""B_FACE4/5 root-cause diagnosis. Local PlanningScene only. No execute."""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from keypose_dual_model import dual_moveit_params  # noqa: E402


def _setup(context, *args, **kwargs):
    search_yaml = LaunchConfiguration("search_yaml").perform(context)
    output_dir = LaunchConfiguration("output_dir").perform(context)
    hold = LaunchConfiguration("hold").perform(context)
    start_rviz = LaunchConfiguration("rviz").perform(context)
    keypose = LaunchConfiguration("keypose").perform(context)
    params = dual_moveit_params()
    pkg = get_package_share_directory("fr_task_planner")
    rviz_config = os.path.join(pkg, "config", "keypose_candidate.rviz")
    if not os.path.isfile(rviz_config):
        rviz_config = os.path.expanduser(
            "~/fr_task_ws/src/fr_task_planner/config/keypose_candidate.rviz"
        )

    actions = [
        LogInfo(
            msg=[
                "B_FACE4/5 DIAGNOSIS ONLY. No candidate rewrite, no follow-joint, ",
                "no hardware, no full MTC/OMPL search.",
            ]
        ),
        Node(
            package="fr_task_planner",
            executable="keypose_b_face_diag",
            name="keypose_b_face_diag",
            output="screen",
            emulate_tty=True,
            parameters=[
                params,
                {
                    "search_yaml": search_yaml,
                    "output_dir": output_dir,
                    "hold": hold.lower() in ("1", "true", "yes"),
                },
            ],
        ),
    ]
    if start_rviz.lower() in ("1", "true", "yes"):
        actions.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="keypose_b_face_diag_rviz",
                output="screen",
                arguments=["-d", rviz_config],
                parameters=[
                    {
                        "robot_description": params["robot_description"],
                        "robot_description_semantic": params["robot_description_semantic"],
                        "use_sim_time": False,
                    }
                ],
                condition=IfCondition("true"),
            )
        )
        actions.append(
            Node(
                package="fr_task_planner",
                executable="keypose_b_face_diag_preview.py",
                name="keypose_b_face_diag_preview",
                output="screen",
                parameters=[
                    params,
                    {
                        "output_dir": output_dir,
                        "keypose": keypose,
                    },
                ],
            )
        )
    return actions


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
            DeclareLaunchArgument("hold", default_value="false"),
            DeclareLaunchArgument("rviz", default_value="false"),
            DeclareLaunchArgument("keypose", default_value="DIAG_DUAL7_I1"),
            OpaqueFunction(function=_setup),
        ]
    )
