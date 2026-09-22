#!/usr/bin/env python3
"""KEYPOSE V1 RViz candidate preview. Display only. Does not command the robot."""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from rcl_interfaces.srv import GetParameters
import rclpy

sys.path.insert(0, os.path.dirname(__file__))
from keypose_dual_model import dual_moveit_params  # noqa: E402


def _try_move_group_descriptions():
    try:
        rclpy.init()
        node = rclpy.create_node("keypose_rviz_param_fetch")
        client = node.create_client(GetParameters, "/move_group/get_parameters")
        if not client.wait_for_service(timeout_sec=2.0):
            node.destroy_node()
            rclpy.shutdown()
            return None
        req = GetParameters.Request()
        req.names = ["robot_description", "robot_description_semantic"]
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=3.0)
        res = fut.result()
        node.destroy_node()
        rclpy.shutdown()
        if res is None or len(res.values) < 2:
            return None
        urdf = res.values[0].string_value
        srdf = res.values[1].string_value
        if not urdf or not srdf:
            return None
        return urdf, srdf
    except Exception:
        if rclpy.ok():
            rclpy.shutdown()
        return None


def _setup(context, *args, **kwargs):
    keypose = LaunchConfiguration("keypose").perform(context)
    candidate = LaunchConfiguration("candidate").perform(context)
    output_dir = LaunchConfiguration("output_dir").perform(context)
    pkg = get_package_share_directory("fr_task_planner")
    rviz_config = os.path.join(pkg, "config", "keypose_candidate.rviz")
    if not os.path.isfile(rviz_config):
        rviz_config = os.path.expanduser(
            "~/fr_task_ws/src/fr_task_planner/config/keypose_candidate.rviz"
        )

    live = _try_move_group_descriptions()
    if live:
        urdf, srdf = live
        robot_params = {
            "robot_description": urdf,
            "robot_description_semantic": srdf,
            "use_sim_time": False,
        }
        src = "move_group overlay"
    else:
        params = dual_moveit_params()
        robot_params = {
            "robot_description": params["robot_description"],
            "robot_description_semantic": params["robot_description_semantic"],
            "use_sim_time": False,
        }
        src = "xacro (no move_group)"

    preview = Node(
        package="fr_task_planner",
        executable="keypose_candidate_rviz_preview.py",
        name="keypose_candidate_rviz_preview",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "output_dir": output_dir,
                "keypose": keypose,
                "candidate": int(candidate),
            }
        ],
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="keypose_candidate_rviz",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[robot_params],
    )
    return [
        LogInfo(
            msg=[
                "KEYPOSE V1 RVIZ PREVIEW ONLY / NOT EXECUTABLE. ",
                f"keypose={keypose} candidate={candidate} model={src}",
            ]
        ),
        preview,
        rviz,
    ]


def generate_launch_description():
    home = os.path.expanduser("~")
    return LaunchDescription(
        [
            DeclareLaunchArgument("keypose", default_value="A_FACE2"),
            DeclareLaunchArgument("candidate", default_value="1"),
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
