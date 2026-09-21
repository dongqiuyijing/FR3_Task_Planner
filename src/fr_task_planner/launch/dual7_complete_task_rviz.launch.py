#!/usr/bin/env python3
"""Open a dedicated DUAL-7 preview RViz. Does not start bringup or send motion."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo, OpaqueFunction
from launch_ros.actions import Node
from rcl_interfaces.srv import GetParameters
import rclpy


def _fetch_descriptions():
    rclpy.init()
    node = rclpy.create_node("dual7_rviz_param_fetch")
    client = node.create_client(GetParameters, "/move_group/get_parameters")
    if not client.wait_for_service(timeout_sec=8.0):
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError("move_group get_parameters unavailable")
    req = GetParameters.Request()
    req.names = ["robot_description", "robot_description_semantic"]
    fut = client.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=8.0)
    res = fut.result()
    node.destroy_node()
    rclpy.shutdown()
    if res is None or len(res.values) < 2:
        raise RuntimeError("failed to copy robot_description from move_group")
    return res.values[0].string_value, res.values[1].string_value


def _setup(context, *args, **kwargs):
    urdf, srdf = _fetch_descriptions()
    pkg = get_package_share_directory("fr_task_planner")
    rviz_config = os.path.join(pkg, "config", "dual7_complete_task.rviz")
    if not os.path.isfile(rviz_config):
        rviz_config = os.path.expanduser(
            "~/fr_task_ws/src/fr_task_planner/config/dual7_complete_task.rviz"
        )
    return [
        LogInfo(msg="DUAL-7 RViz preview window. MODEL PREVIEW ONLY. Not executable."),
        Node(
            package="rviz2",
            executable="rviz2",
            name="dual7_complete_task_rviz",
            output="screen",
            arguments=["-d", rviz_config],
            parameters=[
                {
                    "robot_description": urdf,
                    "robot_description_semantic": srdf,
                    "use_sim_time": False,
                }
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=_setup)])
