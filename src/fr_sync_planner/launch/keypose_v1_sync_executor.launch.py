"""Deployment entry point for the isolated synchronized handover executor."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context, *args, **kwargs):
    execute = LaunchConfiguration("execute").perform(context).lower() == "true"
    scale = float(LaunchConfiguration("trajectory_speed_scale").perform(context))
    config_dir = os.path.join(
        get_package_share_directory("fr_sync_planner"), "config", "keypose_optimization_v1"
    )
    return [
        LogInfo(msg="FR_SYNC_PLANNER executor: isolated deployment entry; default is dry-run."),
        Node(
            package="fr_sync_planner",
            executable="dual7_sync_task_executor",
            name="keypose_v1_sync_executor",
            output="screen",
            parameters=[{
                "execute": execute,
                "task_mode": "keypose_v1",
                "auto_continue": True,
                "continue_after_home": True,
                "trajectory_speed_scale": scale,
                "grasp_post_close_wait_sec": 1.0,
                "handover_post_close_wait_sec": 0.5,
                "place_post_open_wait_sec": 0.5,
                "auto_handover_release": LaunchConfiguration("auto_handover_release").perform(context).lower() == "true",
                "keypose_v1_trajectory": LaunchConfiguration("trajectory").perform(context),
                "keypose_v1_config_dir": config_dir,
                "real_robot_confirmation": LaunchConfiguration("real_robot_confirmation").perform(context),
                "empty_gripper_confirmation": LaunchConfiguration("empty_gripper_confirmation").perform(context),
            }],
        ),
    ]


def generate_launch_description():
    config_dir = os.path.join(
        get_package_share_directory("fr_sync_planner"), "config", "keypose_optimization_v1"
    )
    return LaunchDescription([
        DeclareLaunchArgument("execute", default_value="false"),
        DeclareLaunchArgument("real_robot_confirmation", default_value=""),
        DeclareLaunchArgument("empty_gripper_confirmation", default_value=""),
        DeclareLaunchArgument("auto_handover_release", default_value="false"),
        DeclareLaunchArgument("trajectory_speed_scale", default_value="0.2"),
        DeclareLaunchArgument(
            "trajectory",
            default_value=os.path.join(config_dir, "keypose_v1_six_face_with_sync_handover_candidate.yaml"),
        ),
        OpaqueFunction(function=setup),
    ])
