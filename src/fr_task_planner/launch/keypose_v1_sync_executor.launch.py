"""Isolated dual-handover executor; never used by keypose_v1_executor.launch.py."""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def setup(context, *args, **kwargs):
    execute = LaunchConfiguration("execute").perform(context).lower() == "true"
    scale = float(LaunchConfiguration("trajectory_speed_scale").perform(context))
    return [
        LogInfo(msg="SYNC-HANDOVER executor: isolated development entry; default is dry-run."),
        Node(package="fr_task_planner", executable="dual7_sync_task_executor",
             name="keypose_v1_sync_executor", output="screen", parameters=[{
                 "execute": execute, "task_mode": "keypose_v1", "auto_continue": True,
                 "continue_after_home": True, "trajectory_speed_scale": scale,
                "grasp_post_close_wait_sec": 1.0, "handover_post_close_wait_sec": 0.5,
                "place_post_open_wait_sec": 0.5,
                 "auto_handover_release": LaunchConfiguration("auto_handover_release").perform(context).lower() == "true",
                 "keypose_v1_trajectory": LaunchConfiguration("trajectory").perform(context),
                 "real_robot_confirmation": LaunchConfiguration("real_robot_confirmation").perform(context),
                 "empty_gripper_confirmation": LaunchConfiguration("empty_gripper_confirmation").perform(context),
             }])]

def generate_launch_description():
    root = os.path.expanduser("~/fr_task_sync_ws/src/fr_task_planner/config/keypose_optimization_v1")
    return LaunchDescription([
        DeclareLaunchArgument("execute", default_value="false"),
        DeclareLaunchArgument("real_robot_confirmation", default_value=""),
        DeclareLaunchArgument("empty_gripper_confirmation", default_value=""),
        DeclareLaunchArgument("auto_handover_release", default_value="false"),
        DeclareLaunchArgument("trajectory_speed_scale", default_value="0.2"),
        DeclareLaunchArgument("trajectory", default_value=os.path.join(root, "keypose_v1_six_face_with_sync_handover_candidate.yaml")),
        OpaqueFunction(function=setup),
    ])
