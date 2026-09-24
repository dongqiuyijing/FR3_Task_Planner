"""Offline RViz playback for the isolated sync-handover candidate."""
import os
import sys
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from keypose_dual_model import dual_moveit_params

def generate_launch_description():
    root = os.path.expanduser("~/fr_task_sync_ws/src/fr_task_planner")
    return LaunchDescription([
        DeclareLaunchArgument("trajectory", default_value=os.path.join(
            root, "config/keypose_optimization_v1/keypose_v1_six_face_with_sync_handover_candidate.yaml")),
        LogInfo(msg="SYNC-HANDOVER OFFLINE RViz only: no controller or gripper clients."),
        Node(package="fr_task_planner", executable="dual_complete_six_face_rviz_preview.py",
             name="keypose_sync_handover_preview", output="screen",
             arguments=[LaunchConfiguration("trajectory")]),
        Node(package="rviz2", executable="rviz2", name="keypose_sync_handover_rviz", output="screen",
             arguments=["-d", os.path.join(root, "config/dual7_complete_task.rviz")],
             parameters=[dual_moveit_params()]),
    ])
