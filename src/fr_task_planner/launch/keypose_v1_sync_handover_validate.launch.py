import os
import sys
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node
sys.path.insert(0, os.path.dirname(__file__))
from keypose_dual_model import dual_moveit_params

def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg="SYNC HANDOVER collision validator: read-only /get_planning_scene; no motion/gripper clients."),
        Node(package="fr_task_planner", executable="keypose_v1_sync_handover_validate",
             name="keypose_v1_sync_handover_validate", output="screen",
             parameters=[dual_moveit_params(), {"trajectory": os.path.expanduser(
                 "~/fr_task_sync_ws/src/fr_task_planner/config/keypose_optimization_v1/keypose_v1_six_face_with_sync_handover_candidate.yaml")}]),
    ])
