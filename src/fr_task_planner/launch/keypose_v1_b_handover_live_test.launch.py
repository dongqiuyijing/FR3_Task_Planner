#!/usr/bin/env python3
"""Independent B_HANDOVER alignment test; plan-only unless explicitly armed."""
import os
import sys
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
sys.path.insert(0, os.path.dirname(__file__))
from keypose_dual_model import dual_moveit_params  # noqa: E402

def setup(context):
    execute = LaunchConfiguration('execute').perform(context).lower() == 'true'
    return [LogInfo(msg='B_HANDOVER LIVE TEST: plan-only by default; no gripper command is automatic.'),
            Node(package='fr_task_planner', executable='keypose_v1_b_handover_live_test',
                 name='keypose_v1_b_handover_live_test', output='screen',
                 parameters=[dual_moveit_params(), {
                   'execute': execute,
                   'real_robot_confirmation': LaunchConfiguration('real_robot_confirmation').perform(context),
                   'final_approach_confirmation': LaunchConfiguration('final_approach_confirmation').perform(context),
                 }])]
def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('execute', default_value='false'),
      DeclareLaunchArgument('real_robot_confirmation', default_value=''),
      DeclareLaunchArgument('final_approach_confirmation', default_value=''), OpaqueFunction(function=setup)])
