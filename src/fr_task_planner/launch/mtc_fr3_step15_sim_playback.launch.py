"""STEP15 Gazebo playback of the already-saved frozen trajectory.

Starts Stage4 (Gazebo GUI + MoveIt). Does not search, replan, or call the
real gripper. Does not change STEP13 default trajectory_file.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


_ROS_DOMAIN_ID = "77"


def _playback(context, *args, **kwargs):
    delay = float(LaunchConfiguration("playback_delay").perform(context))
    if delay < 0.0:
        delay = 0.0
    node = Node(
        package="fr_task_planner",
        executable="step15_sim_playback.py",
        name="step15_sim_playback",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "trajectory_file": LaunchConfiguration("trajectory_file").perform(context),
                "speed_scale": float(LaunchConfiguration("speed_scale").perform(context)),
                "controller_wait_sec": float(
                    LaunchConfiguration("controller_wait_sec").perform(context)
                ),
            }
        ],
    )
    if delay > 0.0:
        return [TimerAction(period=delay, actions=[node])]
    return [node]


def generate_launch_description():
    control_share = get_package_share_directory("fr_control")
    default_config = os.path.expanduser(
        "~/fairino_ws/src/fr_control/config/stage4_config.yaml"
    )
    default_traj = os.path.expanduser(
        "~/fr_task_ws/src/fr_task_planner/config/"
        "step15_optimized_lift_to_a_trajectory.yaml"
    )
    return LaunchDescription(
        [
            SetEnvironmentVariable(name="ROS_DOMAIN_ID", value=_ROS_DOMAIN_ID),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("moveit_delay", default_value="8.0"),
            DeclareLaunchArgument("playback_delay", default_value="16.0"),
            DeclareLaunchArgument("speed_scale", default_value="1.0"),
            DeclareLaunchArgument("controller_wait_sec", default_value="60.0"),
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("trajectory_file", default_value=default_traj),
            LogInfo(
                msg=[
                    "STEP15 Gazebo playback. Saved trajectory only. ",
                    "No IK/search. No real robot. No real gripper. ",
                    "trajectory=",
                    LaunchConfiguration("trajectory_file"),
                ]
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(control_share, "launch", "stage4_full.launch.py")
                ),
                launch_arguments={
                    "use_sim_time": "true",
                    "use_rviz": LaunchConfiguration("use_rviz"),
                    "headless": LaunchConfiguration("headless"),
                    "config_file": LaunchConfiguration("config_file"),
                    "moveit_delay": LaunchConfiguration("moveit_delay"),
                }.items(),
            ),
            OpaqueFunction(function=_playback),
        ]
    )
