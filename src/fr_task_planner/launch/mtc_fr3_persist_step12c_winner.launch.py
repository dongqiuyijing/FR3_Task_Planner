"""STEP 12D: persist the fixed STEP12C winner JointTrajectory.

Plan / serialize / read-back / RViz only.
No Gazebo execute. No real robot. No execute=true path.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory("fr_task_planner")
    search_launch = os.path.join(pkg_share, "launch", "mtc_fr3_complete_abc_search.launch.py")
    default_winner = os.path.expanduser(
        "~/fr_task_ws/src/fr_task_planner/config/step12c_tilted_camera_winner.yaml"
    )
    default_traj = os.path.expanduser(
        "~/fr_task_ws/src/fr_task_planner/config/step12c_tilted_camera_winner_trajectory.yaml"
    )
    default_diag = os.path.expanduser(
        "~/fr_task_ws/src/fr_task_planner/config/step12d_trajectory_persistence_diagnostics.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("start_stage4", default_value="true"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("visualize_search", default_value="true"),
            DeclareLaunchArgument("visualize_saved_trajectory", default_value="true"),
            DeclareLaunchArgument("visualization_hold_seconds", default_value="20.0"),
            DeclareLaunchArgument("moveit_delay", default_value="8.0"),
            DeclareLaunchArgument("search_delay", default_value="18.0"),
            DeclareLaunchArgument("winner_input_path", default_value=default_winner),
            DeclareLaunchArgument("trajectory_output_path", default_value=default_traj),
            DeclareLaunchArgument("diagnostic_output_path", default_value=default_diag),
            DeclareLaunchArgument("full_plan_retries", default_value="5"),
            DeclareLaunchArgument("planning_time", default_value="10.0"),
            LogInfo(
                msg=[
                    "STEP 12D persist-only launch. execution is forced false. ",
                    "winner_input=",
                    LaunchConfiguration("winner_input_path"),
                    " trajectory_output=",
                    LaunchConfiguration("trajectory_output_path"),
                ]
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(search_launch),
                launch_arguments={
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "start_stage4": LaunchConfiguration("start_stage4"),
                    "headless": LaunchConfiguration("headless"),
                    "visualize_search": LaunchConfiguration("visualize_search"),
                    "visualize_saved_trajectory": LaunchConfiguration(
                        "visualize_saved_trajectory"
                    ),
                    "visualization_hold_seconds": LaunchConfiguration(
                        "visualization_hold_seconds"
                    ),
                    "moveit_delay": LaunchConfiguration("moveit_delay"),
                    "search_delay": LaunchConfiguration("search_delay"),
                    "winner_replay_only": "true",
                    "winner_input_path": LaunchConfiguration("winner_input_path"),
                    "trajectory_output_path": LaunchConfiguration("trajectory_output_path"),
                    "diagnostic_output_path": LaunchConfiguration("diagnostic_output_path"),
                    "full_plan_retries": LaunchConfiguration("full_plan_retries"),
                }.items(),
            ),
        ]
    )
