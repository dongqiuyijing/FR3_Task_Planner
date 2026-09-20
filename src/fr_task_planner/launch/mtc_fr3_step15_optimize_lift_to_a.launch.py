"""STEP 15: jointly search PreGrasp IK and A IK to reduce Lift→A J1 travel.

Plan / compare / serialize / RViz only.
Does not overwrite STEP12C/12D/13/14 files or STEP13 trajectory_file.
New trajectory is offline review only; not for real execution.
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
    default_winner_in = os.path.expanduser(
        "~/fr_task_ws/src/fr_task_planner/config/step12c_tilted_camera_winner.yaml"
    )
    default_frozen = os.path.expanduser(
        "~/fr_task_ws/src/fr_task_planner/config/step12c_tilted_camera_winner_trajectory.yaml"
    )
    default_winner_out = os.path.expanduser(
        "~/fr_task_ws/src/fr_task_planner/config/step15_optimized_lift_to_a_winner.yaml"
    )
    default_traj_out = os.path.expanduser(
        "~/fr_task_ws/src/fr_task_planner/config/step15_optimized_lift_to_a_trajectory.yaml"
    )
    default_diag = os.path.expanduser(
        "~/fr_task_ws/src/fr_task_planner/config/step15_lift_to_a_diagnostics.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("start_stage4", default_value="true"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("visualize_search", default_value="true"),
            DeclareLaunchArgument("visualization_hold_seconds", default_value="8.0"),
            DeclareLaunchArgument("moveit_delay", default_value="8.0"),
            DeclareLaunchArgument("search_delay", default_value="18.0"),
            DeclareLaunchArgument("winner_input_path", default_value=default_winner_in),
            DeclareLaunchArgument("frozen_trajectory_path", default_value=default_frozen),
            DeclareLaunchArgument("winner_output_path", default_value=default_winner_out),
            DeclareLaunchArgument("trajectory_output_path", default_value=default_traj_out),
            DeclareLaunchArgument("diagnostic_output_path", default_value=default_diag),
            DeclareLaunchArgument("max_pregrasp_ik_candidates", default_value="24"),
            DeclareLaunchArgument("max_prefix_plan_candidates", default_value="12"),
            DeclareLaunchArgument("max_full_task_candidates", default_value="12"),
            DeclareLaunchArgument("full_candidate_plan_attempts", default_value="3"),
            DeclareLaunchArgument("max_ik_attempts", default_value="96"),
            DeclareLaunchArgument("max_a_ik_candidates", default_value="16"),
            DeclareLaunchArgument("max_a_ik_per_prefix", default_value="4"),
            DeclareLaunchArgument("max_lift_to_a_j1_travel_rad", default_value="0.5"),
            DeclareLaunchArgument("step14_search_timeout_sec", default_value="900.0"),
            DeclareLaunchArgument("step15_search_timeout_sec", default_value="900.0"),
            DeclareLaunchArgument("step14_prefix_planning_time", default_value="5.0"),
            DeclareLaunchArgument("step14_full_planning_time", default_value="8.0"),
            LogInfo(
                msg=[
                    "STEP 15 Lift->A J1 search. execution is forced false. ",
                    "Does not overwrite STEP12C/12D/14 YAML or STEP13 defaults. ",
                    "A TCP pose locked; B/C joints locked. ",
                    "winner_output=",
                    LaunchConfiguration("winner_output_path"),
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
                    "visualize_saved_trajectory": "true",
                    "visualization_hold_seconds": LaunchConfiguration(
                        "visualization_hold_seconds"
                    ),
                    "moveit_delay": LaunchConfiguration("moveit_delay"),
                    "search_delay": LaunchConfiguration("search_delay"),
                    "winner_replay_only": "false",
                    "optimize_grasp_prefix": "false",
                    "optimize_lift_to_a": "true",
                    "winner_input_path": LaunchConfiguration("winner_input_path"),
                    "frozen_trajectory_path": LaunchConfiguration("frozen_trajectory_path"),
                    "winner_output_path": LaunchConfiguration("winner_output_path"),
                    "trajectory_output_path": LaunchConfiguration("trajectory_output_path"),
                    "diagnostic_output_path": LaunchConfiguration("diagnostic_output_path"),
                    "max_pregrasp_ik_candidates": LaunchConfiguration(
                        "max_pregrasp_ik_candidates"
                    ),
                    "max_prefix_plan_candidates": LaunchConfiguration(
                        "max_prefix_plan_candidates"
                    ),
                    "max_full_task_candidates": LaunchConfiguration("max_full_task_candidates"),
                    "full_candidate_plan_attempts": LaunchConfiguration(
                        "full_candidate_plan_attempts"
                    ),
                    "max_ik_attempts": LaunchConfiguration("max_ik_attempts"),
                    "max_a_ik_candidates": LaunchConfiguration("max_a_ik_candidates"),
                    "max_a_ik_per_prefix": LaunchConfiguration("max_a_ik_per_prefix"),
                    "max_lift_to_a_j1_travel_rad": LaunchConfiguration(
                        "max_lift_to_a_j1_travel_rad"
                    ),
                    "step14_search_timeout_sec": LaunchConfiguration(
                        "step14_search_timeout_sec"
                    ),
                    "step15_search_timeout_sec": LaunchConfiguration(
                        "step15_search_timeout_sec"
                    ),
                    "step14_prefix_planning_time": LaunchConfiguration(
                        "step14_prefix_planning_time"
                    ),
                    "step14_full_planning_time": LaunchConfiguration(
                        "step14_full_planning_time"
                    ),
                }.items(),
            ),
        ]
    )
