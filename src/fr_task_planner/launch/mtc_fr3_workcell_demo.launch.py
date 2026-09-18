import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """One-command Stage4 workcell + current planner visualization."""
    pkg_share = get_package_share_directory("fr_task_planner")
    rviz_config = os.path.join(pkg_share, "config", "mtc_workcell_demo.rviz")
    return LaunchDescription(
        [
            SetEnvironmentVariable(name="ROS_DOMAIN_ID", value="77"),
            DeclareLaunchArgument("hold_for_introspection", default_value="true"),
            DeclareLaunchArgument("visualization_hold_seconds", default_value="0.0"),
            DeclareLaunchArgument("headless", default_value="false"),
            LogInfo(
                msg="\n".join(
                    [
                        "========== STEP 11D WORKCELL DEMO ==========",
                        "Starts Stage4 Gazebo + MoveIt + RViz + MTC visualizer.",
                        "ROS_DOMAIN_ID=77. PLAN ONLY. No real robot.",
                    ]
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_share, "launch", "mtc_fr3_sim_visualization.launch.py")
                ),
                launch_arguments={
                    "rviz_config": rviz_config,
                    "start_stage4": "true",
                    "use_mtc_rviz": "true",
                    "hold_for_introspection": LaunchConfiguration("hold_for_introspection"),
                    "visualization_hold_seconds": LaunchConfiguration(
                        "visualization_hold_seconds"
                    ),
                    "visualize_only": "true",
                    "execute_gazebo": "false",
                    "headless": LaunchConfiguration("headless"),
                }.items(),
            ),
        ]
    )
