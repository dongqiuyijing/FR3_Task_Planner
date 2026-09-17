import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    """Start the plan-only FR3 MTC smoke-test node. Does not start MoveIt."""
    moveit_config = (
        MoveItConfigsBuilder(
            "fairino3_v6_robot",
            package_name="fairino3_v6_moveit2_config",
        )
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    pkg_share = get_package_share_directory("fr_task_planner")
    rviz_config = os.path.join(pkg_share, "config", "mtc_smoke.rviz")

    use_sim_time = LaunchConfiguration("use_sim_time")
    use_mtc_rviz = LaunchConfiguration("use_mtc_rviz")

    smoke_params = [
        moveit_config.robot_description,
        moveit_config.robot_description_semantic,
        moveit_config.robot_description_kinematics,
        moveit_config.joint_limits,
        moveit_config.planning_pipelines,
        {"use_sim_time": use_sim_time},
    ]

    smoke_node = Node(
        package="fr_task_planner",
        executable="fr3_mtc_smoke_test",
        name="fr3_mtc_smoke_test",
        output="screen",
        parameters=smoke_params,
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="fr3_mtc_smoke_rviz",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(use_mtc_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Must be true when the existing FR3 Gazebo / stage4 stack is running.",
            ),
            DeclareLaunchArgument(
                "use_mtc_rviz",
                default_value="true",
                description="Open a dedicated RViz with the MTC Motion Planning Tasks panel.",
            ),
            smoke_node,
            rviz_node,
        ]
    )
