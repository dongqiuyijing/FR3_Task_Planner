import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

_LAUNCH_DIR = os.path.dirname(__file__)
if _LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _LAUNCH_DIR)

from stage4_pregrasp import compute_stage4_pregrasp_params, format_step4_preflight  # noqa: E402


def _launch_nodes(context, *args, **kwargs):
    """Reuse Stage4 YAML geometry and start the plan-only grasp-chain node."""
    config_file = LaunchConfiguration("config_file").perform(context)
    params = compute_stage4_pregrasp_params(config_file or None)
    params["max_solutions"] = 5

    moveit_config = (
        MoveItConfigsBuilder(
            "fairino3_v6_robot",
            package_name="fairino3_v6_moveit2_config",
        )
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )
    pkg_share = get_package_share_directory("fr_task_planner")
    rviz_config = os.path.join(pkg_share, "config", "mtc_smoke.rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")

    node_params = [
        moveit_config.robot_description,
        moveit_config.robot_description_semantic,
        moveit_config.robot_description_kinematics,
        moveit_config.joint_limits,
        moveit_config.planning_pipelines,
        moveit_config.pilz_cartesian_limits,
        {"use_sim_time": use_sim_time},
        params,
    ]

    return [
        LogInfo(msg=format_step4_preflight(params)),
        Node(
            package="fr_task_planner",
            executable="fr3_mtc_grasp_chain_test",
            name="fr3_mtc_grasp_chain_test",
            output="screen",
            parameters=node_params,
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="fr3_mtc_grasp_chain_rviz",
            output="log",
            arguments=["-d", rviz_config],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.planning_pipelines,
                {"use_sim_time": use_sim_time},
            ],
            condition=IfCondition(LaunchConfiguration("use_mtc_rviz")),
        ),
    ]


def generate_launch_description():
    """Plan-only Home → PreGrasp → Grasp. Does not start Stage4 or execute."""
    default_config = os.path.join(
        get_package_share_directory("fr_control"),
        "config",
        "stage4_config.yaml",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Must be true for stage4_full / Gazebo.",
            ),
            DeclareLaunchArgument(
                "use_mtc_rviz",
                default_value="true",
                description="Open a dedicated RViz with the MTC panel.",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Stage4 YAML. Read-only reuse of fr_control config.",
            ),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
