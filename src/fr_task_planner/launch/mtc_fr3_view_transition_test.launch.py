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

from inspection_view_geometry import load_stage6_geometry  # noqa: E402
from stage4_pregrasp import (  # noqa: E402
    _pose_to_dict,
    compute_stage4_pregrasp_params,
)


_REQUIRED_VIEWS = ("side_pos_y", "side_neg_y", "top_circle")


def _flatten_view(prefix: str, target) -> dict:
    """STEP 6 target → C++ ViewSpec parameters. No hardcoded pose."""
    view = target.view
    params = {
        f"{prefix}_view": view.name,
        f"{prefix}_center_in_object_x": float(view.center_in_object[0]),
        f"{prefix}_center_in_object_y": float(view.center_in_object[1]),
        f"{prefix}_center_in_object_z": float(view.center_in_object[2]),
        f"{prefix}_normal_in_object_x": float(view.normal_in_object[0]),
        f"{prefix}_normal_in_object_y": float(view.normal_in_object[1]),
        f"{prefix}_normal_in_object_z": float(view.normal_in_object[2]),
        f"{prefix}_up_in_object_x": float(view.up_in_object[0]),
        f"{prefix}_up_in_object_y": float(view.up_in_object[1]),
        f"{prefix}_up_in_object_z": float(view.up_in_object[2]),
    }
    params.update(_pose_to_dict(prefix, target.tcp_pose, target.frame))
    params.update(_pose_to_dict(f"{prefix}_object", target.object_pose, target.frame))
    return params


def _launch_nodes(context, *args, **kwargs):
    """One complete grasp prefix + source View + target View. Plan-only."""
    config_file = LaunchConfiguration("config_file").perform(context)
    source_view = LaunchConfiguration("source_view").perform(context)
    target_view = LaunchConfiguration("target_view").perform(context)
    if source_view not in _REQUIRED_VIEWS or target_view not in _REQUIRED_VIEWS:
        raise RuntimeError(
            f"source_view/target_view must be in {_REQUIRED_VIEWS}, "
            f"got {source_view} → {target_view}"
        )
    if source_view == target_view:
        raise RuntimeError("ABORT: source_view == target_view; no self-edge")

    params = compute_stage4_pregrasp_params(config_file or None)
    geo = load_stage6_geometry(config_file or None)
    source = geo["targets"][source_view]
    target = geo["targets"][target_view]
    params["max_solutions"] = 5
    params["hold_for_introspection"] = (
        LaunchConfiguration("hold_for_introspection").perform(context).lower() == "true"
    )
    params["p1_x"] = float(geo["p1"][0])
    params["p1_y"] = float(geo["p1"][1])
    params["p1_z"] = float(geo["p1"][2])
    params["d1_x"] = float(geo["direction"][0])
    params["d1_y"] = float(geo["direction"][1])
    params["d1_z"] = float(geo["direction"][2])
    params["preferred_up_x"] = float(geo["up"][0])
    params["preferred_up_y"] = float(geo["up"][1])
    params["preferred_up_z"] = float(geo["up"][2])
    params["max_up_angle_error_deg"] = 5.0
    params["view_center_tolerance"] = 0.005
    params.update(_flatten_view("source", source))
    params.update(_flatten_view("target", target))

    from fr_control.stage4_config import load_yaml

    cfg = load_yaml(geo["config_file"])
    validation = cfg.get("validation", {})
    if "max_up_angle_error_deg" in validation:
        params["max_up_angle_error_deg"] = float(validation["max_up_angle_error_deg"])

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

    preflight = "\n".join(
        [
            "========== STEP 8 PREFLIGHT ==========",
            f"Source view: {source_view}",
            f"Target view: {target_view}",
            "required_views = {side_pos_y, side_neg_y, top_circle}",
            "Self-edge? NO",
            "Order search? NO",
            "Roll sampling? NO",
            "Geometry source: STEP 6 inspection_view_geometry",
            "Source → Target planner: OMPL",
            "Source/target are stop-required inspection endpoints.",
            "Hold time is not executed in plan-only STEP 8.",
            "THIS STEP IS PLAN-ONLY. NO TRAJECTORY EXECUTION IS PERFORMED.",
        ]
    )

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
        LogInfo(msg=preflight),
        Node(
            package="fr_task_planner",
            executable="fr3_mtc_view_transition_test",
            name="fr3_mtc_view_transition_test",
            output="screen",
            parameters=node_params,
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="fr3_mtc_view_transition_rviz",
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
    """Plan-only full grasp chain + one directed inspection transition."""
    source_config = os.path.expanduser(
        "~/fairino_ws/src/fr_control/config/stage4_config.yaml"
    )
    default_config = (
        source_config
        if os.path.isfile(source_config)
        else os.path.join(
            get_package_share_directory("fr_control"),
            "config",
            "stage4_config.yaml",
        )
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("use_mtc_rviz", default_value="false"),
            DeclareLaunchArgument("hold_for_introspection", default_value="false"),
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument(
                "source_view",
                default_value="side_pos_y",
                description="One of side_pos_y, side_neg_y, top_circle.",
            ),
            DeclareLaunchArgument(
                "target_view",
                default_value="side_neg_y",
                description="One of side_pos_y, side_neg_y, top_circle. Must differ.",
            ),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
