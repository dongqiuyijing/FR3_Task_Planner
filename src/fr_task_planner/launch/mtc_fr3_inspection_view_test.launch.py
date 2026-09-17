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

from inspection_view_geometry import (  # noqa: E402
    cpp_inspection_vectors,
    load_stage6_geometry,
    pose_in_planning_frame,
)
from stage4_pregrasp import (  # noqa: E402
    _pose_to_dict,
    compute_stage4_pregrasp_params,
)


_REQUIRED_VIEWS = ("side_pos_y", "side_neg_y", "top_circle")


def _launch_nodes(context, *args, **kwargs):
    """Reuse STEP 5 prefix + STEP 6 view target. One view per launch."""
    config_file = LaunchConfiguration("config_file").perform(context)
    view_name = LaunchConfiguration("view_name").perform(context)
    if view_name not in _REQUIRED_VIEWS:
        raise RuntimeError(
            f"view_name must be one of {_REQUIRED_VIEWS}, got {view_name}"
        )

    params = compute_stage4_pregrasp_params(config_file or None)
    geo = load_stage6_geometry(config_file or None)
    target = geo["targets"][view_name]
    view = target.view
    params["max_solutions"] = 5
    params["view_name"] = view_name
    params.update(cpp_inspection_vectors(geo))
    planning_frame = str(params.get("planning_frame", "base_link"))
    inspect_tcp = pose_in_planning_frame(target.tcp_pose, geo, planning_frame)
    inspect_object = pose_in_planning_frame(target.object_pose, geo, planning_frame)
    params["center_in_object_x"] = float(view.center_in_object[0])
    params["center_in_object_y"] = float(view.center_in_object[1])
    params["center_in_object_z"] = float(view.center_in_object[2])
    params["normal_in_object_x"] = float(view.normal_in_object[0])
    params["normal_in_object_y"] = float(view.normal_in_object[1])
    params["normal_in_object_z"] = float(view.normal_in_object[2])
    params["up_in_object_x"] = float(view.up_in_object[0])
    params["up_in_object_y"] = float(view.up_in_object[1])
    params["up_in_object_z"] = float(view.up_in_object[2])
    params["max_up_angle_error_deg"] = 5.0
    params["view_center_tolerance"] = 0.005
    params.update(_pose_to_dict("inspect", inspect_tcp, planning_frame))
    params.update(_pose_to_dict("inspect_object", inspect_object, planning_frame))

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
            "========== STEP 7 PREFLIGHT ==========",
            f"View name: {view_name}",
            "required_views = {side_pos_y, side_neg_y, top_circle}",
            "Fixed order encoded? NO",
            f"Canonical TCP world: frame={target.frame} "
            f"xyz=({target.tcp_pose.position.x:.6f}, {target.tcp_pose.position.y:.6f}, "
            f"{target.tcp_pose.position.z:.6f})",
            f"Canonical TCP planning: frame={planning_frame} "
            f"xyz=({inspect_tcp.position.x:.6f}, {inspect_tcp.position.y:.6f}, "
            f"{inspect_tcp.position.z:.6f})",
            f"P1 world: ({geo['p1_world'][0]:.6f}, {geo['p1_world'][1]:.6f}, {geo['p1_world'][2]:.6f})",
            f"D1 world: ({geo['d1_world'][0]:.6f}, {geo['d1_world'][1]:.6f}, {geo['d1_world'][2]:.6f})",
            "Geometry source: STEP 6 inspection_view_geometry",
            "Lift → View planner: OMPL",
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
            executable="fr3_mtc_inspection_view_test",
            name="fr3_mtc_inspection_view_test",
            output="screen",
            parameters=node_params,
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="fr3_mtc_inspection_view_rviz",
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
    """Plan-only full grasp chain + one inspection view. No execute."""
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
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument(
                "view_name",
                default_value="side_pos_y",
                description="One of side_pos_y, side_neg_y, top_circle.",
            ),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
