import os
import sys

import yaml
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
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

_LAUNCH_DIR = os.path.dirname(__file__)
if _LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _LAUNCH_DIR)

from inspection_view_geometry import (  # noqa: E402
    cpp_inspection_vectors,
    generate_roll_candidates,
    load_stage6_geometry,
    pose_in_planning_frame,
    retarget_inspection_p1,
)
from stage4_pregrasp import _pose_to_dict, compute_stage4_pregrasp_params  # noqa: E402


_ROS_DOMAIN_ID = "77"
_VIEWS = ("side_pos_y", "side_neg_y", "top_circle")
# User-fixed for STEP 12. Do not search. Do not write stage4_config.yaml.
_FIXED_P1_Z = 1.20
_CAMERA_DESIGN = (0.0, 0.10, 1.20)


def _flatten_rolls(geo, roll_step_deg: float, planning_frame: str) -> dict:
    views = []
    degs = []
    indexes = []
    obj = {key: [] for key in ("x", "y", "z", "qx", "qy", "qz", "qw")}
    tcp = {key: [] for key in ("x", "y", "z", "qx", "qy", "qz", "qw")}
    for name in _VIEWS:
        for cand in generate_roll_candidates(
            geo["targets"][name], geo["tcp_t_object"], roll_step_deg
        ):
            object_pose = pose_in_planning_frame(cand.object_pose, geo, planning_frame)
            tcp_pose = pose_in_planning_frame(cand.tcp_pose, geo, planning_frame)
            views.append(name)
            degs.append(float(cand.roll_deg))
            indexes.append(float(cand.pose_index))
            obj["x"].append(float(object_pose.position.x))
            obj["y"].append(float(object_pose.position.y))
            obj["z"].append(float(object_pose.position.z))
            obj["qx"].append(float(object_pose.orientation.x))
            obj["qy"].append(float(object_pose.orientation.y))
            obj["qz"].append(float(object_pose.orientation.z))
            obj["qw"].append(float(object_pose.orientation.w))
            tcp["x"].append(float(tcp_pose.position.x))
            tcp["y"].append(float(tcp_pose.position.y))
            tcp["z"].append(float(tcp_pose.position.z))
            tcp["qx"].append(float(tcp_pose.orientation.x))
            tcp["qy"].append(float(tcp_pose.orientation.y))
            tcp["qz"].append(float(tcp_pose.orientation.z))
            tcp["qw"].append(float(tcp_pose.orientation.w))
    return {
        "roll_views": views,
        "roll_degs": degs,
        "roll_pose_index": indexes,
        "roll_obj_x": obj["x"],
        "roll_obj_y": obj["y"],
        "roll_obj_z": obj["z"],
        "roll_obj_qx": obj["qx"],
        "roll_obj_qy": obj["qy"],
        "roll_obj_qz": obj["qz"],
        "roll_obj_qw": obj["qw"],
        "roll_tcp_x": tcp["x"],
        "roll_tcp_y": tcp["y"],
        "roll_tcp_z": tcp["z"],
        "roll_tcp_qx": tcp["qx"],
        "roll_tcp_qy": tcp["qy"],
        "roll_tcp_qz": tcp["qz"],
        "roll_tcp_qw": tcp["qw"],
    }


def _env_bounds(config_file: str) -> dict:
    with open(config_file, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    table = cfg.get("table", {})
    column = cfg.get("column", {})
    tp = table.get("initial_pose", {}).get("position", {})
    td = table.get("dimensions", {})
    cp = column.get("initial_pose", {}).get("position", {})
    cd = column.get("dimensions", {})
    return {
        "column_name": str(column.get("name", "mounting_column")),
        "table_cx": float(tp.get("x", 0.0)),
        "table_cy": float(tp.get("y", 0.0)),
        "table_cz": float(tp.get("z", 0.0)),
        "table_dx": float(td.get("x", 0.0)),
        "table_dy": float(td.get("y", 0.0)),
        "table_dz": float(td.get("z", 0.0)),
        "column_cx": float(cp.get("x", 0.0)),
        "column_cy": float(cp.get("y", 0.0)),
        "column_cz": float(cp.get("z", 0.0)),
        "column_dx": float(cd.get("x", 0.0)),
        "column_dy": float(cd.get("y", 0.0)),
        "column_dz": float(cd.get("z", 0.0)),
    }


def _launch_nodes(context, *args, **kwargs):
    config_file = LaunchConfiguration("config_file").perform(context)
    roll_step_deg = float(LaunchConfiguration("roll_step_deg").perform(context))
    params = compute_stage4_pregrasp_params(config_file or None)
    geo = retarget_inspection_p1(
        load_stage6_geometry(config_file or None),
        (0.0, 0.4, _FIXED_P1_Z),
    )
    params["roll_step_deg"] = roll_step_deg
    params["max_ik_solutions_per_pose"] = int(
        LaunchConfiguration("max_ik_solutions_per_pose").perform(context)
    )
    params["min_ik_solution_distance"] = float(
        LaunchConfiguration("min_ik_solution_distance").perform(context)
    )
    params["top_k"] = int(LaunchConfiguration("top_k").perform(context))
    params["beam_width"] = int(LaunchConfiguration("beam_width").perform(context))
    params["edge_attempts"] = int(LaunchConfiguration("edge_attempts").perform(context))
    params["prefix_retries"] = int(LaunchConfiguration("prefix_retries").perform(context))
    params["complete_candidate_budget"] = int(
        LaunchConfiguration("complete_candidate_budget").perform(context)
    )
    params["search_planning_time"] = float(
        LaunchConfiguration("search_planning_time").perform(context)
    )
    params["visualize_search"] = (
        LaunchConfiguration("visualize_search").perform(context).lower() == "true"
    )
    params["visualization_hold_seconds"] = float(
        LaunchConfiguration("visualization_hold_seconds").perform(context)
    )
    params["diagnostic_output_path"] = LaunchConfiguration("diagnostic_output_path").perform(
        context
    )
    params["winner_output_path"] = LaunchConfiguration("winner_output_path").perform(context)
    params["camera_design_x"] = _CAMERA_DESIGN[0]
    params["camera_design_y"] = _CAMERA_DESIGN[1]
    params["camera_design_z"] = _CAMERA_DESIGN[2]
    params.update(_env_bounds(config_file))
    params.update(cpp_inspection_vectors(geo))
    planning_frame = str(params.get("planning_frame", "base_link"))
    for name in _VIEWS:
        view = geo["targets"][name].view
        params[f"{name}_center_x"] = float(view.center_in_object[0])
        params[f"{name}_center_y"] = float(view.center_in_object[1])
        params[f"{name}_center_z"] = float(view.center_in_object[2])
        params[f"{name}_normal_x"] = float(view.normal_in_object[0])
        params[f"{name}_normal_y"] = float(view.normal_in_object[1])
        params[f"{name}_normal_z"] = float(view.normal_in_object[2])
        params[f"{name}_up_x"] = float(view.up_in_object[0])
        params[f"{name}_up_y"] = float(view.up_in_object[1])
        params[f"{name}_up_z"] = float(view.up_in_object[2])
        params.update(
            _pose_to_dict(
                f"{name}_canonical_object",
                pose_in_planning_frame(geo["targets"][name].object_pose, geo, planning_frame),
                planning_frame,
            )
        )
    params.update(_flatten_rolls(geo, roll_step_deg, planning_frame))

    moveit_config = (
        MoveItConfigsBuilder(
            "fairino3_v6_robot",
            package_name="fairino3_v6_moveit2_config",
        )
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )
    delay = float(LaunchConfiguration("search_delay").perform(context))
    start_stage4 = LaunchConfiguration("start_stage4").perform(context).lower() == "true"
    visualize = LaunchConfiguration("visualize_search").perform(context).lower() == "true"
    if start_stage4 and delay < 1.0:
        delay = 18.0
    pkg_share = get_package_share_directory("fr_task_planner")
    rviz_config = os.path.join(pkg_share, "config", "mtc_workcell_demo.rviz")
    search = Node(
        package="fr_task_planner",
        executable="fr3_mtc_complete_abc_search",
        name="fr3_mtc_complete_abc_search",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            moveit_config.planning_pipelines,
            moveit_config.pilz_cartesian_limits,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            params,
        ],
    )
    delayed = [search]
    if delay > 0.0:
        delayed = [TimerAction(period=delay, actions=delayed)]
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="fr3_step12_search_rviz",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        condition=IfCondition(LaunchConfiguration("visualize_search")),
    )
    return [
        LogInfo(
            msg="\n".join(
                [
                    "========== STEP 12 FIXED ABC COMPLETE-TASK SEARCH ==========",
                    f"ROS_DOMAIN_ID={_ROS_DOMAIN_ID}",
                    "PLAN / RViz ONLY. No Gazebo execute. No real robot.",
                    "P1 fixed: [0.0, 0.4, 1.20]  (YAML z not rewritten)",
                    "camera design: [0.0, 0.10, 1.20] forward +Y  (NOT calibrated)",
                    "order: A → B → C",
                    f"roll step: {roll_step_deg} deg",
                    f"visualize_search: {visualize}",
                ]
            )
        ),
        rviz_node,
        *delayed,
    ]


def generate_launch_description():
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
    control_share = get_package_share_directory("fr_control")
    default_winner = os.path.expanduser(
        "~/fr_task_ws/src/fr_task_planner/config/step12_best_sampled_complete_task.yaml"
    )
    return LaunchDescription(
        [
            SetEnvironmentVariable(name="ROS_DOMAIN_ID", value=_ROS_DOMAIN_ID),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("start_stage4", default_value="true"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("visualize_search", default_value="true"),
            DeclareLaunchArgument("visualization_hold_seconds", default_value="20.0"),
            DeclareLaunchArgument("moveit_delay", default_value="8.0"),
            DeclareLaunchArgument("search_delay", default_value="18.0"),
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("roll_step_deg", default_value="5.0"),
            DeclareLaunchArgument("max_ik_solutions_per_pose", default_value="8"),
            DeclareLaunchArgument("min_ik_solution_distance", default_value="0.1"),
            DeclareLaunchArgument("top_k", default_value="10"),
            DeclareLaunchArgument("beam_width", default_value="5"),
            DeclareLaunchArgument("edge_attempts", default_value="3"),
            DeclareLaunchArgument("prefix_retries", default_value="5"),
            DeclareLaunchArgument("complete_candidate_budget", default_value="40"),
            DeclareLaunchArgument("search_planning_time", default_value="3.0"),
            DeclareLaunchArgument(
                "diagnostic_output_path",
                default_value="/tmp/fr3_step12_search.yaml",
            ),
            DeclareLaunchArgument("winner_output_path", default_value=default_winner),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(control_share, "launch", "stage4_full.launch.py")
                ),
                launch_arguments={
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "use_rviz": "false",
                    "headless": LaunchConfiguration("headless"),
                    "config_file": LaunchConfiguration("config_file"),
                    "moveit_delay": LaunchConfiguration("moveit_delay"),
                }.items(),
                condition=IfCondition(LaunchConfiguration("start_stage4")),
            ),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
