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
    generate_roll_candidates,
    load_stage6_geometry,
    pose_in_planning_frame,
)
from stage4_pregrasp import (  # noqa: E402
    _pose_to_dict,
    compute_stage4_pregrasp_params,
)


_REQUIRED_VIEWS = ("side_pos_y", "side_neg_y", "top_circle")


def _flatten_rolls(geo, roll_step_deg: float, planning_frame: str) -> dict:
    """Pack deterministic roll poses into C++ array parameters (planning frame)."""
    views = []
    degs = []
    indexes = []
    obj = {key: [] for key in ("x", "y", "z", "qx", "qy", "qz", "qw")}
    tcp = {key: [] for key in ("x", "y", "z", "qx", "qy", "qz", "qw")}
    center_err = []
    normal_err = []
    up_err = []
    for name in _REQUIRED_VIEWS:
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
            center_err.append(float(cand.view_center_error_m))
            normal_err.append(float(cand.normal_angle_error_deg))
            up_err.append(float(cand.up_error_deg))
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
        "roll_center_err": center_err,
        "roll_normal_err": normal_err,
        "roll_up_err": up_err,
    }


def _launch_nodes(context, *args, **kwargs):
    """Reuse STEP 5 prefix + STEP 6/10 roll grid. One Lift scene, all views."""
    config_file = LaunchConfiguration("config_file").perform(context)
    roll_step_deg = float(LaunchConfiguration("roll_step_deg").perform(context))
    params = compute_stage4_pregrasp_params(config_file or None)
    geo = load_stage6_geometry(config_file or None)
    params["roll_step_deg"] = roll_step_deg
    params["max_ik_solutions_per_pose"] = int(
        LaunchConfiguration("max_ik_solutions_per_pose").perform(context)
    )
    params["min_ik_solution_distance"] = float(
        LaunchConfiguration("min_ik_solution_distance").perform(context)
    )
    params["output_dir"] = LaunchConfiguration("output_dir").perform(context)
    params["hold_for_introspection"] = (
        LaunchConfiguration("hold_for_introspection").perform(context).lower()
        == "true"
    )
    params["column_name"] = "mounting_column"
    params.update(cpp_inspection_vectors(geo))
    planning_frame = str(params.get("planning_frame", "base_link"))
    for name in _REQUIRED_VIEWS:
        target = geo["targets"][name]
        view = target.view
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
                pose_in_planning_frame(target.object_pose, geo, planning_frame),
                planning_frame,
            )
        )
        params.update(
            _pose_to_dict(
                f"{name}_canonical_tcp",
                pose_in_planning_frame(target.tcp_pose, geo, planning_frame),
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
    pkg_share = get_package_share_directory("fr_task_planner")
    rviz_config = os.path.join(pkg_share, "config", "mtc_smoke.rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")

    preflight = "\n".join(
        [
            "========== STEP 10 PREFLIGHT ==========",
            f"roll_step_deg: {roll_step_deg}",
            f"roll poses: {len(params['roll_views'])}",
            "Geometry source: STEP 6 inspection_view_geometry.generate_roll_candidates",
            "Same Lift scene for all roll / IK candidates: YES",
            "Inspection transit planned? NO",
            "Order search? NO",
            "THIS STEP IS PLAN-ONLY. NO TRAJECTORY EXECUTION IS PERFORMED.",
        ]
    )

    return [
        LogInfo(msg=preflight),
        Node(
            package="fr_task_planner",
            executable="fr3_mtc_roll_candidate_test",
            name="fr3_mtc_roll_candidate_test",
            output="screen",
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.joint_limits,
                moveit_config.planning_pipelines,
                moveit_config.pilz_cartesian_limits,
                {"use_sim_time": use_sim_time},
                params,
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="fr3_mtc_roll_candidate_rviz",
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
    """Plan-only Lift prefix + roll/IK endpoint enumeration. No execute."""
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
            DeclareLaunchArgument("roll_step_deg", default_value="30.0"),
            DeclareLaunchArgument("max_ik_solutions_per_pose", default_value="8"),
            DeclareLaunchArgument("min_ik_solution_distance", default_value="0.1"),
            DeclareLaunchArgument("output_dir", default_value="/tmp"),
            DeclareLaunchArgument("hold_for_introspection", default_value="false"),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
