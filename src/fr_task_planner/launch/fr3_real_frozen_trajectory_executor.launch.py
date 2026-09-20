"""STEP 13: guarded FAIRINO FR3 frozen-trajectory executor.

Default is DRY RUN. This launch never sets execute:=true.
It does not start real_bringup, Gazebo, or move_group.
"""

from __future__ import annotations

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_yaml(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _discover_controller() -> dict:
    discovered = {
        "controller_name": "fairino3_controller",
        "trajectory_action_name": "/fairino3_controller/follow_joint_trajectory",
        "controller_joint_names": ["j1", "j2", "j3", "j4", "j5", "j6"],
        "source": "fallback",
    }
    try:
        moveit_share = get_package_share_directory("fairino3_v6_moveit2_config")
    except Exception:
        return discovered
    real_path = os.path.join(moveit_share, "config", "moveit_controllers_real.yaml")
    data = _load_yaml(real_path)
    mgr = data.get("moveit_simple_controller_manager") or {}
    names = list(mgr.get("controller_names") or [])
    controller = "fairino3_controller" if "fairino3_controller" in names else (
        names[0] if names else "fairino3_controller"
    )
    block = mgr.get(controller) or {}
    action_ns = str(block.get("action_ns", "follow_joint_trajectory"))
    joints = [str(n) for n in (block.get("joints") or ["j1", "j2", "j3", "j4", "j5", "j6"])]
    discovered.update(
        {
            "controller_name": controller,
            "trajectory_action_name": f"/{controller}/{action_ns}",
            "controller_joint_names": joints,
            "source": real_path,
        }
    )
    return discovered


def _discover_gripper() -> dict:
    discovered = {
        "gripper_service_name": "/fairino_gripper/command",
        "gripper_id": 1,
        "gripper_open_position": 0,
        "gripper_close_position": 85,
        "gripper_velocity": 20,
        "gripper_force": 20,
        "gripper_max_time_ms": 5000,
        "gripper_block": 1,
        "gripper_type": 0,
        "gripper_rot_num": 0.0,
        "gripper_rot_vel": 0,
        "gripper_rot_torque": 0,
        "source": "fallback",
    }
    try:
        share = get_package_share_directory("fr_control")
    except Exception:
        return discovered
    path = os.path.join(share, "config", "gripper.yaml")
    data = _load_yaml(path)
    block = data.get("gripper", data)
    real = dict(block.get("real") or {})
    discovered.update(
        {
            "gripper_service_name": str(real.get("service_name", "/fairino_gripper/command")),
            "gripper_id": int(block.get("id", 1)),
            "gripper_open_position": int(real.get("open_position", 0)),
            "gripper_close_position": int(real.get("close_position", 85)),
            "gripper_velocity": int(real.get("velocity", 20)),
            "gripper_force": int(real.get("force", 20)),
            "gripper_max_time_ms": int(real.get("max_time_ms", 5000)),
            "gripper_block": int(real.get("block", 1)),
            "gripper_type": int(real.get("type", 0)),
            "gripper_rot_num": float(real.get("rot_num", 0.0)),
            "gripper_rot_vel": int(real.get("rot_vel", 0)),
            "gripper_rot_torque": int(real.get("rot_torque", 0)),
            "source": path,
        }
    )
    return discovered


def _launch_nodes(context, *args, **kwargs):
    pkg_share = get_package_share_directory("fr_task_planner")
    default_traj = os.path.expanduser(
        "~/fr_task_ws/src/fr_task_planner/config/"
        "step12c_tilted_camera_winner_trajectory.yaml"
    )
    if not os.path.isfile(default_traj):
        default_traj = os.path.join(
            pkg_share, "config", "step12c_tilted_camera_winner_trajectory.yaml"
        )
    local_cfg = _load_yaml(os.path.join(pkg_share, "config", "fr3_real_executor.yaml"))
    controller = _discover_controller()
    gripper = _discover_gripper()
    touch_links = list(local_cfg.get("touch_links") or [])
    params = {
        "use_sim_time": LaunchConfiguration("use_sim_time").perform(context).lower() == "true",
        "execute": LaunchConfiguration("execute").perform(context).lower() == "true",
        "plan_current_to_home": LaunchConfiguration("plan_current_to_home")
        .perform(context)
        .lower()
        == "true",
        "trajectory_file": LaunchConfiguration("trajectory_file").perform(context),
        "trajectory_speed_scale": float(
            LaunchConfiguration("trajectory_speed_scale").perform(context)
        ),
        "home_tolerance_rad": float(LaunchConfiguration("home_tolerance_rad").perform(context)),
        "segment_start_tolerance_rad": float(
            LaunchConfiguration("segment_start_tolerance_rad").perform(context)
        ),
        "segment_end_tolerance_rad": float(
            LaunchConfiguration("segment_end_tolerance_rad").perform(context)
        ),
        "start_state_tolerance_rad": float(
            LaunchConfiguration("start_state_tolerance_rad").perform(context)
        ),
        "joint_settle_timeout_sec": float(
            LaunchConfiguration("joint_settle_timeout_sec").perform(context)
        ),
        "joint_settle_poll_period_sec": float(
            LaunchConfiguration("joint_settle_poll_period_sec").perform(context)
        ),
        "joint_settle_required_samples": int(
            LaunchConfiguration("joint_settle_required_samples").perform(context)
        ),
        "startup_ready_timeout_sec": float(
            LaunchConfiguration("startup_ready_timeout_sec").perform(context)
        ),
        "gripper_enabled": LaunchConfiguration("gripper_enabled").perform(context).lower()
        == "true",
        "real_robot_confirmation": LaunchConfiguration("real_robot_confirmation").perform(
            context
        ),
        "controller_name": controller["controller_name"],
        "trajectory_action_name": controller["trajectory_action_name"],
        "controller_joint_names": controller["controller_joint_names"],
        "controller_config_source": controller["source"],
        "gripper_service_name": str(
            gripper.get("gripper_service_name", "/fairino_gripper/command")
        ),
        "gripper_config_source": gripper["source"],
        "gripper_id": int(gripper.get("gripper_id", 1)),
        "gripper_open_position": int(gripper.get("gripper_open_position", 0)),
        "gripper_close_position": int(gripper.get("gripper_close_position", 85)),
        "gripper_velocity": int(gripper.get("gripper_velocity", 20)),
        "gripper_force": int(gripper.get("gripper_force", 20)),
        "gripper_max_time_ms": int(gripper.get("gripper_max_time_ms", 5000)),
        "gripper_block": int(gripper.get("gripper_block", 1)),
        "gripper_type": int(gripper.get("gripper_type", 0)),
        "gripper_rot_num": float(gripper.get("gripper_rot_num", 0.0)),
        "gripper_rot_vel": int(gripper.get("gripper_rot_vel", 0)),
        "gripper_rot_torque": int(gripper.get("gripper_rot_torque", 0)),
        "gripper_timeout_sec": float(
            LaunchConfiguration("gripper_timeout_sec").perform(context)
        ),
        "gripper_post_close_wait_sec": float(
            LaunchConfiguration("gripper_post_close_wait_sec").perform(context)
        ),
        "gripper_ping_only": LaunchConfiguration("gripper_ping_only").perform(context).lower()
        == "true",
        "planning_group": str(local_cfg.get("planning_group", "fairino3_v6_group")),
        "attach_link": str(local_cfg.get("attach_link", "gripper_tcp")),
        "object_name": str(local_cfg.get("object_name", "small_part")),
        "table_name": str(local_cfg.get("table_name", "table")),
        "touch_links": touch_links,
        "write_scaled_debug_yaml": True,
        "scaled_debug_yaml": "/tmp/fr3_step13_scaled_trajectory.yaml",
    }
    if not params["trajectory_file"]:
        params["trajectory_file"] = default_traj
    node = Node(
        package="fr_task_planner",
        executable="fr3_real_frozen_trajectory_executor",
        name="fr3_real_frozen_trajectory_executor",
        output="screen",
        parameters=[params],
    )
    return [
        LogInfo(
            msg=[
                "STEP 13 REAL FROZEN TRAJECTORY EXECUTOR. Default is DRY RUN. ",
                "execute=",
                LaunchConfiguration("execute"),
                " action=",
                params["trajectory_action_name"],
                " gripper=",
                params["gripper_service_name"],
            ]
        ),
        node,
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory("fr_task_planner")
    default_traj = os.path.expanduser(
        "~/fr_task_ws/src/fr_task_planner/config/"
        "step12c_tilted_camera_winner_trajectory.yaml"
    )
    if not os.path.isfile(default_traj):
        default_traj = os.path.join(
            pkg_share, "config", "step12c_tilted_camera_winner_trajectory.yaml"
        )
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("execute", default_value="false"),
            DeclareLaunchArgument("plan_current_to_home", default_value="false"),
            DeclareLaunchArgument("trajectory_file", default_value=default_traj),
            DeclareLaunchArgument("trajectory_speed_scale", default_value="0.2"),
            DeclareLaunchArgument("home_tolerance_rad", default_value="0.02"),
            DeclareLaunchArgument("segment_start_tolerance_rad", default_value="0.02"),
            DeclareLaunchArgument("segment_end_tolerance_rad", default_value="0.02"),
            DeclareLaunchArgument("start_state_tolerance_rad", default_value="0.02"),
            DeclareLaunchArgument("joint_settle_timeout_sec", default_value="10.0"),
            DeclareLaunchArgument("joint_settle_poll_period_sec", default_value="0.10"),
            DeclareLaunchArgument("joint_settle_required_samples", default_value="3"),
            DeclareLaunchArgument("startup_ready_timeout_sec", default_value="15.0"),
            DeclareLaunchArgument("gripper_enabled", default_value="true"),
            DeclareLaunchArgument("gripper_timeout_sec", default_value="15.0"),
            DeclareLaunchArgument("gripper_post_close_wait_sec", default_value="0.5"),
            DeclareLaunchArgument("gripper_ping_only", default_value="false"),
            DeclareLaunchArgument("real_robot_confirmation", default_value=""),
            LogInfo(
                msg=[
                    "STEP13 gripper sequencing: Home settle -> open(position=0) -> ",
                    "wait bridge motion done -> Home_to_PreGrasp. ",
                    "Grasp settle -> close -> wait GetGripperMotionDone/ServoJ resume -> ",
                    "Attach -> Lift. Open/close never sent unless execute:=true. ",
                    "Before Home open: confirm fingers/area are clear; opening can drop a held object.",
                ]
            ),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
