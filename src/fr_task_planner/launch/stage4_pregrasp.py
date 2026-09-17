"""Thin adapter: reuse fr_control Stage4 geometry, do not reimplement it."""

from __future__ import annotations

import math
from typing import Any

from fr_control.constants import (
    ARM_JOINTS,
    ATTACH_LINK,
    BASE_FRAME,
    EE_LINK,
    GRIPPER_TOUCH_LINKS,
    PLANNING_GROUP,
)
from fr_control.grasp_poses import (
    compute_grasp_poses,
    pose_in_frame,
    pose_inverse,
    pose_multiply,
)
from fr_control.stage4_config import (
    block_pose,
    grasp_tcp_xyzw,
    joint_positions,
    load_yaml,
    robot_base_pose,
    workcell_config_path,
)


def _pose_to_dict(prefix: str, pose, frame: str) -> dict[str, Any]:
    """Flatten a geometry_msgs Pose into launch/C++ parameters."""
    return {
        f"{prefix}_frame": frame,
        f"{prefix}_x": float(pose.position.x),
        f"{prefix}_y": float(pose.position.y),
        f"{prefix}_z": float(pose.position.z),
        f"{prefix}_qx": float(pose.orientation.x),
        f"{prefix}_qy": float(pose.orientation.y),
        f"{prefix}_qz": float(pose.orientation.z),
        f"{prefix}_qw": float(pose.orientation.w),
    }


def _to_base_link(pose, t_world_base):
    """Express a world pose in base_link using YAML T_world_base."""
    return pose_multiply(pose_inverse(t_world_base), pose)


def compute_stage4_pregrasp_params(config_file: str | None = None) -> dict[str, Any]:
    """Compute Stage4 PreGrasp with the existing Python authority path."""
    path = config_file or workcell_config_path()
    cfg = load_yaml(path)
    world_frame = str(cfg.get("frames", {}).get("world", "world"))
    planning_frame = str(cfg.get("planning_frame", BASE_FRAME))
    if planning_frame != BASE_FRAME:
        raise RuntimeError(f"STEP 3 requires planning_frame=base_link, got {planning_frame}")

    grasp_cfg = cfg["grasp"]
    object_cfg = cfg["object"]
    motion_cfg = cfg["motion"]
    t_world_base = robot_base_pose(cfg)

    object_source = str(object_cfg["initial_pose"].get("frame", world_frame))
    object_in_source = block_pose(object_cfg["initial_pose"])
    if object_source == world_frame:
        object_world = object_in_source
    elif object_source == planning_frame:
        object_world = pose_multiply(t_world_base, object_in_source)
    else:
        raise RuntimeError(f"Unsupported object frame: {object_source}")

    pregrasp_distance = float(grasp_cfg["pregrasp_distance"])
    poses_world = compute_grasp_poses(
        object_world,
        approach_xyzw=grasp_tcp_xyzw(grasp_cfg),
        pre_grasp_offset=pregrasp_distance,
        lift_height=float(grasp_cfg["lift_distance"]),
        table_top_z=float(grasp_cfg["table_top_z"]),
        fingertip_from_tcp=float(grasp_cfg["fingertip_from_tcp"]),
        table_clearance=float(grasp_cfg["table_clearance"]),
        planning_frame=world_frame,
    )

    object_base = _to_base_link(poses_world.object_pose, t_world_base)
    grasp_base = _to_base_link(poses_world.grasp_pose, t_world_base)
    pregrasp_base = _to_base_link(poses_world.pre_grasp_pose, t_world_base)
    lift_base = _to_base_link(poses_world.lift_pose, t_world_base)
    tcp_object = pose_in_frame(poses_world.grasp_pose, poses_world.object_pose)
    touch_links = [
        str(name) for name in cfg.get("gripper_touch_links", GRIPPER_TOUCH_LINKS)
    ]
    table_name = str(cfg.get("table", {}).get("name", "table"))
    lift_distance = float(grasp_cfg["lift_distance"])

    # YAML robot.initial_joint_positions is degree. Convert once here.
    home_deg = joint_positions(cfg)
    params: dict[str, Any] = {
        "planning_group": str(cfg.get("move_group", PLANNING_GROUP)),
        "planning_frame": planning_frame,
        "world_frame": world_frame,
        "ee_link": str(cfg.get("ee_link", EE_LINK)),
        "attach_link": str(cfg.get("ee_link", ATTACH_LINK)),
        "pregrasp_distance": pregrasp_distance,
        "lift_distance": lift_distance,
        "position_tolerance": float(motion_cfg.get("position_tolerance", 0.005)),
        "orientation_tolerance_deg": float(
            motion_cfg.get("orientation_tolerance_deg", 3.0)
        ),
        "max_home_error_rad": 0.03,
        "planning_time": float(motion_cfg.get("planning_time", 10.0)),
        "max_solutions": 3,
        "object_name": str(object_cfg.get("name", "small_part")),
        "object_shape": str(object_cfg.get("shape", "")),
        "object_radius": float(object_cfg.get("dimensions", {}).get("radius", 0.0)),
        "object_height": float(object_cfg.get("dimensions", {}).get("height", 0.0)),
        "table_name": table_name,
        "touch_links": list(touch_links),
        "config_file": path,
        "geometry_source": "fr_control.stage4_config + fr_control.grasp_poses",
    }
    for name, deg in zip(ARM_JOINTS, home_deg):
        params[f"home_{name}_deg"] = float(deg)
        params[f"home_{name}"] = math.radians(float(deg))

    params.update(_pose_to_dict("world_base", t_world_base, world_frame))
    params.update(_pose_to_dict("object_world", poses_world.object_pose, world_frame))
    params.update(_pose_to_dict("object", object_base, planning_frame))
    params.update(_pose_to_dict("grasp", grasp_base, planning_frame))
    params.update(_pose_to_dict("pregrasp", pregrasp_base, planning_frame))
    params.update(_pose_to_dict("lift", lift_base, planning_frame))
    params.update(_pose_to_dict("lift_world", poses_world.lift_pose, world_frame))
    params.update(_pose_to_dict("tcp_object", tcp_object, str(cfg.get("ee_link", ATTACH_LINK))))
    return params


def format_preflight(params: dict[str, Any]) -> str:
    """Human-readable STEP 3 preflight for logs."""
    lines = [
        "========== STEP 3 PREFLIGHT ==========",
        f"Planning group: {params['planning_group']}",
        f"Planning frame: {params['planning_frame']}",
        f"EE: {params['ee_link']}",
        f"Geometry source: {params['geometry_source']}",
        f"Object: {params['object_name']} {params['object_shape']} "
        f"r={params['object_radius']} h={params['object_height']}",
        "Expected Home (rad / deg):",
    ]
    for name in ARM_JOINTS:
        lines.append(
            f"  {name}={params[f'home_{name}']:.6f} rad "
            f"({params[f'home_{name}_deg']:.3f} deg)"
        )
    lines.extend(
        [
            f"Object pose world: xyz=({params['object_world_x']:.6f}, "
            f"{params['object_world_y']:.6f}, {params['object_world_z']:.6f}) "
            f"xyzw=({params['object_world_qx']:.6f}, {params['object_world_qy']:.6f}, "
            f"{params['object_world_qz']:.6f}, {params['object_world_qw']:.6f})",
            f"Object pose base_link: xyz=({params['object_x']:.6f}, "
            f"{params['object_y']:.6f}, {params['object_z']:.6f}) "
            f"xyzw=({params['object_qx']:.6f}, {params['object_qy']:.6f}, "
            f"{params['object_qz']:.6f}, {params['object_qw']:.6f})",
            f"Grasp: xyz=({params['grasp_x']:.6f}, {params['grasp_y']:.6f}, "
            f"{params['grasp_z']:.6f}) xyzw=({params['grasp_qx']:.6f}, "
            f"{params['grasp_qy']:.6f}, {params['grasp_qz']:.6f}, "
            f"{params['grasp_qw']:.6f})",
            f"PreGrasp: xyz=({params['pregrasp_x']:.6f}, {params['pregrasp_y']:.6f}, "
            f"{params['pregrasp_z']:.6f}) xyzw=({params['pregrasp_qx']:.6f}, "
            f"{params['pregrasp_qy']:.6f}, {params['pregrasp_qz']:.6f}, "
            f"{params['pregrasp_qw']:.6f})",
            f"PreGrasp distance: {params['pregrasp_distance']}",
            "Planner: OMPL",
            "THIS STEP IS PLAN-ONLY. NO TRAJECTORY EXECUTION IS PERFORMED.",
        ]
    )
    return "\n".join(lines)


def format_step4_preflight(params: dict[str, Any]) -> str:
    """Human-readable STEP 4 preflight. Geometry still comes from compute_grasp_poses()."""
    dx = float(params["grasp_x"]) - float(params["pregrasp_x"])
    dy = float(params["grasp_y"]) - float(params["pregrasp_y"])
    dz = float(params["grasp_z"]) - float(params["pregrasp_z"])
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    lines = [
        "========== STEP 4 PREFLIGHT ==========",
        f"Planning group: {params['planning_group']}",
        f"Planning frame / Goal frame: {params['planning_frame']}",
        f"IK frame: {params['ee_link']}",
        f"Geometry source: {params['geometry_source']}",
        f"Object: {params['object_name']} {params['object_shape']} "
        f"r={params['object_radius']} h={params['object_height']}",
        "Expected Home (rad / deg):",
    ]
    for name in ARM_JOINTS:
        lines.append(
            f"  {name}={params[f'home_{name}']:.6f} rad "
            f"({params[f'home_{name}_deg']:.3f} deg)"
        )
    lines.extend(
        [
            f"PreGrasp: frame={params['pregrasp_frame']} "
            f"xyz=({params['pregrasp_x']:.6f}, {params['pregrasp_y']:.6f}, "
            f"{params['pregrasp_z']:.6f}) xyzw=({params['pregrasp_qx']:.6f}, "
            f"{params['pregrasp_qy']:.6f}, {params['pregrasp_qz']:.6f}, "
            f"{params['pregrasp_qw']:.6f})",
            f"Grasp: frame={params['grasp_frame']} "
            f"xyz=({params['grasp_x']:.6f}, {params['grasp_y']:.6f}, "
            f"{params['grasp_z']:.6f}) xyzw=({params['grasp_qx']:.6f}, "
            f"{params['grasp_qy']:.6f}, {params['grasp_qz']:.6f}, "
            f"{params['grasp_qw']:.6f})",
            "PreGrasp → Grasp Cartesian displacement:",
            f"  dx={dx:.6f}",
            f"  dy={dy:.6f}",
            f"  dz={dz:.6f}",
            f"  distance={distance:.6f}",
            f"Approach / PreGrasp distance: {params['pregrasp_distance']}",
            "Stage: MoveTo PreGrasp  Pipeline: ompl",
            "Stage: MoveTo Grasp     Pipeline: pilz_industrial_motion_planner  Planner ID: LIN",
            "small_part: ABSENT — known Step 4 limitation",
            "STEP 4 validates task-level chained motion planning, not a physically complete grasp.",
            "THIS STEP IS PLAN-ONLY. NO TRAJECTORY EXECUTION IS PERFORMED.",
        ]
    )
    return "\n".join(lines)


_ARM_LINKS_MUST_NOT_TOUCH = (
    "base_link",
    "shoulder_link",
    "upperarm_link",
    "forearm_link",
    "wrist1_link",
    "wrist2_link",
    "wrist3_link",
)


def format_step5a_touch_check(params: dict[str, Any]) -> str:
    """Static STEP 5A check: YAML touch links must match gripper-only constants."""
    yaml_links = [str(name) for name in params.get("touch_links", [])]
    const_links = [str(name) for name in GRIPPER_TOUCH_LINKS]
    extra = [name for name in yaml_links if name not in const_links]
    missing = [name for name in const_links if name not in yaml_links]
    arm_allowed = [name for name in _ARM_LINKS_MUST_NOT_TOUCH if name in yaml_links]
    lines = [
        "========== STEP 5A TOUCH LINKS ==========",
        "YAML gripper_touch_links:",
        ", ".join(yaml_links) if yaml_links else "(empty)",
        "constants GRIPPER_TOUCH_LINKS:",
        ", ".join(const_links),
        f"Exact match?\n{'YES' if yaml_links == const_links else 'NO'}",
    ]
    if extra:
        lines.append(f"YAML extras: {extra}")
    if missing:
        lines.append(f"YAML missing: {missing}")
    lines.append(
        "Arm links incorrectly allowed:\n"
        + (", ".join(arm_allowed) if arm_allowed else "NONE")
    )
    return "\n".join(lines)


def format_step5_preflight(params: dict[str, Any]) -> str:
    """Human-readable STEP 5 preflight. Geometry still comes from compute_grasp_poses()."""
    lines = [
        "========== STEP 5 PREFLIGHT ==========",
        f"Planning group: {params['planning_group']}",
        f"Goal frame: {params['planning_frame']}",
        f"World frame: {params['world_frame']}",
        f"IK / attach link: {params['ee_link']}",
        f"Geometry source: {params['geometry_source']}",
        f"Object: {params['object_name']} {params['object_shape']} "
        f"r={params['object_radius']} h={params['object_height']}",
        f"Table: {params['table_name']}",
        f"Touch links: {params['touch_links']}",
        f"Object world: xyz=({params['object_world_x']:.6f}, "
        f"{params['object_world_y']:.6f}, {params['object_world_z']:.6f})",
        f"PreGrasp: xyz=({params['pregrasp_x']:.6f}, {params['pregrasp_y']:.6f}, "
        f"{params['pregrasp_z']:.6f})",
        f"Grasp: xyz=({params['grasp_x']:.6f}, {params['grasp_y']:.6f}, "
        f"{params['grasp_z']:.6f})",
        f"Lift: xyz=({params['lift_x']:.6f}, {params['lift_y']:.6f}, "
        f"{params['lift_z']:.6f})",
        f"Expected T_tcp_object: xyz=({params['tcp_object_x']:.6f}, "
        f"{params['tcp_object_y']:.6f}, {params['tcp_object_z']:.6f}) "
        f"xyzw=({params['tcp_object_qx']:.6f}, {params['tcp_object_qy']:.6f}, "
        f"{params['tcp_object_qz']:.6f}, {params['tcp_object_qw']:.6f})",
        f"PreGrasp distance: {params['pregrasp_distance']}",
        f"Lift distance from YAML: {params['lift_distance']}",
        "Physical gripper close: NOT EXECUTED",
        "Predicted grasp scene transition: YES",
        "THIS STEP IS PLAN-ONLY. NO TRAJECTORY EXECUTION IS PERFORMED.",
        "========== GRIPPER TCP ==========",
        "parent / fixed transform: gripper_base_link -> gripper_tcp xyz=(0, 0, 0.150)",
        "physical interpretation: center between the two fingertip ends",
        "T_tcp_object identity expected?\nYES",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_preflight(compute_stage4_pregrasp_params()))
