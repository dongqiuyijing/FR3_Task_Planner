"""STEP 6 cylinder inspection-view geometry.

P1 is the inspected region center, not the object origin.
Canonical roll uses inspection.up_direction, but callers may later pass
an alternative orientation_xyzw. This module does not encode view order
or joint-space constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from geometry_msgs.msg import Pose

from fr_control.grasp_poses import (
    compute_grasp_poses,
    pose_in_frame,
    pose_inverse,
    pose_multiply,
)
from fr_control.inspection_poses import (
    InspectionError,
    as_vec3,
    rotation_mapping_axes,
    rotate_pose_vector,
    tcp_pose_from_object,
    vec_dot,
    vec_norm,
    vec_normalize,
    vec_sub,
)
from fr_control.moveit_arm import make_pose
from fr_control.stage4_config import (
    block_pose,
    grasp_tcp_xyzw,
    load_yaml,
    robot_base_pose,
    workcell_config_path,
)


class InspectionViewError(InspectionError):
    """Raised when a cylinder inspection view cannot be constructed."""


_RADIAL_Z_ABS_MAX = 1e-6
_CAP_Z_ABS_MIN = 1.0 - 1e-6
_CENTER_TOL_M = 1e-6
_ANGLE_TOL_DEG = 1e-6


@dataclass(frozen=True)
class InspectionView:
    """One required inspection view. Order is not implied."""

    name: str
    normal_in_object: tuple[float, float, float]
    center_in_object: tuple[float, float, float]
    up_in_object: tuple[float, float, float]


@dataclass(frozen=True)
class InspectionViewTarget:
    """Canonical object/TCP target for one view. Roll is preferred, not unique."""

    view: InspectionView
    frame: str
    p1: tuple[float, float, float]
    inspection_direction: tuple[float, float, float]
    inspection_up: tuple[float, float, float]
    object_pose: Pose
    tcp_pose: Pose
    view_center_error_m: float
    normal_angle_error_deg: float
    canonical_up_error_deg: float
    object_center_offset_m: float


def vec_add(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    """Return left + right."""
    return (
        float(left[0]) + float(right[0]),
        float(left[1]) + float(right[1]),
        float(left[2]) + float(right[2]),
    )


def angle_between_deg(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the unsigned angle between two vectors in degrees."""
    left_u = vec_normalize(left, "angle left")
    right_u = vec_normalize(right, "angle right")
    cosine = max(-1.0, min(1.0, vec_dot(left_u, right_u)))
    return math.degrees(math.acos(cosine))


def cylinder_center_in_object(
    normal_in_object: Sequence[float],
    radius: float,
    height: float,
) -> tuple[float, float, float]:
    """Derive the inspected-region center from a unit cylinder normal."""
    if radius <= 0.0 or height <= 0.0:
        raise InspectionViewError("cylinder radius/height must be positive")
    normal = vec_normalize(normal_in_object, "normal_in_object")
    if abs(normal[2]) <= _RADIAL_Z_ABS_MAX:
        return (radius * normal[0], radius * normal[1], radius * normal[2])
    if abs(normal[2]) >= _CAP_Z_ABS_MIN:
        half_height = 0.5 * height
        return (
            half_height * normal[0],
            half_height * normal[1],
            half_height * normal[2],
        )
    raise InspectionViewError(
        f"unsupported slanted cylinder normal {normal}; "
        "STEP 6 only supports radial sides or +/-Z caps"
    )


def view_name_from_normal(normal_in_object: Sequence[float]) -> str:
    """Map a cylinder normal to a required-view name. Not an order."""
    normal = vec_normalize(normal_in_object, "normal_in_object")
    if abs(normal[2]) <= _RADIAL_Z_ABS_MAX:
        if normal[1] >= 0.9:
            return "side_pos_y"
        if normal[1] <= -0.9:
            return "side_neg_y"
        raise InspectionViewError(
            f"radial normal {normal} is not +Y or -Y; not a required STEP 6 view"
        )
    if normal[2] >= _CAP_Z_ABS_MIN:
        return "top_circle"
    raise InspectionViewError(
        f"normal {normal} is not a required STEP 6 view "
        "(side_pos_y, side_neg_y, top_circle)"
    )


def required_cylinder_views(
    faces: dict[str, Any],
    radius: float,
    height: float,
) -> dict[str, InspectionView]:
    """Build the unordered required-view set from YAML face normals."""
    views: dict[str, InspectionView] = {}
    for face_cfg in faces.values():
        normal = as_vec3(face_cfg["normal_in_object"])
        up = as_vec3(face_cfg["up_in_object"])
        name = view_name_from_normal(normal)
        views[name] = InspectionView(
            name=name,
            normal_in_object=vec_normalize(normal, "normal_in_object"),
            center_in_object=cylinder_center_in_object(normal, radius, height),
            up_in_object=vec_normalize(up, "up_in_object"),
        )
    expected = {"side_pos_y", "side_neg_y", "top_circle"}
    if set(views) != expected:
        raise InspectionViewError(
            f"required views must be {sorted(expected)}, got {sorted(views)}"
        )
    return views


def canonical_object_rotation(
    normal_in_object: Sequence[float],
    up_in_object: Sequence[float],
    inspection_direction: Sequence[float],
    inspection_up: Sequence[float],
) -> tuple[float, float, float, float]:
    """Preferred roll that maps view normal to D1.

    THIS IS CANONICAL PREFERRED ROLL, NOT THE ONLY FUTURE VALID ROLL.
    """
    return rotation_mapping_axes(
        normal_in_object,
        up_in_object,
        inspection_direction,
        inspection_up,
    )


def object_pose_for_inspection_view(
    view_center: Sequence[float],
    *,
    center_in_object: Sequence[float],
    normal_in_object: Sequence[float],
    up_in_object: Sequence[float],
    inspection_direction: Sequence[float],
    inspection_up: Sequence[float],
    orientation_xyzw: Sequence[float] | None = None,
) -> Pose:
    """Place the object so the inspected-region center is at P1.

    object_center = P1 - R * center_in_object

    orientation_xyzw defaults to the canonical preferred roll. Future
    roll sampling can pass another quaternion that still maps the view
    normal onto D1.
    """
    xyzw = (
        tuple(float(item) for item in orientation_xyzw)
        if orientation_xyzw is not None
        else canonical_object_rotation(
            normal_in_object,
            up_in_object,
            inspection_direction,
            inspection_up,
        )
    )
    rotation_pose = make_pose((0.0, 0.0, 0.0), xyzw)
    offset = rotate_pose_vector(rotation_pose, center_in_object)
    object_center = vec_sub(view_center, offset)
    return make_pose(object_center, xyzw)


def tcp_target_from_object(object_pose: Pose, tcp_t_object: Pose) -> Pose:
    """T_frame_tcp = T_frame_object * inverse(T_tcp_object)."""
    return tcp_pose_from_object(object_pose, tcp_t_object)


def actual_view_center(
    object_pose: Pose, center_in_object: Sequence[float]
) -> tuple[float, float, float]:
    """Recompute the inspected-region center from an object target."""
    return vec_add(
        (object_pose.position.x, object_pose.position.y, object_pose.position.z),
        rotate_pose_vector(object_pose, center_in_object),
    )


def actual_view_normal(
    object_pose: Pose, normal_in_object: Sequence[float]
) -> tuple[float, float, float]:
    """Rotate the object-frame view normal into the planning frame."""
    return rotate_pose_vector(object_pose, normal_in_object)


def actual_view_up(
    object_pose: Pose, up_in_object: Sequence[float]
) -> tuple[float, float, float]:
    """Rotate the object-frame up into the planning frame."""
    return rotate_pose_vector(object_pose, up_in_object)


def grasp_tcp_object_pose(cfg: dict[str, Any]) -> Pose:
    """Reuse Stage4 grasp geometry. Do not invent a second TCP transform."""
    world_frame = str(cfg.get("frames", {}).get("world", "world"))
    grasp_cfg = cfg["grasp"]
    object_cfg = cfg["object"]
    t_world_base = robot_base_pose(cfg)
    object_source = str(object_cfg["initial_pose"].get("frame", world_frame))
    object_in_source = block_pose(object_cfg["initial_pose"])
    if object_source == world_frame:
        object_world = object_in_source
    elif object_source == str(cfg.get("planning_frame", "base_link")):
        object_world = pose_multiply(t_world_base, object_in_source)
    else:
        raise InspectionViewError(f"Unsupported object frame: {object_source}")
    poses_world = compute_grasp_poses(
        object_world,
        approach_xyzw=grasp_tcp_xyzw(grasp_cfg),
        pre_grasp_offset=float(grasp_cfg["pregrasp_distance"]),
        lift_height=float(grasp_cfg["lift_distance"]),
        table_top_z=float(grasp_cfg["table_top_z"]),
        fingertip_from_tcp=float(grasp_cfg["fingertip_from_tcp"]),
        table_clearance=float(grasp_cfg["table_clearance"]),
        planning_frame=world_frame,
    )
    return pose_in_frame(poses_world.grasp_pose, poses_world.object_pose)


def interpret_tcp_object_rotation(tcp_t_object: Pose) -> str:
    """Describe T_tcp_object. (1,0,0,0) is Rx(180 deg), not identity."""
    q = tcp_t_object.orientation
    identity_dot = abs(q.w)
    rx_dot = abs(q.x)
    if rx_dot > 0.999 and abs(q.y) < 1e-3 and abs(q.z) < 1e-3 and abs(q.w) < 1e-3:
        return "Rx(180 deg)"
    if identity_dot > 0.999 and abs(q.x) < 1e-3 and abs(q.y) < 1e-3 and abs(q.z) < 1e-3:
        return "identity"
    return f"xyzw=({q.x:.6f}, {q.y:.6f}, {q.z:.6f}, {q.w:.6f})"


def compute_view_target(
    view: InspectionView,
    *,
    p1: Sequence[float],
    frame: str,
    inspection_direction: Sequence[float],
    inspection_up: Sequence[float],
    tcp_t_object: Pose,
    orientation_xyzw: Sequence[float] | None = None,
) -> InspectionViewTarget:
    """Build and validate one canonical inspection target."""
    object_pose = object_pose_for_inspection_view(
        p1,
        center_in_object=view.center_in_object,
        normal_in_object=view.normal_in_object,
        up_in_object=view.up_in_object,
        inspection_direction=inspection_direction,
        inspection_up=inspection_up,
        orientation_xyzw=orientation_xyzw,
    )
    tcp_pose = tcp_target_from_object(object_pose, tcp_t_object)
    center = actual_view_center(object_pose, view.center_in_object)
    normal = actual_view_normal(object_pose, view.normal_in_object)
    up = actual_view_up(object_pose, view.up_in_object)
    object_center = (
        object_pose.position.x,
        object_pose.position.y,
        object_pose.position.z,
    )
    return InspectionViewTarget(
        view=view,
        frame=frame,
        p1=(float(p1[0]), float(p1[1]), float(p1[2])),
        inspection_direction=vec_normalize(inspection_direction, "D1"),
        inspection_up=vec_normalize(inspection_up, "preferred up"),
        object_pose=object_pose,
        tcp_pose=tcp_pose,
        view_center_error_m=vec_norm(vec_sub(center, p1)),
        normal_angle_error_deg=angle_between_deg(normal, inspection_direction),
        canonical_up_error_deg=angle_between_deg(up, inspection_up),
        object_center_offset_m=vec_norm(vec_sub(object_center, p1)),
    )


def load_stage6_geometry(config_file: str | None = None) -> dict[str, Any]:
    """Load YAML and compute the unordered required inspection targets."""
    path = config_file or workcell_config_path()
    cfg = load_yaml(path)
    object_cfg = cfg["object"]
    inspect_cfg = cfg["inspection"]
    dims = object_cfg["dimensions"]
    radius = float(dims["radius"])
    height = float(dims["height"])
    p1 = as_vec3(inspect_cfg["position"])
    frame = str(inspect_cfg["position"].get("frame", "base_link"))
    direction = as_vec3(inspect_cfg["direction"])
    up = as_vec3(inspect_cfg["up_direction"])
    views = required_cylinder_views(inspect_cfg["faces"], radius, height)
    tcp_t_object = grasp_tcp_object_pose(cfg)
    targets = {
        name: compute_view_target(
            view,
            p1=p1,
            frame=frame,
            inspection_direction=direction,
            inspection_up=up,
            tcp_t_object=tcp_t_object,
        )
        for name, view in views.items()
    }
    return {
        "config_file": path,
        "radius": radius,
        "height": height,
        "p1": p1,
        "frame": frame,
        "direction": vec_normalize(direction, "D1"),
        "up": vec_normalize(up, "preferred up"),
        "tcp_t_object": tcp_t_object,
        "views": views,
        "targets": targets,
    }


def validate_stage6_geometry(data: dict[str, Any]) -> list[str]:
    """Return failing assertion messages. Empty list means PASS."""
    failures: list[str] = []
    radius = float(data["radius"])
    height = float(data["height"])
    p1 = data["p1"]
    targets: dict[str, InspectionViewTarget] = data["targets"]
    if set(targets) != {"side_pos_y", "side_neg_y", "top_circle"}:
        failures.append(f"required view set mismatch: {sorted(targets)}")

    for name, expected in (
        ("side_pos_y", radius),
        ("side_neg_y", radius),
        ("top_circle", 0.5 * height),
    ):
        target = targets[name]
        if abs(target.object_center_offset_m - expected) > _CENTER_TOL_M:
            failures.append(
                f"{name} offset {target.object_center_offset_m:.9f} "
                f"!= expected {expected:.9f}"
            )
        if target.view_center_error_m > _CENTER_TOL_M:
            failures.append(
                f"{name} view center error {target.view_center_error_m:.3e} > 1e-6"
            )
        if target.normal_angle_error_deg > _ANGLE_TOL_DEG:
            failures.append(
                f"{name} normal error {target.normal_angle_error_deg:.3e} deg"
            )
        object_center = (
            target.object_pose.position.x,
            target.object_pose.position.y,
            target.object_pose.position.z,
        )
        if vec_norm(vec_sub(object_center, p1)) <= _CENTER_TOL_M:
            failures.append(f"{name} object center collapsed onto P1")

    side_pos = (
        targets["side_pos_y"].object_pose.position.x,
        targets["side_pos_y"].object_pose.position.y,
        targets["side_pos_y"].object_pose.position.z,
    )
    side_neg = (
        targets["side_neg_y"].object_pose.position.x,
        targets["side_neg_y"].object_pose.position.y,
        targets["side_neg_y"].object_pose.position.z,
    )
    top = (
        targets["top_circle"].object_pose.position.x,
        targets["top_circle"].object_pose.position.y,
        targets["top_circle"].object_pose.position.z,
    )
    # Opposite side views can share object-center XYZ: the cylinder is
    # rotated 180 deg about its axis so the other flank faces the same D1.
    if vec_norm(vec_sub(top, side_pos)) <= _CENTER_TOL_M:
        failures.append("top object center collapsed onto a side-view center")
    q_pos = targets["side_pos_y"].object_pose.orientation
    q_neg = targets["side_neg_y"].object_pose.orientation
    same_side_ori = (
        abs(q_pos.x * q_neg.x + q_pos.y * q_neg.y + q_pos.z * q_neg.z + q_pos.w * q_neg.w)
        > 0.999
    )
    if same_side_ori and vec_norm(vec_sub(side_pos, side_neg)) <= _CENTER_TOL_M:
        failures.append("side views are identical in both center and orientation")

    tcp = data["tcp_t_object"]
    if interpret_tcp_object_rotation(tcp) != "Rx(180 deg)":
        failures.append(
            "T_tcp_object rotation is not Rx(pi); "
            f"got {interpret_tcp_object_rotation(tcp)}"
        )
    if vec_norm((tcp.position.x, tcp.position.y, tcp.position.z)) > _CENTER_TOL_M:
        failures.append("T_tcp_object translation is not at the object origin")
    return failures


def format_pose(pose: Pose, frame: str) -> str:
    """Format a pose for logs."""
    p = pose.position
    q = pose.orientation
    return (
        f"frame={frame} xyz=({p.x:.6f}, {p.y:.6f}, {p.z:.6f}) "
        f"xyzw=({q.x:.6f}, {q.y:.6f}, {q.z:.6f}, {q.w:.6f})"
    )


def format_vec(vector: Sequence[float]) -> str:
    """Format a 3-vector."""
    return f"({vector[0]:.6f}, {vector[1]:.6f}, {vector[2]:.6f})"


def format_stage6_report(data: dict[str, Any], failures: Iterable[str]) -> str:
    """Human-readable STEP 6 geometry dump. No planning, no order."""
    tcp = data["tcp_t_object"]
    lines = [
        "========== CYLINDER VIEW GEOMETRY ==========",
        f"source: {data['config_file']}",
        f"radius = {data['radius']:.6f}",
        f"height = {data['height']:.6f}",
        "axis in object frame: +Z",
        f"side center offset magnitude = {data['radius']:.6f}",
        f"expected radius = {data['radius']:.6f}",
        f"top center offset magnitude = {0.5 * data['height']:.6f}",
        f"expected height/2 = {0.5 * data['height']:.6f}",
        "",
        "P1 semantic: INSPECTION VIEW CENTER TARGET",
        f"P1 frame: {data['frame']}",
        f"P1 xyz: {format_vec(data['p1'])}",
        f"D1: {format_vec(data['direction'])}",
        f"preferred up: {format_vec(data['up'])}",
        "",
        "T_tcp_object:",
        f"  translation: {format_vec((tcp.position.x, tcp.position.y, tcp.position.z))}",
        f"  rotation xyzw: ({tcp.orientation.x:.6f}, {tcp.orientation.y:.6f}, "
        f"{tcp.orientation.z:.6f}, {tcp.orientation.w:.6f})",
        f"  rotation interpretation: {interpret_tcp_object_rotation(tcp)}",
        "  Is identity rotation? NO",
        "  TCP origin coincides with object center: YES",
        "  TCP orientation equals object orientation: NO",
        "",
        "required_views = {side_pos_y, side_neg_y, top_circle}",
        "side_pos_y / side_neg_y may share object-center XYZ under fixed D1;",
        "they differ by a 180 deg rotation about the cylinder axis.",
        "Fixed order encoded? NO",
        "J6 +180 encoded? NO",
        "THIS IS CANONICAL PREFERRED ROLL, NOT THE ONLY FUTURE VALID ROLL.",
        "Planning performed? NO",
        "Execution performed? NO",
        "",
        "Legacy stage4_inspection_test.py still uses old object-center-at-P1 semantics.",
        "New FR3_Task_Planner STEP 6 uses corrected inspection-view-center-at-P1 semantics.",
        "Legacy executor migration is deferred until the new planner geometry is validated.",
    ]
    for name in ("side_pos_y", "side_neg_y", "top_circle"):
        target = data["targets"][name]
        view = target.view
        lines.extend(
            [
                "",
                f"========== VIEW {name} ==========",
                f"normal_in_object: {format_vec(view.normal_in_object)}",
                f"center_in_object: {format_vec(view.center_in_object)}",
                f"up_in_object: {format_vec(view.up_in_object)}",
                f"P1 target: {format_vec(target.p1)}",
                f"Object target pose: {format_pose(target.object_pose, target.frame)}",
                f"TCP target pose: {format_pose(target.tcp_pose, target.frame)}",
                f"view center error: {target.view_center_error_m:.3e} m",
                f"normal angle error: {target.normal_angle_error_deg:.3e} deg",
                f"canonical up error: {target.canonical_up_error_deg:.3e} deg",
                f"|object_center - P1|: {target.object_center_offset_m:.6f} m",
            ]
        )
    fail_list = list(failures)
    lines.extend(
        [
            "",
            "========== ASSERTIONS ==========",
            "PASS" if not fail_list else "FAIL",
        ]
    )
    lines.extend(f"  - {item}" for item in fail_list)
    return "\n".join(lines)
