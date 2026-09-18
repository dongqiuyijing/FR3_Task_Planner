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
from fr_control.grasp_poses import quat_to_matrix, quat_xyzw
from fr_control.inspection_poses import (
    InspectionError,
    as_vec3,
    matrix_to_xyzw,
    quat_angle_deg,
    quat_normalize,
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
_ORTHO_DOT_TOL = 1e-6
_UNIT_NORM_TOL = 1e-9

# STEP 12C user-fixed design. Launch-time override only; do not rewrite
# stage4_config.yaml inspection/camera fields, and do not search these values.
STEP12C_P1 = (0.0, 0.30, 1.20)
STEP12C_CAMERA_POSITION = (0.0, 0.0, 1.40)
STEP12C_CAMERA_RPY = (-2.35619449, 0.0, 0.0)
STEP12C_SURFACE_RPY = (0.785398, 0.0, 0.0)
STEP12C_HOME_DEG = (
    -130.142302560334,
    -94.12911896658416,
    -107.1221379950495,
    -164.5057965269183,
    -0.19340288521039606,
    48.12272780012376,
)


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


def identity_pose() -> Pose:
    """Return a Pose with zero translation and identity rotation."""
    return make_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def transform_point(point: Sequence[float], transform: Pose) -> tuple[float, float, float]:
    """Apply rotation + translation: R*p + t."""
    rotated = rotate_pose_vector(transform, point)
    return vec_add(
        (transform.position.x, transform.position.y, transform.position.z),
        rotated,
    )


def transform_vector(
    vector: Sequence[float], transform: Pose
) -> tuple[float, float, float]:
    """Apply rotation only. Do not add translation (for D1 / up)."""
    return rotate_pose_vector(transform, vector)


def transform_pose(pose: Pose, transform: Pose) -> Pose:
    """Left-multiply: T_out = transform * pose."""
    return pose_multiply(transform, pose)


def _named_frames(cfg: dict[str, Any]) -> tuple[str, str]:
    world_frame = str(cfg.get("frames", {}).get("world", "world"))
    base_frame = str(cfg.get("frames", {}).get("robot_base", cfg.get("planning_frame", "base_link")))
    return world_frame, base_frame


def frame_transform(
    source_frame: str,
    target_frame: str,
    t_world_base: Pose,
    world_frame: str,
    base_frame: str,
) -> Pose:
    """Return T_target_source from YAML T_world_base. No hardcoded numbers."""
    if source_frame == target_frame:
        return identity_pose()
    if source_frame == world_frame and target_frame == base_frame:
        return pose_inverse(t_world_base)
    if source_frame == base_frame and target_frame == world_frame:
        return t_world_base
    raise InspectionViewError(
        f"unsupported frame conversion {source_frame} -> {target_frame}; "
        f"known frames are {world_frame!r} and {base_frame!r}"
    )


def express_point(
    point: Sequence[float],
    source_frame: str,
    target_frame: str,
    t_world_base: Pose,
    world_frame: str,
    base_frame: str,
) -> tuple[float, float, float]:
    """Express a position in target_frame."""
    return transform_point(
        point,
        frame_transform(source_frame, target_frame, t_world_base, world_frame, base_frame),
    )


def express_vector(
    vector: Sequence[float],
    source_frame: str,
    target_frame: str,
    t_world_base: Pose,
    world_frame: str,
    base_frame: str,
) -> tuple[float, float, float]:
    """Express a free vector (direction) in target_frame. Rotation only."""
    return transform_vector(
        vector,
        frame_transform(source_frame, target_frame, t_world_base, world_frame, base_frame),
    )


def express_pose(
    pose: Pose,
    source_frame: str,
    target_frame: str,
    t_world_base: Pose,
    world_frame: str,
    base_frame: str,
) -> Pose:
    """Express a Pose in target_frame."""
    return transform_pose(
        pose,
        frame_transform(source_frame, target_frame, t_world_base, world_frame, base_frame),
    )


def require_unit_orthogonal(
    direction: Sequence[float],
    up: Sequence[float],
    *,
    direction_name: str = "D1",
    up_name: str = "preferred up",
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Abort if D1/up are not unit length or not orthogonal."""
    direction_u = vec_normalize(direction, direction_name)
    up_u = vec_normalize(up, up_name)
    if abs(vec_norm(direction_u) - 1.0) > _UNIT_NORM_TOL:
        raise InspectionViewError(f"ABORT: {direction_name} is not unit length")
    if abs(vec_norm(up_u) - 1.0) > _UNIT_NORM_TOL:
        raise InspectionViewError(f"ABORT: {up_name} is not unit length")
    if abs(vec_dot(direction_u, up_u)) > _ORTHO_DOT_TOL:
        raise InspectionViewError(
            f"ABORT: {direction_name} and {up_name} are not orthogonal "
            f"(dot={vec_dot(direction_u, up_u):.6e})"
        )
    return direction_u, up_u


def pose_in_planning_frame(pose: Pose, geo: dict[str, Any], planning_frame: str) -> Pose:
    """Convert a working-frame pose into the MoveIt planning frame."""
    return express_pose(
        pose,
        geo["working_frame"],
        planning_frame,
        geo["t_world_base"],
        geo["world_frame"],
        geo["base_frame"],
    )


def cpp_inspection_vectors(geo: dict[str, Any]) -> dict[str, Any]:
    """P1/D1/up in world for C++ world-frame validation."""
    p1 = geo["p1_world"]
    d1 = geo["d1_world"]
    up = geo["up_world"]
    return {
        "p1_x": float(p1[0]),
        "p1_y": float(p1[1]),
        "p1_z": float(p1[2]),
        "d1_x": float(d1[0]),
        "d1_y": float(d1[1]),
        "d1_z": float(d1[2]),
        "preferred_up_x": float(up[0]),
        "preferred_up_y": float(up[1]),
        "preferred_up_z": float(up[2]),
        "inspection_vector_frame": geo["working_frame"],
    }


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
    if normal[2] <= -_CAP_Z_ABS_MIN:
        return "bottom_circle"
    raise InspectionViewError(
        f"normal {normal} is not a required cylinder face "
        "(side_pos_y, side_neg_y, top_circle, bottom_circle)"
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
    if not expected.issubset(set(views)):
        raise InspectionViewError(
            f"required views must include {sorted(expected)}, got {sorted(views)}"
        )
    if "bottom_circle" in views and views["top_circle"].normal_in_object[2] <= 0.0:
        raise InspectionViewError("top_circle must remain object +Z; do not swap with bottom")
    return views


def make_bottom_circle_view(height: float) -> InspectionView:
    """ARM1 original table-contact face: object -Z. Does not replace top_circle."""
    normal = (0.0, 0.0, -1.0)
    # Orthogonal to -Z. Canonical roll maps this up onto world +Z.
    up = (0.0, 1.0, 0.0)
    return InspectionView(
        name="bottom_circle",
        normal_in_object=normal,
        center_in_object=cylinder_center_in_object(normal, 1.0, height),
        up_in_object=up,
    )


def attach_bottom_circle_view(data: dict[str, Any]) -> dict[str, Any]:
    """Add ARM1 C = original bottom without reinterpreting top_circle as -Z."""
    views = dict(data["views"])
    if "top_circle" not in views:
        raise InspectionViewError("top_circle must be preserved for future ARM2")
    top = views["top_circle"]
    if top.normal_in_object[2] < _CAP_Z_ABS_MIN:
        raise InspectionViewError(
            f"top_circle must stay object +Z, got {top.normal_in_object}"
        )
    view = make_bottom_circle_view(float(data["height"]))
    if abs(view.center_in_object[2] + 0.5 * float(data["height"])) > _CENTER_TOL_M:
        raise InspectionViewError(
            f"bottom_circle center {view.center_in_object} is not [0,0,-H/2]"
        )
    views["bottom_circle"] = view
    targets = dict(data["targets"])
    targets["bottom_circle"] = compute_view_target(
        view,
        p1=data["p1_world"],
        frame=data["working_frame"],
        inspection_direction=data["d1_world"],
        inspection_up=data["up_world"],
        tcp_t_object=data["tcp_t_object"],
    )
    out = dict(data)
    out["views"] = views
    out["targets"] = targets
    out["arm1_views"] = ("side_pos_y", "side_neg_y", "bottom_circle")
    out["arm2_future_view"] = "top_circle"
    return out


def sample_circular_cap_local(
    z_local: float, radius: float
) -> list[tuple[float, float, float]]:
    """Object-frame disk samples on z = z_local. Matches C++ circular ROI rings."""
    if radius <= 0.0:
        raise InspectionViewError("circular ROI radius must be positive")
    pts: list[tuple[float, float, float]] = [(0.0, 0.0, float(z_local))]
    rings = (0.25, 0.50, 0.75, 0.95)
    n_ang = (8, 8, 12, 16)
    for ring, count in zip(rings, n_ang):
        rad = ring * radius
        for k in range(count):
            ang = 2.0 * math.pi * float(k) / float(count)
            pts.append((rad * math.cos(ang), rad * math.sin(ang), float(z_local)))
    return pts


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


def canonicalize_roll_deg(angle_deg: float) -> float:
    """Map an angle to (-180, 180]. +180 is kept; -180 becomes +180."""
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if abs(wrapped + 180.0) < 1e-9:
        return 180.0
    if abs(wrapped) < 1e-12:
        return 0.0
    return wrapped


def roll_sample_degrees(roll_step_deg: float) -> list[float]:
    """Deterministic roll grid. 0 first, then +step, -step, ..., 180."""
    step = float(roll_step_deg)
    if step <= 0.0 or step > 180.0:
        raise InspectionViewError(
            f"roll_step_deg must satisfy 0 < step <= 180, got {step}"
        )
    n = 360.0 / step
    if abs(n - round(n)) > 1e-9:
        raise InspectionViewError(
            f"360 / roll_step_deg must be an integer, got 360/{step}={n}"
        )
    count = int(round(n))
    ordered: list[float] = [0.0]
    k = 1
    while True:
        pos = canonicalize_roll_deg(k * step)
        if abs(pos - 180.0) < 1e-9 or pos < 0.0:
            if 180.0 not in ordered:
                ordered.append(180.0)
            break
        neg = canonicalize_roll_deg(-k * step)
        if pos not in ordered:
            ordered.append(pos)
        if neg not in ordered and abs(neg - pos) > 1e-9:
            ordered.append(neg)
        if len(ordered) >= count:
            break
        k += 1
    if len(ordered) != count:
        raise InspectionViewError(
            f"roll grid size {len(ordered)} != expected {count}"
        )
    return ordered


def _matmul(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(
            left[i][0] * right[0][j]
            + left[i][1] * right[1][j]
            + left[i][2] * right[2][j]
            for j in range(3)
        )
        for i in range(3)
    )


def rotation_about_axis(
    axis: Sequence[float], angle_deg: float
) -> tuple[tuple[float, float, float], ...]:
    """Rodrigues rotation about a unit axis, in the reference frame."""
    axis_u = vec_normalize(axis, "roll axis")
    rad = math.radians(float(angle_deg))
    c = math.cos(rad)
    s = math.sin(rad)
    x, y, z = axis_u
    return (
        (c + x * x * (1.0 - c), x * y * (1.0 - c) - z * s, x * z * (1.0 - c) + y * s),
        (y * x * (1.0 - c) + z * s, c + y * y * (1.0 - c), y * z * (1.0 - c) - x * s),
        (z * x * (1.0 - c) - y * s, z * y * (1.0 - c) + x * s, c + z * z * (1.0 - c)),
    )


@dataclass(frozen=True)
class InspectionRollCandidate:
    """One deterministic roll pose. up_error is a soft diagnostic."""

    view: InspectionView
    roll_deg: float
    pose_index: int
    frame: str
    object_pose: Pose
    tcp_pose: Pose
    view_center_error_m: float
    normal_angle_error_deg: float
    up_error_deg: float


def generate_roll_candidates(
    target: InspectionViewTarget,
    tcp_t_object: Pose,
    roll_step_deg: float = 30.0,
) -> list[InspectionRollCandidate]:
    """Enumerate unique roll poses around D1. roll=0 is the STEP 6 canonical."""
    angles = roll_sample_degrees(roll_step_deg)
    r_canonical = quat_to_matrix(quat_xyzw(target.object_pose.orientation))
    out: list[InspectionRollCandidate] = []
    seen: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]] = []
    for index, raw_deg in enumerate(angles):
        roll_deg = canonicalize_roll_deg(raw_deg)
        r_roll = _matmul(rotation_about_axis(target.inspection_direction, roll_deg), r_canonical)
        xyzw = quat_normalize(matrix_to_xyzw(r_roll))
        object_pose = object_pose_for_inspection_view(
            target.p1,
            center_in_object=target.view.center_in_object,
            normal_in_object=target.view.normal_in_object,
            up_in_object=target.view.up_in_object,
            inspection_direction=target.inspection_direction,
            inspection_up=target.inspection_up,
            orientation_xyzw=xyzw,
        )
        tcp_pose = tcp_target_from_object(object_pose, tcp_t_object)
        pos = (
            round(object_pose.position.x, 9),
            round(object_pose.position.y, 9),
            round(object_pose.position.z, 9),
        )
        q = quat_normalize(quat_xyzw(object_pose.orientation))
        # Treat q and -q as the same orientation.
        q_flip = tuple(round(-v, 9) for v in q)
        q_flip = quat_normalize(q_flip)
        duplicate = False
        for prev_pos, prev_q in seen:
            if prev_pos != pos:
                continue
            same = all(abs(a - b) < 1e-9 for a, b in zip(q, prev_q))
            flipped = all(abs(a - b) < 1e-9 for a, b in zip(q_flip, prev_q))
            if same or flipped:
                duplicate = True
                break
        if duplicate:
            continue
        seen.append((pos, q))
        center = actual_view_center(object_pose, target.view.center_in_object)
        normal = actual_view_normal(object_pose, target.view.normal_in_object)
        up = actual_view_up(object_pose, target.view.up_in_object)
        out.append(
            InspectionRollCandidate(
                view=target.view,
                roll_deg=roll_deg,
                pose_index=len(out),
                frame=target.frame,
                object_pose=object_pose,
                tcp_pose=tcp_pose,
                view_center_error_m=vec_norm(vec_sub(center, target.p1)),
                normal_angle_error_deg=angle_between_deg(normal, target.inspection_direction),
                up_error_deg=angle_between_deg(up, target.inspection_up),
            )
        )
    return out


def retarget_inspection_p1(data: dict[str, Any], p1_world: Sequence[float]) -> dict[str, Any]:
    """Rebuild view targets at a new P1. Does not write stage4_config.yaml."""
    p1 = (float(p1_world[0]), float(p1_world[1]), float(p1_world[2]))
    out = dict(data)
    out["p1"] = p1
    out["p1_world"] = p1
    out["p1_base"] = express_point(
        p1,
        out["world_frame"],
        out["base_frame"],
        out["t_world_base"],
        out["world_frame"],
        out["base_frame"],
    )
    out["targets"] = {
        name: compute_view_target(
            view,
            p1=p1,
            frame=out["working_frame"],
            inspection_direction=out["d1_world"],
            inspection_up=out["up_world"],
            tcp_t_object=out["tcp_t_object"],
        )
        for name, view in out["views"].items()
    }
    return out


def rpy_rotation_matrix(
    roll: float, pitch: float, yaw: float
) -> tuple[tuple[float, float, float], ...]:
    """Standard ROS/URDF RPY: R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    rx = ((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr))
    ry = ((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp))
    rz = ((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0))
    return _matmul(rz, _matmul(ry, rx))


def rotate_matrix_vec(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> tuple[float, float, float]:
    """Apply a 3x3 rotation matrix to a 3-vector."""
    return (
        float(matrix[0][0]) * float(vector[0])
        + float(matrix[0][1]) * float(vector[1])
        + float(matrix[0][2]) * float(vector[2]),
        float(matrix[1][0]) * float(vector[0])
        + float(matrix[1][1]) * float(vector[1])
        + float(matrix[1][2]) * float(vector[2]),
        float(matrix[2][0]) * float(vector[0])
        + float(matrix[2][1]) * float(vector[1])
        + float(matrix[2][2]) * float(vector[2]),
    )


def camera_optical_axes_from_rpy(
    rpy: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Optical +Z forward, +X right, +Y down from world RPY."""
    rot = rpy_rotation_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    forward = vec_normalize(rotate_matrix_vec(rot, (0.0, 0.0, 1.0)), "camera forward")
    x_right = vec_normalize(rotate_matrix_vec(rot, (1.0, 0.0, 0.0)), "camera +X")
    y_down = vec_normalize(rotate_matrix_vec(rot, (0.0, 1.0, 0.0)), "camera +Y down")
    return forward, x_right, y_down


def surface_frame_from_rpy(
    rpy: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Surface frame: local +Z = outward normal, +X/+Y in plane."""
    rot = rpy_rotation_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    x_axis = vec_normalize(rotate_matrix_vec(rot, (1.0, 0.0, 0.0)), "surface +X")
    y_axis = vec_normalize(rotate_matrix_vec(rot, (0.0, 1.0, 0.0)), "surface +Y")
    z_axis = vec_normalize(rotate_matrix_vec(rot, (0.0, 0.0, 1.0)), "surface normal")
    return x_axis, y_axis, z_axis


def retarget_inspection_geometry(
    data: dict[str, Any],
    p1_world: Sequence[float],
    d1_world: Sequence[float],
    up_world: Sequence[float],
) -> dict[str, Any]:
    """Rebuild targets at a new P1 and canonical n_target. Does not write YAML."""
    d1, up = require_unit_orthogonal(d1_world, up_world)
    p1 = (float(p1_world[0]), float(p1_world[1]), float(p1_world[2]))
    out = dict(data)
    out["p1"] = p1
    out["p1_world"] = p1
    out["direction"] = d1
    out["d1_world"] = d1
    out["up"] = up
    out["up_world"] = up
    out["p1_base"] = express_point(
        p1,
        out["world_frame"],
        out["base_frame"],
        out["t_world_base"],
        out["world_frame"],
        out["base_frame"],
    )
    out["d1_base"] = express_vector(
        d1,
        out["world_frame"],
        out["base_frame"],
        out["t_world_base"],
        out["world_frame"],
        out["base_frame"],
    )
    out["up_base"] = express_vector(
        up,
        out["world_frame"],
        out["base_frame"],
        out["t_world_base"],
        out["world_frame"],
        out["base_frame"],
    )
    out["targets"] = {
        name: compute_view_target(
            view,
            p1=p1,
            frame=out["working_frame"],
            inspection_direction=d1,
            inspection_up=up,
            tcp_t_object=out["tcp_t_object"],
        )
        for name, view in out["views"].items()
    }
    return out


def apply_step12c_tilted_geometry(data: dict[str, Any]) -> dict[str, Any]:
    """Fixed STEP12C P1 + tilted n_target + ARM1 bottom face. Camera is separate."""
    _, up_axis, n_target = surface_frame_from_rpy(STEP12C_SURFACE_RPY)
    return attach_bottom_circle_view(
        retarget_inspection_geometry(data, STEP12C_P1, n_target, up_axis)
    )


def load_stage6_geometry(config_file: str | None = None) -> dict[str, Any]:
    """Load YAML and compute required inspection targets in world."""
    path = config_file or workcell_config_path()
    cfg = load_yaml(path)
    world_frame, base_frame = _named_frames(cfg)
    t_world_base = robot_base_pose(cfg)
    object_cfg = cfg["object"]
    inspect_cfg = cfg["inspection"]
    dims = object_cfg["dimensions"]
    radius = float(dims["radius"])
    height = float(dims["height"])
    if "frame" not in inspect_cfg["position"]:
        raise InspectionViewError("inspection.position.frame is required")
    if "frame" not in inspect_cfg["direction"]:
        raise InspectionViewError("inspection.direction.frame is required")
    if "frame" not in inspect_cfg["up_direction"]:
        raise InspectionViewError("inspection.up_direction.frame is required")
    p1_src_frame = str(inspect_cfg["position"]["frame"])
    d1_src_frame = str(inspect_cfg["direction"]["frame"])
    up_src_frame = str(inspect_cfg["up_direction"]["frame"])
    p1_src = as_vec3(inspect_cfg["position"])
    d1_src, up_src = require_unit_orthogonal(
        as_vec3(inspect_cfg["direction"]),
        as_vec3(inspect_cfg["up_direction"]),
    )
    p1_world = express_point(
        p1_src, p1_src_frame, world_frame, t_world_base, world_frame, base_frame
    )
    d1_world = vec_normalize(
        express_vector(
            d1_src, d1_src_frame, world_frame, t_world_base, world_frame, base_frame
        ),
        "D1 world",
    )
    up_world = vec_normalize(
        express_vector(
            up_src, up_src_frame, world_frame, t_world_base, world_frame, base_frame
        ),
        "preferred up world",
    )
    d1_world, up_world = require_unit_orthogonal(d1_world, up_world)
    p1_base = express_point(
        p1_world, world_frame, base_frame, t_world_base, world_frame, base_frame
    )
    d1_base = vec_normalize(
        express_vector(
            d1_world, world_frame, base_frame, t_world_base, world_frame, base_frame
        ),
        "D1 base",
    )
    up_base = vec_normalize(
        express_vector(
            up_world, world_frame, base_frame, t_world_base, world_frame, base_frame
        ),
        "preferred up base",
    )
    views = required_cylinder_views(inspect_cfg["faces"], radius, height)
    tcp_t_object = grasp_tcp_object_pose(cfg)
    working_frame = world_frame
    targets = {
        name: compute_view_target(
            view,
            p1=p1_world,
            frame=working_frame,
            inspection_direction=d1_world,
            inspection_up=up_world,
            tcp_t_object=tcp_t_object,
        )
        for name, view in views.items()
    }
    return {
        "config_file": path,
        "radius": radius,
        "height": height,
        "world_frame": world_frame,
        "base_frame": base_frame,
        "working_frame": working_frame,
        "t_world_base": t_world_base,
        "p1_source_frame": p1_src_frame,
        "d1_source_frame": d1_src_frame,
        "up_source_frame": up_src_frame,
        "p1": p1_world,
        "p1_world": p1_world,
        "p1_base": p1_base,
        "frame": working_frame,
        "direction": d1_world,
        "d1_world": d1_world,
        "d1_base": d1_base,
        "up": up_world,
        "up_world": up_world,
        "up_base": up_base,
        "tcp_t_object": tcp_t_object,
        "views": views,
        "targets": targets,
    }


def validate_stage6_geometry(data: dict[str, Any]) -> list[str]:
    """Return failing assertion messages. Empty list means PASS."""
    failures: list[str] = []
    radius = float(data["radius"])
    height = float(data["height"])
    p1 = data["p1_world"]
    if data.get("working_frame") != data.get("world_frame"):
        failures.append(
            f"working_frame {data.get('working_frame')!r} != world {data.get('world_frame')!r}"
        )
    try:
        require_unit_orthogonal(data["d1_world"], data["up_world"])
    except InspectionViewError as exc:
        failures.append(str(exc))
    targets: dict[str, InspectionViewTarget] = data["targets"]
    required = {"side_pos_y", "side_neg_y", "top_circle"}
    if not required.issubset(set(targets)):
        failures.append(f"required view set mismatch: {sorted(targets)}")
    if "top_circle" in data.get("views", {}):
        top_n = data["views"]["top_circle"].normal_in_object
        if top_n[2] < _CAP_Z_ABS_MIN:
            failures.append(f"top_circle must remain object +Z, got {top_n}")

    offset_checks: list[tuple[str, float]] = [
        ("side_pos_y", radius),
        ("side_neg_y", radius),
        ("top_circle", 0.5 * height),
    ]
    if "bottom_circle" in targets:
        offset_checks.append(("bottom_circle", 0.5 * height))
    for name, expected in offset_checks:
        if name not in targets:
            continue
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
    if "bottom_circle" in data.get("views", {}):
        bottom = data["views"]["bottom_circle"]
        if bottom.normal_in_object[2] > -_CAP_Z_ABS_MIN:
            failures.append(
                f"bottom_circle must be object -Z, got {bottom.normal_in_object}"
            )
        if abs(bottom.center_in_object[2] + 0.5 * height) > _CENTER_TOL_M:
            failures.append(
                f"bottom_circle center {bottom.center_in_object} != [0,0,-H/2]"
            )

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
        "P1 semantic: INSPECTION VIEW / SURFACE CENTER TARGET",
        f"working frame: {data['working_frame']}",
        f"P1 source frame: {data['p1_source_frame']}",
        f"D1 source frame: {data['d1_source_frame']}",
        f"up source frame: {data['up_source_frame']}",
        f"target P1 world: {format_vec(data['p1_world'])}",
        f"P1 base (derived): {format_vec(data['p1_base'])}",
        f"target D1 world: {format_vec(data['d1_world'])}",
        f"D1 base (derived): {format_vec(data['d1_base'])}",
        f"target preferred up world: {format_vec(data['up_world'])}",
        f"up base (derived): {format_vec(data['up_base'])}",
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
                f"target P1 world: {format_vec(data['p1_world'])}",
                f"actual inspected center world: {format_vec(actual_view_center(target.object_pose, view.center_in_object))}",
                f"P1 error: {target.view_center_error_m:.3e} m",
                f"target D1 world: {format_vec(data['d1_world'])}",
                f"actual face normal world: {format_vec(actual_view_normal(target.object_pose, view.normal_in_object))}",
                f"D1 error: {target.normal_angle_error_deg:.3e} deg",
                f"target preferred up world: {format_vec(data['up_world'])}",
                f"actual up world: {format_vec(actual_view_up(target.object_pose, view.up_in_object))}",
                f"canonical up error: {target.canonical_up_error_deg:.3e} deg",
                f"Object target pose: {format_pose(target.object_pose, target.frame)}",
                f"TCP target pose: {format_pose(target.tcp_pose, target.frame)}",
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
