"""DUAL-5-S Arm B remaining +X/-X side targets.

Reuses STEP 6 cylinder side geometry and STEP12C P1/D1/up. Does not change
inspection_view_geometry.py required-view naming, so STEP 6/12 callers stay
on +Y/-Y only.

Physical identity is the object-local normal, not the display name.
"""

from __future__ import annotations

from typing import Any, Sequence

from inspection_view_geometry import (
    InspectionView,
    InspectionViewTarget,
    InspectionViewError,
    apply_step12c_tilted_geometry,
    compute_view_target,
    cylinder_center_in_object,
    load_stage6_geometry,
    vec_norm,
    vec_sub,
)
from fr_control.inspection_poses import vec_normalize


_RADIAL_Z_ABS_MAX = 1e-6
_CAP_Z_ABS_MIN = 1.0 - 1e-6
_AXIS_DOT_MIN = 0.9

# User-required six-face assignment. Physical IDs, not display names.
ARM_A_REQUIRED = ("+Y", "-Y", "-Z")
ARM_B_REQUIRED = ("+X", "-X", "+Z")
ALL_SIX = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
ARM_B_ORDER_NAMES = ("side_pos_x", "side_neg_x", "top_circle")
NAME_TO_PHYSICAL = {
    "side_pos_x": "+X",
    "side_neg_x": "-X",
    "side_pos_y": "+Y",
    "side_neg_y": "-Y",
    "top_circle": "+Z",
    "bottom_circle": "-Z",
}


def physical_face_id_from_normal(normal_in_object: Sequence[float]) -> str:
    """Map an object-local face normal to a six-face physical ID."""
    normal = vec_normalize(normal_in_object, "normal_in_object")
    if abs(normal[2]) <= _RADIAL_Z_ABS_MAX:
        if normal[0] >= _AXIS_DOT_MIN and abs(normal[1]) < 0.1:
            return "+X"
        if normal[0] <= -_AXIS_DOT_MIN and abs(normal[1]) < 0.1:
            return "-X"
        if normal[1] >= _AXIS_DOT_MIN and abs(normal[0]) < 0.1:
            return "+Y"
        if normal[1] <= -_AXIS_DOT_MIN and abs(normal[0]) < 0.1:
            return "-Y"
        return "UNKNOWN"
    if normal[2] >= _CAP_Z_ABS_MIN:
        return "+Z"
    if normal[2] <= -_CAP_Z_ABS_MIN:
        return "-Z"
    return "UNKNOWN"


def make_radial_side_view(
    name: str, normal_in_object: Sequence[float], radius: float
) -> InspectionView:
    """Same side model as +Y/-Y: center = radius * n, up = object +Z."""
    normal = vec_normalize(normal_in_object, "normal_in_object")
    if abs(normal[2]) > _RADIAL_Z_ABS_MAX:
        raise InspectionViewError(
            f"{name} is not a radial cylinder side; normal={normal}"
        )
    return InspectionView(
        name=name,
        normal_in_object=normal,
        center_in_object=cylinder_center_in_object(normal, radius, 1.0),
        up_in_object=(0.0, 0.0, 1.0),
    )


def make_side_pos_x_view(radius: float) -> InspectionView:
    """Original object +X circumferential inspection region."""
    return make_radial_side_view("side_pos_x", (1.0, 0.0, 0.0), radius)


def make_side_neg_x_view(radius: float) -> InspectionView:
    """Original object -X circumferential inspection region."""
    return make_radial_side_view("side_neg_x", (-1.0, 0.0, 0.0), radius)


def compute_side_x_targets(data: dict[str, Any]) -> dict[str, InspectionViewTarget]:
    """Canonical preferred-roll +X/-X targets at the same P1/D1 as STEP12C."""
    radius = float(data["radius"])
    views = {
        "side_pos_x": make_side_pos_x_view(radius),
        "side_neg_x": make_side_neg_x_view(radius),
    }
    return {
        name: compute_view_target(
            view,
            p1=data["p1_world"],
            frame=data["working_frame"],
            inspection_direction=data["d1_world"],
            inspection_up=data["up_world"],
            tcp_t_object=data["tcp_t_object"],
        )
        for name, view in views.items()
    }


def load_step12c_geometry_with_side_x(
    config_file: str | None = None,
) -> dict[str, Any]:
    """STEP12C P1/D1/up plus Arm B remaining X-side targets. Does not rewrite YAML."""
    data = apply_step12c_tilted_geometry(load_stage6_geometry(config_file))
    data["side_x_targets"] = compute_side_x_targets(data)
    return data


def _name_normal_mismatch(
    names: Sequence[str] | None, ids: Sequence[str]
) -> bool:
    if names is None:
        return False
    if len(names) != len(ids):
        return True
    for name, fid in zip(names, ids):
        expected = NAME_TO_PHYSICAL.get(str(name))
        if expected is None or expected != fid:
            return True
    return False


def validateSixFaceAssignment(
    arm_a: Sequence[Sequence[float]],
    arm_b: Sequence[Sequence[float]],
    *,
    arm_a_names: Sequence[str] | None = None,
    arm_b_names: Sequence[str] | None = None,
    expected_a: Sequence[str] = ARM_A_REQUIRED,
    expected_b: Sequence[str] = ARM_B_REQUIRED,
) -> dict[str, Any]:
    """Gate six-face coverage by physical normals, not display names.

    FAIL reasons:
      DUPLICATED PHYSICAL FACE
      MISSING PHYSICAL FACE
      UNKNOWN PHYSICAL FACE
      WRONG ORDER
      WRONG PHYSICAL NORMAL
    """
    a_ids = [physical_face_id_from_normal(n) for n in arm_a]
    b_ids = [physical_face_id_from_normal(n) for n in arm_b]
    unknown = [fid for fid in a_ids + b_ids if fid == "UNKNOWN"]
    duplicates = sorted(set(a_ids) & set(b_ids) - {"UNKNOWN"})
    covered = [fid for fid in a_ids + b_ids if fid != "UNKNOWN"]
    missing = [fid for fid in ALL_SIX if fid not in covered]
    wrong_normal = _name_normal_mismatch(arm_a_names, a_ids) or _name_normal_mismatch(
        arm_b_names, b_ids
    )
    order_a_ok = tuple(a_ids) == tuple(expected_a)
    order_b_ok = tuple(b_ids) == tuple(expected_b)

    reason = ""
    pass_ok = True
    if unknown:
        pass_ok = False
        reason = "UNKNOWN PHYSICAL FACE"
    elif wrong_normal:
        pass_ok = False
        reason = "WRONG PHYSICAL NORMAL"
    elif duplicates:
        pass_ok = False
        reason = "DUPLICATED PHYSICAL FACE"
    elif missing:
        pass_ok = False
        reason = "MISSING PHYSICAL FACE"
    elif not order_a_ok or not order_b_ok:
        pass_ok = False
        reason = "WRONG ORDER"

    return {
        "pass": pass_ok,
        "reason": reason,
        "arm_a": a_ids,
        "arm_b": b_ids,
        "duplicates": duplicates,
        "missing": missing,
        "unknown": unknown,
        "order_a_ok": order_a_ok,
        "order_b_ok": order_b_ok,
        "wrong_normal": wrong_normal,
    }


def assert_side_x_not_y(
    x_targets: dict[str, InspectionViewTarget],
    y_targets: dict[str, InspectionViewTarget],
) -> None:
    """Rename-only is not a correction: +X must not equal +Y pose."""
    pos_x = x_targets["side_pos_x"].object_pose
    neg_x = x_targets["side_neg_x"].object_pose
    pos_y = y_targets["side_pos_y"].object_pose
    neg_y = y_targets["side_neg_y"].object_pose
    for left, right, label in (
        (pos_x, pos_y, "+X vs +Y"),
        (neg_x, neg_y, "-X vs -Y"),
        (pos_x, neg_y, "+X vs -Y"),
        (neg_x, pos_y, "-X vs +Y"),
    ):
        q_dot = abs(
            left.orientation.x * right.orientation.x
            + left.orientation.y * right.orientation.y
            + left.orientation.z * right.orientation.z
            + left.orientation.w * right.orientation.w
        )
        same_ori = q_dot > 0.999
        same_pos = vec_norm(
            vec_sub(
                (left.position.x, left.position.y, left.position.z),
                (right.position.x, right.position.y, right.position.z),
            )
        ) <= 1e-9
        if same_ori and same_pos:
            raise InspectionViewError(
                f"SIDE X TARGET GEOMETRY AMBIGUOUS: {label} object poses are identical"
            )


def format_pose_yaml(pose) -> dict[str, list[float]]:
    p = pose.position
    q = pose.orientation
    return {
        "xyz": [float(p.x), float(p.y), float(p.z)],
        "xyzw": [float(q.x), float(q.y), float(q.z), float(q.w)],
    }


def generate_side_x_rolls(data: dict[str, Any], roll_step_deg: float = 30.0):
    """Enumerate D1-roll candidates for +X/-X. Canonical roll is included first.

    Does not change P1, D1, physical face IDs, or T_tcpB_object.
    """
    from inspection_view_geometry import generate_roll_candidates

    tcp_t_object = data["tcp_t_object"]
    out = {}
    for name in ("side_pos_x", "side_neg_x"):
        out[name] = generate_roll_candidates(
            data["side_x_targets"][name], tcp_t_object, roll_step_deg
        )
    return out

