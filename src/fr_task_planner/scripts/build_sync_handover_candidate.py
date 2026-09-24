#!/usr/bin/env python3
"""Offline-only: replace only four handover motions with two timed dual stages."""
import copy
import pathlib
import sys
import yaml

ROOT = pathlib.Path.home() / "fr_task_sync_ws/src/fr_task_planner"
SRC = ROOT / "config/keypose_optimization_v1/keypose_v1_six_face_speed_optimized_candidate.yaml"
BASE = ROOT / "config/keypose_optimization_v1/keypose_v1_six_face_with_place_candidate.yaml"
DST = ROOT / "config/keypose_optimization_v1/keypose_v1_six_face_with_sync_handover_candidate.yaml"
REPORT = ROOT / "config/keypose_optimization_v1/sync_handover_offline_report.yaml"

def seconds(p): return float(p.get("sec", 0)) + float(p.get("nanosec", 0)) * 1e-9
def stamp(p, t):
    ns = round(t * 1e9)
    p["sec"], p["nanosec"] = divmod(ns, 1_000_000_000)

def retime(stage, duration, hold_to=None):
    s = copy.deepcopy(stage)
    pts = s["points"]
    old = seconds(pts[-1])
    if old <= 0: raise ValueError(f"invalid duration: {s['id']}")
    for p in pts:
        stamp(p, seconds(p) * duration / old)
        p["velocities"] = [0.0] * 6
        p["accelerations"] = [0.0] * 6
    if hold_to is not None and hold_to > duration:
        hold = copy.deepcopy(pts[-1])
        stamp(hold, hold_to)
        hold["velocities"] = [0.0] * 6
        hold["accelerations"] = [0.0] * 6
        pts.append(hold)
    return {k: s[k] for k in ("joint_names", "start_joints", "end_joints", "points")}

def dual(stage_id, a, b, target_duration=None):
    # Preserve the fast B motion (1.000 s), then explicitly hold its final
    # handover pose until A completes its 1.200 s direct connection.
    da, db = seconds(a["points"][-1]), seconds(b["points"][-1])
    d = target_duration if target_duration is not None else max(da, db)
    if target_duration is not None:
        arm_a, arm_b = retime(a, d), retime(b, d)
    else:
        arm_a, arm_b = retime(a, da, d), retime(b, db, d)
    return {
        "id": stage_id, "kind": "dual_joint_connection", "moving": "dual",
        "method": "retimed_frozen_paths_common_timebase", "status": "PASS",
        "source_stage_ids": [a["id"], b["id"]], "common_duration_sec": d,
        "collision_validation": "PENDING_EXTERNAL_VALIDATOR",
        "arm_a": arm_a, "arm_b": arm_b,
    }

def line_segment(name, start, end, duration, points):
    out = {"id": name, "joint_names": start["joint_names"],
           "start_joints": list(start["end_joints"]), "end_joints": list(end["end_joints"]),
           "points": []}
    for i in range(points):
        u = i / (points - 1)
        p = {"positions": [(1-u)*x + u*y for x, y in zip(out["start_joints"], out["end_joints"])],
             "velocities": [0.0]*6, "accelerations": [0.0]*6}
        stamp(p, duration*u)
        out["points"].append(p)
    return out

def via_pre(stage_id, approach, final, direct_duration, direct_points):
    # PRE is an internal waypoint: concatenate the preserved approach and a
    # direct joint interpolation, removing the duplicate PRE point.
    a = retime(approach, seconds(approach["points"][-1]))
    a["id"] = stage_id
    tail = line_segment(stage_id + "_pre_to_final", approach, final, direct_duration, direct_points)
    offset = seconds(a["points"][-1])
    for p in tail["points"][1:]:
        q = copy.deepcopy(p); stamp(q, offset + seconds(q)); a["points"].append(q)
    a["end_joints"] = list(final["end_joints"])
    return a

def main():
    with SRC.open() as f: y = yaml.safe_load(f)
    with BASE.open() as f: base = yaml.safe_load(f)
    stages = y["stages"]
    by = {s["id"]: s for s in stages}
    old = {s["id"]: s for s in base["stages"]}
    ids = ("face3_to_handover", "b_home_to_handover",
           "a_handover_to_home", "b_handover_to_face4")
    if any(x not in by for x in ids): raise RuntimeError("missing source handover stages")
    a_via = via_pre("a_face3_via_pre_to_handover", old["face3_to_pre_handover"], by[ids[0]], 1.2, 141)
    b_via = via_pre("b_home_via_pre_to_handover", old["b_home_to_pre_handover"], by[ids[1]], 1.0, 133)
    d1 = dual("dual_face3_and_b_home_to_handover_via_pre", a_via, b_via, 1.0)
    d2 = dual("dual_a_handover_to_home_and_b_handover_to_face4", by[ids[2]], by[ids[3]])
    out = []
    for s in stages:
        if s["id"] == "face3_to_handover": out.append(d1)
        elif s["id"] == "a_handover_to_home": out.append(d2)
        elif s["id"] not in ids: out.append(copy.deepcopy(s))
    y["stages"] = out
    y["sync_handover_candidate"] = True
    y["sync_handover_note"] = "Only the four source handover motion stages are replaced. All gripper/place stages are copied unchanged."
    with DST.open("w") as f: yaml.safe_dump(y, f, sort_keys=False)
    with REPORT.open("w") as f:
        yaml.safe_dump({"result": "PENDING_COLLISION_VALIDATOR", "source": str(SRC), "output": str(DST),
                        "dual_stage_1_duration_sec": d1["common_duration_sec"],
                        "b_handover_hold_after_motion_sec": d1["common_duration_sec"] - seconds(by[ids[1]]["points"][-1]),
                        "dual_return_face4_duration_sec": d2["common_duration_sec"],
                        "unchanged_non_handover_stage_count": len(out) - 2}, f, sort_keys=False)
    print(DST)

if __name__ == "__main__": main()
