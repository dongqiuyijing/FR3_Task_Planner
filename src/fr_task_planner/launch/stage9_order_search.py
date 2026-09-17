#!/usr/bin/env python3
"""Run six independent STEP 9 full-task order searches from real Home.

Each permutation launches a fresh complete MTC Task. Same candidate budget.
Ranking uses complete-candidate T then near-fastest joint path length.
NO STEP 7/8 edge-cost addition. NO execution. NO global-optimum claim.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

ORDERS = (
    ("side_pos_y", "side_neg_y", "top_circle"),
    ("side_pos_y", "top_circle", "side_neg_y"),
    ("side_neg_y", "side_pos_y", "top_circle"),
    ("side_neg_y", "top_circle", "side_pos_y"),
    ("top_circle", "side_pos_y", "side_neg_y"),
    ("top_circle", "side_neg_y", "side_pos_y"),
)

LABEL = {
    "side_pos_y": "A",
    "side_neg_y": "B",
    "top_circle": "C",
}

NEAR_FASTEST_S = 0.3


def _order_key(order: tuple[str, str, str]) -> str:
    return f"{LABEL[order[0]]}_{LABEL[order[1]]}_{LABEL[order[2]]}"


def _order_csv(order: tuple[str, str, str]) -> str:
    return ",".join(order)


def _run_order(order: tuple[str, str, str], budget: int, config_file: str) -> int:
    cmd = [
        "ros2",
        "launch",
        "fr_task_planner",
        "mtc_fr3_full_order_test.launch.py",
        f"view_order:={_order_csv(order)}",
        f"solutions_per_order:={budget}",
        "use_mtc_rviz:=false",
        "hold_for_introspection:=false",
        f"config_file:={config_file}",
    ]
    print(f"\n===== ORDER {_order_key(order)} {_order_csv(order)} =====", flush=True)
    subprocess.call(
        ["pkill", "-f", "fr3_mtc_full_order_test"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        return subprocess.call(cmd, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"ORDER {_order_key(order)} timed out after 600 s", flush=True)
        subprocess.call(
            ["pkill", "-f", "fr3_mtc_full_order_test"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return 124


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    # Minimal fallback: only top-level scalars.
    data = {"candidates": []}
    for line in text.splitlines():
        if ":" not in line or line.startswith(" ") or line.startswith("\t"):
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def _collect_valid(orders: list[tuple[str, str, str]]) -> list[dict]:
    valid = []
    for order in orders:
        path = Path(f"/tmp/fr3_step9_{_order_key(order)}.yaml")
        data = _load_yaml(path)
        for cand in data.get("candidates") or []:
            if cand.get("valid"):
                item = dict(cand)
                item["order"] = _order_csv(order)
                item["order_key"] = _order_key(order)
                valid.append(item)
    return valid


def _select(valid: list[dict]) -> tuple[dict | None, list[dict], float | None]:
    if not valid:
        return None, [], None
    t_min = min(float(c["total_predicted_motion_duration"]) for c in valid)
    near = [
        c
        for c in valid
        if float(c["total_predicted_motion_duration"]) - t_min <= NEAR_FASTEST_S
    ]
    near.sort(
        key=lambda c: (
            float(c["total_joint_path_length"]),
            float(c["total_predicted_motion_duration"]),
            c["order_key"],
            int(c.get("candidate_index", 0)),
        )
    )
    return near[0], near, t_min


def _write_summary(results: dict, valid: list[dict], winner: dict | None, near: list[dict], t_min: float | None) -> None:
    path = Path("/tmp/fr3_step9_order_search.yaml")
    lines = [
        "step: 9",
        "preliminary_canonical_winner: true",
        "production_cache: false",
        "global_optimum_claimed: false",
        "step7_8_edge_costs_used: false",
        "roll_sampling: false",
        "execution: false",
        f"solutions_per_order: {results['budget']}",
        f"total_requested: {results['budget'] * 6}",
        f"total_returned: {results['returned']}",
        f"total_valid: {len(valid)}",
        f"near_fastest_band_s: {NEAR_FASTEST_S}",
    ]
    if t_min is not None:
        lines.append(f"t_min: {t_min}")
        lines.append(f"near_fastest_threshold: {t_min + NEAR_FASTEST_S}")
    if winner:
        lines.append("selected:")
        lines.append(f"  order: {winner['order']}")
        lines.append(f"  candidate_index: {winner.get('candidate_index')}")
        lines.append(f"  total_predicted_motion_duration: {winner.get('total_predicted_motion_duration')}")
        lines.append(f"  total_joint_path_length: {winner.get('total_joint_path_length')}")
        lines.append(f"  predicted_with_hold_time: {winner.get('predicted_with_hold_time')}")
        lines.append(f"  digest: {winner.get('digest', '')}")
        lines.append("  note: STEP 9 PRELIMINARY CANONICAL WINNER NOT PRODUCTION CACHE")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {path}", flush=True)


def main() -> int:
    budget = int(os.environ.get("STEP9_SOLUTIONS_PER_ORDER", "5"))
    config = os.environ.get(
        "STAGE4_CONFIG",
        os.path.expanduser("~/fairino_ws/src/fr_control/config/stage4_config.yaml"),
    )
    results = {"budget": budget, "returned": 0, "rcs": {}}
    for order in ORDERS:
        rc = _run_order(order, budget, config)
        results["rcs"][_order_key(order)] = rc
        data = _load_yaml(Path(f"/tmp/fr3_step9_{_order_key(order)}.yaml"))
        results["returned"] += int(data.get("returned", 0) or 0)

    valid = _collect_valid(list(ORDERS))
    winner, near, t_min = _select(valid)
    _write_summary(results, valid, winner, near, t_min)

    print("\n===== STEP 9 BATCH SUMMARY =====")
    print(f"requested total: {budget * 6}")
    print(f"returned: {results['returned']}")
    print(f"valid: {len(valid)}")
    if t_min is not None:
        print(f"T_min: {t_min:.6f} s")
        print(f"near-fastest threshold: {t_min + NEAR_FASTEST_S:.6f} s")
        print("near-fastest set:")
        for c in near:
            print(
                f"  {c['order_key']} cand{c.get('candidate_index')} "
                f"T={float(c['total_predicted_motion_duration']):.6f} "
                f"L={float(c['total_joint_path_length']):.6f}"
            )
    if winner:
        print(
            f"SELECTED: {winner['order_key']} candidate {winner.get('candidate_index')} "
            f"T={float(winner['total_predicted_motion_duration']):.6f} "
            f"L={float(winner['total_joint_path_length']):.6f}"
        )
        print(
            "Best sampled feasible complete-task candidate within the configured "
            "planning budget and canonical inspection orientations."
        )
    else:
        print("SELECTED: NONE")
    print("NO ORDER RANKING FROM STEP 7/8 EDGE COSTS")
    print("NO GLOBAL OPTIMUM CLAIMED")
    return 0 if winner else 1


if __name__ == "__main__":
    sys.exit(main())
