#!/usr/bin/env python3
"""STEP 9A: independent fresh full-task trials per canonical order.

Each trial launches a new process and a new MTC Task. task.plan(1).
Same budget for every order. Failures count. No STEP 9 winner injection.
No execution. No roll sampling. No global-optimum claim.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import time
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
STEP9_HISTORICAL = {
    "order_key": "C_B_A",
    "duration": 15.5884,
    "path_length": 19.2208,
    "digest": "367dec03ec9fccad",
}


def _order_key(order: tuple[str, str, str]) -> str:
    return f"{LABEL[order[0]]}_{LABEL[order[1]]}_{LABEL[order[2]]}"


def _order_csv(order: tuple[str, str, str]) -> str:
    return ",".join(order)


def _trial_path(order: tuple[str, str, str], trial: int) -> Path:
    return Path(f"/tmp/fr3_step9a_{_order_key(order)}_trial_{trial}.yaml")


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return {}


def _run_trial(order: tuple[str, str, str], trial: int, config_file: str) -> int:
    out = _trial_path(order, trial)
    if out.exists():
        out.unlink()
    cmd = [
        "ros2",
        "launch",
        "fr_task_planner",
        "mtc_fr3_full_order_test.launch.py",
        f"view_order:={_order_csv(order)}",
        "solutions_per_order:=1",
        f"trial_index:={trial}",
        f"diagnostic_output_path:={out}",
        "use_mtc_rviz:=false",
        "hold_for_introspection:=false",
        f"config_file:={config_file}",
    ]
    print(
        f"\n===== ORDER {_order_key(order)} trial {trial} {_order_csv(order)} =====",
        flush=True,
    )
    try:
        return subprocess.call(cmd, timeout=180)
    except subprocess.TimeoutExpired:
        print(f"ORDER {_order_key(order)} trial {trial} timed out", flush=True)
        return 124


def _candidate_from_trial(order: tuple[str, str, str], trial: int, data: dict) -> dict | None:
    cands = data.get("candidates") or []
    if not cands:
        return None
    cand = dict(cands[0])
    if not cand.get("valid"):
        return None
    segs = cand.get("segment_metrics") or {}
    prefix = segs.get("home_to_pregrasp") or {}
    cand["order"] = _order_csv(order)
    cand["order_key"] = _order_key(order)
    cand["trial_index"] = int(data.get("trial_index", trial))
    cand["prefix_duration"] = float(prefix.get("duration", 0.0) or 0.0)
    cand["prefix_path"] = float(prefix.get("path_length", 0.0) or 0.0)
    return cand


def _stats(values: list[float]) -> dict:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


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
            int(c.get("trial_index", 0)),
        )
    )
    return near[0], near, t_min


def main() -> int:
    runs = int(os.environ.get("STEP9A_RUNS_PER_ORDER", "5"))
    config = os.environ.get(
        "STAGE4_CONFIG",
        os.path.expanduser("~/fairino_ws/src/fr_control/config/stage4_config.yaml"),
    )
    wall0 = time.monotonic()
    records: dict[str, list[dict]] = {}
    valid: list[dict] = []

    for order in ORDERS:
        key = _order_key(order)
        records[key] = []
        for trial in range(runs):
            rc = _run_trial(order, trial, config)
            data = _load_yaml(_trial_path(order, trial))
            data["launch_rc"] = rc
            records[key].append(data)
            cand = _candidate_from_trial(order, trial, data)
            if cand:
                valid.append(cand)

    winner, near, t_min = _select(valid)
    unique_all = {c.get("digest") for c in valid if c.get("digest")}
    wall = time.monotonic() - wall0

    lines = [
        "step: 9a",
        "preliminary_canonical_winner: true",
        "production_cache: false",
        "global_optimum_claimed: false",
        "step9_historical_injected: false",
        "roll_sampling: false",
        "execution: false",
        f"runs_per_order: {runs}",
        f"total_trials: {runs * 6}",
        f"valid_candidates: {len(valid)}",
        f"unique_motion_candidates: {len(unique_all)}",
        f"batch_wall_time: {wall}",
        f"near_fastest_band_s: {NEAR_FASTEST_S}",
    ]
    if t_min is not None:
        lines.append(f"t_min: {t_min}")
        lines.append(f"near_fastest_threshold: {t_min + NEAR_FASTEST_S}")
    if winner:
        lines.append("selected:")
        lines.append(f"  order: {winner['order']}")
        lines.append(f"  trial_index: {winner.get('trial_index')}")
        lines.append(f"  digest: {winner.get('digest', '')}")
        lines.append(
            f"  total_predicted_motion_duration: {winner.get('total_predicted_motion_duration')}"
        )
        lines.append(f"  total_joint_path_length: {winner.get('total_joint_path_length')}")
        lines.append(f"  predicted_with_hold_time: {winner.get('predicted_with_hold_time')}")
        lines.append("  note: STEP 9A PRELIMINARY CANONICAL WINNER NOT PRODUCTION CACHE")
    Path("/tmp/fr3_step9a_multi_run_search.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("\n===== STEP 9A BATCH SUMMARY =====")
    print(f"runs_per_order: {runs}")
    print(f"total trials: {runs * 6}")
    print(f"valid: {len(valid)}")
    print(f"unique motion candidates: {len(unique_all)}")
    print("ORDER       SUCCESS   UNIQUE   MIN T      MEDIAN T   MAX T")
    for order in ORDERS:
        key = _order_key(order)
        trials = records[key]
        order_valid = [
            _candidate_from_trial(order, i, d)
            for i, d in enumerate(trials)
        ]
        order_valid = [c for c in order_valid if c]
        ts = [float(c["total_predicted_motion_duration"]) for c in order_valid]
        unique = {c.get("digest") for c in order_valid if c.get("digest")}
        st = _stats(ts)
        print(
            f"{key:<10}  {len(order_valid)}/{len(trials)}      {len(unique):<6}  "
            f"{st['min'] if st['min'] is not None else float('nan'):.4f}     "
            f"{st['median'] if st['median'] is not None else float('nan'):.4f}     "
            f"{st['max'] if st['max'] is not None else float('nan'):.4f}"
        )
    if t_min is not None:
        print(f"T_min: {t_min:.6f} s")
        print(f"near-fastest threshold: {t_min + NEAR_FASTEST_S:.6f} s")
        print("near-fastest set:")
        for c in near:
            print(
                f"  {c['order_key']} trial{c.get('trial_index')} "
                f"digest={c.get('digest')} "
                f"T={float(c['total_predicted_motion_duration']):.6f} "
                f"L={float(c['total_joint_path_length']):.6f}"
            )
    if winner:
        print(
            f"SELECTED: {winner['order_key']} trial {winner.get('trial_index')} "
            f"digest={winner.get('digest')} "
            f"T={float(winner['total_predicted_motion_duration']):.6f} "
            f"L={float(winner['total_joint_path_length']):.6f}"
        )
        print(
            "Best sampled feasible complete-task candidate within the STEP 9A "
            "canonical-view multi-run planning budget."
        )
    else:
        print("SELECTED: NONE")
    print(f"STEP 9 historical baseline: C-B-A {STEP9_HISTORICAL['duration']} s (NOT used in ranking)")
    print("NO GLOBAL OPTIMUM CLAIMED")
    multi_valid = any(
        sum(1 for d in records[_order_key(o)] if (d.get("candidates") or [{}])[0].get("valid"))
        >= 2
        for o in ORDERS
    )
    print(f"Any order >=2 valid? {multi_valid}")
    print(f"Any distinct motion candidates >=2? {len(unique_all) >= 2}")
    return 0 if winner and len(unique_all) >= 2 and multi_valid else 1


if __name__ == "__main__":
    sys.exit(main())
