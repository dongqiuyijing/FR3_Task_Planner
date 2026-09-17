#!/usr/bin/env python3
"""Run six independent STEP 8 directed-edge tests from real Home.

Each edge launches a fresh complete MTC Task. No execute. No order ranking.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

EDGES = (
    ("side_pos_y", "side_neg_y"),
    ("side_pos_y", "top_circle"),
    ("side_neg_y", "side_pos_y"),
    ("side_neg_y", "top_circle"),
    ("top_circle", "side_pos_y"),
    ("top_circle", "side_neg_y"),
)

LABEL = {
    "side_pos_y": "+Y",
    "side_neg_y": "-Y",
    "top_circle": "TOP",
}


def _run_edge(source: str, target: str, config_file: str) -> int:
    cmd = [
        "ros2",
        "launch",
        "fr_task_planner",
        "mtc_fr3_view_transition_test.launch.py",
        f"source_view:={source}",
        f"target_view:={target}",
        "use_mtc_rviz:=false",
        "hold_for_introspection:=false",
        f"config_file:={config_file}",
    ]
    print(f"\n===== EDGE {source} → {target} =====", flush=True)
    return subprocess.call(cmd)


def _collect_yaml() -> dict:
    out = {}
    for source, target in EDGES:
        path = Path(f"/tmp/fr3_step8_{source}_{target}.yaml")
        if not path.is_file():
            out[(source, target)] = None
            continue
        data = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line or line.startswith(" ") or line.startswith("\t"):
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
        out[(source, target)] = data
    return out


def _write_matrix(results: dict[tuple[str, str], int]) -> None:
    rows = _collect_yaml()
    matrix_path = Path("/tmp/fr3_step8_transition_matrix.yaml")
    lines = [
        "step: 8",
        "order_ranking: false",
        "roll_sampling: false",
        "execution: false",
        "edges:",
    ]
    for source, target in EDGES:
        rc = results[(source, target)]
        row = rows.get((source, target)) or {}
        lines.append(f"  - source: {source}")
        lines.append(f"    target: {target}")
        lines.append(f"    reachable: {'true' if rc == 0 else 'false'}")
        lines.append(f"    return_code: {rc}")
        if row:
            for key in (
                "complete_solution_count",
                "predicted_duration",
                "joint_path_length",
                "trajectory_point_count",
                "planning_computation_time",
            ):
                if key in row:
                    lines.append(f"    {key}: {row[key]}")
    matrix_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {matrix_path}", flush=True)

    print("\nTRANSITION MATRIX")
    print("                   TO")
    print("             +Y       -Y       TOP")
    names = ("side_pos_y", "side_neg_y", "top_circle")
    for src in names:
        cells = []
        for tgt in names:
            if src == tgt:
                cells.append("   -   ")
            else:
                cells.append(" PASS  " if results[(src, tgt)] == 0 else " FAIL  ")
        print(f"FROM {LABEL[src]:<4} {''.join(cells)}")
    print("\nNO ORDER RANKING PERFORMED")


def main() -> int:
    config = os.environ.get(
        "STAGE4_CONFIG",
        os.path.expanduser("~/fairino_ws/src/fr_control/config/stage4_config.yaml"),
    )
    results = {}
    failed = 0
    for source, target in EDGES:
        rc = _run_edge(source, target, config)
        results[(source, target)] = rc
        if rc != 0:
            failed += 1
    _write_matrix(results)
    print(f"\n{6 - failed}/6 directed edges PASS")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
