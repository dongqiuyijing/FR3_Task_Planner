#!/usr/bin/env python3
"""Run six independent STEP 8 directed-edge tests from real Home.

Each edge launches a fresh complete MTC Task. No execute. No order ranking.

Authoritative result is /tmp/fr3_step8_<src>_<tgt>.yaml
(result / reachable / complete_solution_count), not ros2 launch exit code.
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


def _read_yaml(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.startswith(" ") or line.startswith("\t"):
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def _authoritative_pass(row: dict[str, str] | None) -> tuple[bool, str]:
    if not row:
        return False, "FAIL"
    result = row.get("result", "").upper()
    if result in {"PASS", "FAIL"}:
        return result == "PASS", result
    reachable = row.get("reachable", "").lower()
    try:
        complete = int(float(row.get("complete_solution_count", "0")))
    except ValueError:
        complete = 0
    if reachable == "true" and complete > 0:
        return True, "PASS"
    return False, "FAIL"


def _collect_yaml() -> dict[tuple[str, str], dict[str, str] | None]:
    out: dict[tuple[str, str], dict[str, str] | None] = {}
    for source, target in EDGES:
        out[(source, target)] = _read_yaml(Path(f"/tmp/fr3_step8_{source}_{target}.yaml"))
    return out


def _write_matrix(launch_rcs: dict[tuple[str, str], int]) -> tuple[int, int]:
    rows = _collect_yaml()
    matrix_path = Path("/tmp/fr3_step8_transition_matrix.yaml")
    lines = [
        "step: 8",
        "order_ranking: false",
        "roll_sampling: false",
        "execution: false",
        "authority: edge_yaml",
        "edges:",
    ]
    passed = 0
    failed = 0
    verdicts: dict[tuple[str, str], str] = {}
    for source, target in EDGES:
        row = rows.get((source, target))
        ok, verdict = _authoritative_pass(row)
        verdicts[(source, target)] = verdict
        if ok:
            passed += 1
        else:
            failed += 1
        rc = launch_rcs[(source, target)]
        lines.append(f"  - source: {source}")
        lines.append(f"    target: {target}")
        lines.append(f"    result: {verdict}")
        lines.append(f"    reachable: {'true' if ok else 'false'}")
        lines.append(f"    launch_return_code: {rc}")
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
        else:
            lines.append("    complete_solution_count: 0")
            lines.append("    missing_yaml: true")
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
                cells.append(" PASS  " if verdicts[(src, tgt)] == "PASS" else " FAIL  ")
        print(f"FROM {LABEL[src]:<4} {''.join(cells)}")
    print("\nNO ORDER RANKING PERFORMED")
    print(f"\n{passed}/6 directed edges PASS (authoritative YAML, not launch exit code)")
    return passed, failed


def main() -> int:
    config = os.environ.get(
        "STAGE4_CONFIG",
        os.path.expanduser("~/fairino_ws/src/fr_control/config/stage4_config.yaml"),
    )
    launch_rcs = {}
    for source, target in EDGES:
        yaml_path = Path(f"/tmp/fr3_step8_{source}_{target}.yaml")
        if yaml_path.exists():
            yaml_path.unlink()
        launch_rcs[(source, target)] = _run_edge(source, target, config)
    passed, failed = _write_matrix(launch_rcs)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
