#!/usr/bin/env python3
"""STEP 11: one fresh Task per View, Alternatives inside each Task.

No View→View. No order search. No execution.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

VIEWS = ("side_pos_y", "side_neg_y", "top_circle")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STEP 11 endpoint branching")
    parser.add_argument("--max-endpoint-candidates", default="0")
    parser.add_argument("--output-dir", default="/tmp")
    args = parser.parse_args(argv)

    codes = {}
    for view in VIEWS:
        out = Path(args.output_dir) / f"fr3_step11_{view}.yaml"
        cmd = [
            "ros2",
            "launch",
            "fr_task_planner",
            "mtc_fr3_endpoint_branch_test.launch.py",
            f"view_name:={view}",
            f"max_endpoint_candidates:={args.max_endpoint_candidates}",
            f"diagnostic_output_path:={out}",
            "hold_for_introspection:=false",
        ]
        print(" ".join(cmd), flush=True)
        codes[view] = subprocess.call(cmd)

    summary = Path(args.output_dir) / "fr3_step11_endpoint_branching.yaml"
    lines = ["step: 11", "views:"]
    for view in VIEWS:
        path = Path(args.output_dir) / f"fr3_step11_{view}.yaml"
        lines.append(f"  {view}:")
        lines.append(f"    exit_code: {codes[view]}")
        lines.append(f"    yaml: {path}")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {summary}")
    return 0 if all(code == 0 for code in codes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
