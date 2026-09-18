#!/usr/bin/env python3
"""STEP 11E: dense top_circle roll / IK feasibility search.

Geometry unit tests first, then the 5° sweep launch. No STEP 12.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import unittest
from pathlib import Path


def _run_unit_tests() -> int:
    test_dir = Path(__file__).resolve().parents[1] / "test"
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.discover(str(test_dir), pattern="test_inspection_view_geometry.py"))
    suite.addTests(loader.discover(str(test_dir), pattern="test_inspection_roll_candidates.py"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STEP 11E dense top_circle search")
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument("--roll-step-deg", default="5.0")
    parser.add_argument("--independent-sweeps", default="3")
    parser.add_argument("--max-ik-solutions-per-pose", default="16")
    parser.add_argument("--headless", default="true")
    args = parser.parse_args(argv)

    if not args.skip_unit_tests:
        unit = _run_unit_tests()
        if unit != 0:
            print("STEP 11E FAIL: geometry unit tests")
            return unit

    cmd = [
        "ros2",
        "launch",
        "fr_task_planner",
        "mtc_fr3_dense_top_circle_search.launch.py",
        f"roll_step_deg:={args.roll_step_deg}",
        f"independent_sweeps:={args.independent_sweeps}",
        f"max_ik_solutions_per_pose:={args.max_ik_solutions_per_pose}",
        f"headless:={args.headless}",
    ]
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
