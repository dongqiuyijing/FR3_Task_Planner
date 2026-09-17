#!/usr/bin/env python3
"""STEP 10: geometry unit tests then Lift-scene roll / IK enumeration.

No order search. No Lift→View planning. No execution.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import unittest
from pathlib import Path


def _run_unit_tests() -> int:
    test_file = Path(__file__).resolve().parents[1] / "test" / "test_inspection_roll_candidates.py"
    step6 = Path(__file__).resolve().parents[1] / "test" / "test_inspection_view_geometry.py"
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.discover(str(test_file.parent), pattern=test_file.name))
    suite.addTests(loader.discover(str(step6.parent), pattern=step6.name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STEP 10 roll / IK candidates")
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument("--roll-step-deg", default="30.0")
    parser.add_argument("--max-ik-solutions-per-pose", default="8")
    parser.add_argument("--min-ik-solution-distance", default="0.1")
    parser.add_argument("--output-dir", default="/tmp")
    args = parser.parse_args(argv)

    if not args.skip_unit_tests:
        unit = _run_unit_tests()
        if unit != 0:
            print("STEP 10 FAIL: geometry unit tests")
            return unit

    cmd = [
        "ros2",
        "launch",
        "fr_task_planner",
        "mtc_fr3_roll_candidate_test.launch.py",
        f"roll_step_deg:={args.roll_step_deg}",
        f"max_ik_solutions_per_pose:={args.max_ik_solutions_per_pose}",
        f"min_ik_solution_distance:={args.min_ik_solution_distance}",
        f"output_dir:={args.output_dir}",
        "hold_for_introspection:=false",
    ]
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
