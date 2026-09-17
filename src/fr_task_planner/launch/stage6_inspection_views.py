#!/usr/bin/env python3
"""STEP 6 geometry-only inspection-view check. No Gazebo, no planning."""

from __future__ import annotations

import argparse
import sys

from inspection_view_geometry import (
    compute_view_target,
    format_stage6_report,
    load_stage6_geometry,
    tcp_target_from_object,
    validate_stage6_geometry,
)


def main(argv: list[str] | None = None) -> int:
    """Load YAML, print targets, assert geometry, exit."""
    parser = argparse.ArgumentParser(description="STEP 6 cylinder view geometry")
    parser.add_argument(
        "--config-file",
        default=None,
        help="stage4_config.yaml path. Defaults to fr_control workcell path.",
    )
    args = parser.parse_args(argv)
    data = load_stage6_geometry(args.config_file)
    failures = validate_stage6_geometry(data)

    # TEST 7: TCP targets must use T_object * inverse(T_tcp_object).
    for name, target in data["targets"].items():
        recomputed = tcp_target_from_object(target.object_pose, data["tcp_t_object"])
        dx = recomputed.position.x - target.tcp_pose.position.x
        dy = recomputed.position.y - target.tcp_pose.position.y
        dz = recomputed.position.z - target.tcp_pose.position.z
        if (dx * dx + dy * dy + dz * dz) ** 0.5 > 1e-12:
            failures.append(f"{name} TCP target is not T_object * inv(T_tcp_object)")

    # Orientation hook exists so canonical roll is not the only future pose.
    extra = compute_view_target(
        data["views"]["side_pos_y"],
        p1=data["p1"],
        frame=data["frame"],
        inspection_direction=data["direction"],
        inspection_up=data["up"],
        tcp_t_object=data["tcp_t_object"],
        orientation_xyzw=None,
    )
    if extra.view.name != "side_pos_y":
        failures.append("orientation hook lost the view name")

    print(format_stage6_report(data, failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
