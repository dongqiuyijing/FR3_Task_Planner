#!/usr/bin/env python3
"""DUAL-6B RViz preview only. Publishes /display_planned_path. Never executes."""

from __future__ import annotations

import math
import time

import rclpy
from builtin_interfaces.msg import Duration
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_A = ["arm_a_j1", "arm_a_j2", "arm_a_j3", "arm_a_j4", "arm_a_j5", "arm_a_j6"]
ARM_B = ["arm_b_j1", "arm_b_j2", "arm_b_j3", "arm_b_j4", "arm_b_j5", "arm_b_j6"]
GRIPPERS = ["arm_a_gripper_joint", "arm_b_gripper_joint"]

HOME_A = [
    -2.271411675804,
    -1.642863047968,
    -1.869634009789,
    -2.871167788011,
    -0.003375517130,
    0.839900045153,
]
Q_A = 0.0
Q_B = 0.083

# DUAL-6B PREFERRED J1/J6 = MINIMUM J1 TESTED
WAYPOINTS = {
    "Handover": [
        -1.662443436194,
        -2.302791191481,
        -1.754163563924,
        -0.655430550582,
        1.570792654562,
        -0.091643436201,
    ],
    "I1_side_pos_y": [
        -2.332182311944,
        -2.057041326330,
        -2.358977747706,
        -0.296366232595,
        1.570792653603,
        0.024015688849,
    ],
    "I2_side_neg_y": [
        -2.603657484662,
        -2.412203154165,
        -1.512507239020,
        -1.412085755649,
        1.906398963814,
        2.784896625388,
    ],
    "I3_original_top_circle": [
        -1.963018942902,
        -3.254950640807,
        -0.345597603116,
        -2.682636127041,
        1.178573547341,
        2.356200920792,
    ],
}

VMAX = [3.15, 3.15, 3.15, 3.20, 3.20, 3.20]
MAX_STEP = 0.02


def densify(a, b):
    jump = max(abs(b[i] - a[i]) for i in range(6))
    n = max(1, int(math.ceil(jump / MAX_STEP)))
    out = []
    for s in range(1, n + 1):
        t = s / n
        out.append([a[i] + t * (b[i] - a[i]) for i in range(6)])
    return out


def hold(q, count):
    return [list(q) for _ in range(count)]


def to_duration(t):
    sec = int(t)
    return Duration(sec=sec, nanosec=int((t - sec) * 1e9))


def append_points(jt, pts, t, dt_move=None):
    for i, q in enumerate(pts):
        if not jt.points:
            dt = 0.0
        elif dt_move is None:
            dt = 0.05
        else:
            prev = jt.points[-1].positions
            dt = max(abs(q[j] - prev[j]) / VMAX[j] for j in range(6))
            dt = max(dt, 0.02)
        t += dt
        p = JointTrajectoryPoint()
        p.positions = list(q)
        p.time_from_start = to_duration(t)
        jt.points.append(p)
    return t


class Preview(Node):
    def __init__(self):
        super().__init__("dual_6b_rviz_preview")
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub = self.create_publisher(DisplayTrajectory, "/display_planned_path", qos)
        self.disp = self._build()
        self.timer = self.create_timer(2.0, self._tick)
        self.get_logger().info(
            "DUAL-6B RViz preview: ONE trajectory Handover(hold)→I1→I2→I3. "
            "Arm A stays Home. PLAN DISPLAY ONLY. No execute."
        )
        self._tick()

    def _build(self):
        names = ["Handover", "I1_side_pos_y", "I2_side_neg_y", "I3_original_top_circle"]
        seq = [WAYPOINTS[n] for n in names]
        disp = DisplayTrajectory()
        disp.model_id = "fairino3_dual_robot"
        start = JointState()
        start.name = ARM_A + ARM_B + GRIPPERS
        start.position = HOME_A + seq[0] + [Q_A, Q_B]
        disp.trajectory_start.joint_state = start

        # Humble MotionPlanning often shows only trajectory[0], and plays each
        # waypoint at State Display Time (0.05 s). Handover was a single start
        # sample, so it vanished. Keep one concatenated path and dwell at H.
        jt = JointTrajectory()
        jt.header.frame_id = "world"
        jt.joint_names = list(ARM_B)
        t = 0.0
        # ~2.0 s dwell at Handover (40 * 0.05 s)
        t = append_points(jt, hold(seq[0], 40), t)
        t = append_points(jt, densify(seq[0], seq[1]), t, dt_move=True)
        t = append_points(jt, hold(seq[1], 16), t)
        t = append_points(jt, densify(seq[1], seq[2]), t, dt_move=True)
        t = append_points(jt, hold(seq[2], 16), t)
        t = append_points(jt, densify(seq[2], seq[3]), t, dt_move=True)
        t = append_points(jt, hold(seq[3], 24), t)
        rt = RobotTrajectory()
        rt.joint_trajectory = jt
        disp.trajectory.append(rt)
        return disp

    def _tick(self):
        self.disp.trajectory_start.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.disp)


def main():
    rclpy.init()
    node = Preview()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
