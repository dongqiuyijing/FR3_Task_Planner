#!/usr/bin/env python3
"""SUPERSEDED by dual_5t_rviz_preview.py.

This node interpolated colliding I1/I2 IK onto /display_planned_path.
DUAL-5-T forbids disguising collision candidates as a validated path.
Do not run this node.
"""

from __future__ import annotations

import math

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

# DUAL-5-S tested states. Handover and I3 are valid; I1/I2 unique IK collide.
WAYPOINTS = {
    "Handover": [
        -1.662443,
        -2.302791,
        -1.754164,
        -0.655431,
        1.570793,
        -0.091643,
    ],
    "I1_side_pos_x_COLLISION": [
        -0.994338,
        1.155945,
        1.620296,
        0.365344,
        -0.576458,
        0.785405,
    ],
    "I2_side_neg_x_COLLISION": [
        -0.994339,
        -3.016769,
        -2.462299,
        -0.804122,
        0.576457,
        0.785406,
    ],
    "I3_original_top_circle": [
        -1.963019,
        -3.254885,
        -0.345739,
        -2.682560,
        1.178574,
        2.356201,
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
    for q in pts:
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
        super().__init__("dual_5s_rviz_preview")
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
            "DUAL-5-S RViz preview: Handover(hold)→I1 +X(collision IK)→"
            "I2 -X(collision IK)→I3 +Z(valid). Arm A stays Home. "
            "PLAN DISPLAY ONLY. No execute. No complete validated chain."
        )
        self._tick()

    def _build(self):
        names = [
            "Handover",
            "I1_side_pos_x_COLLISION",
            "I2_side_neg_x_COLLISION",
            "I3_original_top_circle",
        ]
        seq = [WAYPOINTS[n] for n in names]
        disp = DisplayTrajectory()
        disp.model_id = "fairino3_dual_robot"
        start = JointState()
        start.name = ARM_A + ARM_B + GRIPPERS
        start.position = HOME_A + seq[0] + [Q_A, Q_B]
        disp.trajectory_start.joint_state = start

        jt = JointTrajectory()
        jt.header.frame_id = "world"
        jt.joint_names = list(ARM_B)
        t = 0.0
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
