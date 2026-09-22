#!/usr/bin/env python3
"""Play KEYPOSE_V1 connected segments. Not executable."""

from __future__ import annotations

import os

import rclpy
import yaml
from moveit_msgs.msg import DisplayRobotState
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3

A = ["arm_a_j1", "arm_a_j2", "arm_a_j3", "arm_a_j4", "arm_a_j5", "arm_a_j6"]
B = ["arm_b_j1", "arm_b_j2", "arm_b_j3", "arm_b_j4", "arm_b_j5", "arm_b_j6"]
ALL = A + B + ["arm_a_gripper_joint", "arm_b_gripper_joint"]


class Tour(Node):
    def __init__(self):
        super().__init__("keypose_v1_tour_preview")
        path = os.path.expanduser(
            "~/fr_task_ws/src/fr_task_planner/config/keypose_v1_six_face_trajectory.yaml"
        )
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        self.frames = []
        a = list(data["home_a"])
        b = list(data["home_b"])
        qa, qb = 0.0, 0.0
        for st in data.get("stages") or []:
            kind = st.get("kind")
            if kind == "gripper_close" and st.get("moving") == "arm_a":
                qa = 0.083
            elif kind == "gripper_close" and st.get("moving") == "arm_b":
                qb = 0.083
            elif kind == "gripper_open" and st.get("id") == "gripper_open_a":
                qa = 0.0
            if kind != "joint_connection":
                continue
            pts = [list(p["positions"]) for p in (st.get("points") or [])]
            if not pts:
                continue
            step = max(1, len(pts) // 30)
            idxs = list(range(0, len(pts), step))
            if idxs[-1] != len(pts) - 1:
                idxs.append(len(pts) - 1)
            for n, i in enumerate(idxs):
                q = pts[i]
                if st.get("moving") == "arm_a":
                    a = list(q)
                else:
                    b = list(q)
                self.frames.append((st["id"], list(a), list(b), qa, qb, n == len(idxs) - 1))
        self.i = 0
        q = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub = self.create_publisher(DisplayRobotState, "/keypose/preview/robot_state", q)
        self.mpub = self.create_publisher(MarkerArray, "/keypose/preview/markers", q)
        self.get_logger().info(f"KEYPOSE V1 FULL PATH frames={len(self.frames)} NOT EXECUTABLE")
        self.create_timer(0.08, self.tick)

    def tick(self):
        if not self.frames:
            return
        sid, a, b, qa, qb, end = self.frames[self.i]
        js = JointState()
        js.header.frame_id = "world"
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(ALL)
        js.position = a[:6] + b[:6] + [qa, qb]
        st = DisplayRobotState()
        st.state.joint_state = js
        st.state.is_diff = False
        self.pub.publish(st)
        m = Marker()
        m.header = js.header
        m.ns = "keypose"
        m.id = 1
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position = Point(x=0.0, y=0.55, z=1.55)
        m.pose.orientation = Quaternion(w=1.0)
        m.scale = Vector3(z=0.04)
        m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        m.text = f"{sid}  {self.i + 1}/{len(self.frames)}  FULL PATH  NOT EXECUTABLE"
        arr = MarkerArray()
        arr.markers.append(m)
        self.mpub.publish(arr)
        if end:
            self.get_logger().info(sid)
        self.i = (self.i + 1) % len(self.frames)


def main():
    rclpy.init()
    node = Tour()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
