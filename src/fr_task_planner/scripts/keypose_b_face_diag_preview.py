#!/usr/bin/env python3
"""Publish B_FACE4/5 diagnostic collision poses to RViz. Never commands the robot."""

from __future__ import annotations

import os

import rclpy
import yaml
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, DisplayRobotState
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

ARM_A = ["arm_a_j1", "arm_a_j2", "arm_a_j3", "arm_a_j4", "arm_a_j5", "arm_a_j6"]
ARM_B = ["arm_b_j1", "arm_b_j2", "arm_b_j3", "arm_b_j4", "arm_b_j5", "arm_b_j6"]
ALLJ = ARM_A + ARM_B + ["arm_a_gripper_joint", "arm_b_gripper_joint"]
WHITE = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
RED = ColorRGBA(r=0.95, g=0.2, b=0.2, a=0.95)
GREEN = ColorRGBA(r=0.2, g=0.85, b=0.3, a=1.0)
CYAN = ColorRGBA(r=0.2, g=0.8, b=0.95, a=0.95)
ORANGE = ColorRGBA(r=1.0, g=0.5, b=0.1, a=0.95)


def qos():
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


def _header(stamp, frame="world"):
    h = Header()
    h.stamp = stamp
    h.frame_id = frame
    return h


def _pose(xyz, xyzw=(0.0, 0.0, 0.0, 1.0)):
    p = Pose()
    p.position = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
    p.orientation = Quaternion(
        x=float(xyzw[0]), y=float(xyzw[1]), z=float(xyzw[2]), w=float(xyzw[3])
    )
    return p


class DiagPreview(Node):
    def __init__(self):
        super().__init__("keypose_b_face_diag_preview")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("keypose", "DIAG_DUAL7_I1")
        out = os.path.expanduser(self.get_parameter("output_dir").value)
        if not out:
            out = os.path.expanduser(
                "~/fr_task_ws/src/fr_task_planner/config/keypose_optimization_v1"
            )
        self.keypose = str(self.get_parameter("keypose").value)
        path = os.path.join(out, "b_face45_root_cause.yaml")
        with open(path, encoding="utf-8") as handle:
            self.data = yaml.safe_load(handle) or {}
        q = qos()
        self.s_pub = self.create_publisher(DisplayRobotState, "/keypose/preview/robot_state", q)
        self.m_pub = self.create_publisher(MarkerArray, "/keypose/preview/markers", q)
        self.get_logger().info("B_FACE4/5 DIAG PREVIEW ONLY / NOT EXECUTABLE")
        self.get_logger().info(f"keypose={self.keypose} yaml={path}")
        self.timer = self.create_timer(0.5, self._publish)

    def _js(self, a, b, qa, qb):
        js = JointState()
        js.header = _header(self.get_clock().now().to_msg())
        js.name = list(ALLJ)
        js.position = [float(v) for v in list(a[:6]) + list(b[:6]) + [float(qa), float(qb)]]
        return js

    def _attached(self):
        att = AttachedCollisionObject()
        att.link_name = "arm_b_gripper_tcp"
        att.touch_links = ["arm_b_finger_l", "arm_b_finger_r"]
        obj = CollisionObject()
        obj.id = "small_part"
        obj.header.frame_id = att.link_name
        obj.operation = CollisionObject.ADD
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.CYLINDER
        prim.dimensions = [0.035, 0.0075]
        obj.primitives.append(prim)
        obj.primitive_poses.append(_pose((0.0, 0.0, 0.008), (0.0, 0.0, -0.707107, 0.707107)))
        att.object = obj
        return [att]

    def _publish(self):
        preview = (self.data.get("rviz_preview") or {}).get(self.keypose) or {}
        if not preview:
            names = list((self.data.get("rviz_preview") or {}).keys())
            self.get_logger().error(f"missing {self.keypose}, have {names}")
            return
        a = list(preview.get("arm_a") or [0.0] * 6)
        b = list(preview.get("arm_b") or [0.0] * 6)
        illegal = bool(preview.get("illegal"))
        st = DisplayRobotState()
        st.state.joint_state = self._js(a, b, 0.0, 0.083)
        st.state.is_diff = False
        st.state.attached_collision_objects = self._attached()
        self.s_pub.publish(st)

        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        title = Marker()
        title.header = _header(stamp)
        title.ns = "keypose"
        title.id = 1
        title.type = Marker.TEXT_VIEW_FACING
        title.action = Marker.ADD
        title.pose = _pose((0.0, 0.55, 1.55))
        title.scale = Vector3(z=0.042)
        title.color = RED if illegal else GREEN
        title.text = f"DIAG {self.keypose} | A=Home | {'COLLIDES' if illegal else 'NO ILLEGAL COLLISION'}"
        arr.markers.append(title)
        sub = Marker()
        sub.header = _header(stamp)
        sub.ns = "keypose"
        sub.id = 2
        sub.type = Marker.TEXT_VIEW_FACING
        sub.action = Marker.ADD
        sub.pose = _pose((0.0, 0.55, 1.48))
        sub.scale = Vector3(z=0.028)
        sub.color = WHITE
        sub.text = str(preview.get("note") or "")
        arr.markers.append(sub)
        col = self.data.get("dual7_static_collision") or {}
        column = col.get("column") or {}
        if column.get("xyz") and column.get("dim"):
            cube = Marker()
            cube.header = _header(stamp)
            cube.ns = "workcell"
            cube.id = 3
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose = _pose(column["xyz"])
            d = column["dim"]
            cube.scale = Vector3(x=float(d[0]), y=float(d[1]), z=float(d[2]))
            cube.color = ColorRGBA(r=0.9, g=0.15, b=0.15, a=0.35)
            arr.markers.append(cube)
        self.m_pub.publish(arr)


def main():
    rclpy.init()
    node = DiagPreview()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
