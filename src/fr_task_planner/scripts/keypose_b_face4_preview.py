#!/usr/bin/env python3
"""Hold Arm B at the saved B_FACE4 joints. Not executable."""

from __future__ import annotations

import os

import rclpy
import yaml
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from moveit_msgs.msg import DisplayRobotState
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

ARM_A = ["arm_a_j1", "arm_a_j2", "arm_a_j3", "arm_a_j4", "arm_a_j5", "arm_a_j6"]
ARM_B = ["arm_b_j1", "arm_b_j2", "arm_b_j3", "arm_b_j4", "arm_b_j5", "arm_b_j6"]
ALLJ = ARM_A + ARM_B + ["arm_a_gripper_joint", "arm_b_gripper_joint"]
RED = ColorRGBA(r=0.95, g=0.15, b=0.15, a=0.95)
MAG = ColorRGBA(r=0.85, g=0.2, b=0.75, a=0.85)
CYAN = ColorRGBA(r=0.1, g=0.85, b=0.95, a=0.95)
BLUE = ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.9)
GREEN = ColorRGBA(r=0.2, g=0.9, b=0.3, a=0.95)
YELLOW = ColorRGBA(r=1.0, g=0.9, b=0.2, a=0.95)
WHITE = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)


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


class AxisPreview(Node):
    def __init__(self):
        super().__init__("keypose_b_face4_preview")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("show_b", "B_FACE4")
        out = os.path.expanduser(self.get_parameter("output_dir").value)
        if not out:
            out = os.path.expanduser(
                "~/fr_task_ws/src/fr_task_planner/config/keypose_optimization_v1"
            )
        path = os.path.join(out, "b_face45_axis_align.yaml")
        with open(path, encoding="utf-8") as handle:
            self.data = yaml.safe_load(handle) or {}
        self.show_b = "B_FACE4"
        face = self.data.get("B_FACE4") or {}
        joints = list(face.get("joints") or [])
        if len(joints) < 6:
            raise RuntimeError("B_FACE4 joints missing in " + path)
        deg = [v * 180.0 / 3.141592653589793 for v in joints]
        q = qos()
        self.s_pub = self.create_publisher(DisplayRobotState, "/keypose/preview/robot_state", q)
        self.m_pub = self.create_publisher(MarkerArray, "/keypose/preview/markers", q)
        self.get_logger().info("B_FACE4 FIXED PREVIEW / NOT EXECUTABLE")
        self.get_logger().info(
            "locked joints deg J1={:.2f} J2={:.2f} J3={:.2f} J4={:.2f} J5={:.2f} J6={:.2f}".format(*deg)
        )
        self.get_logger().info(path)
        self.timer = self.create_timer(0.5, self._publish)

    def _js(self, a, b, qa, qb):
        js = JointState()
        js.header = _header(self.get_clock().now().to_msg())
        js.name = list(ALLJ)
        js.position = [float(v) for v in list(a[:6]) + list(b[:6]) + [float(qa), float(qb)]]
        return js

    def _cyl(self, stamp, mid, xyz, xyzw, color, ns):
        m = Marker()
        m.header = _header(stamp)
        m.ns = ns
        m.id = mid
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose = _pose(xyz, xyzw)
        m.scale = Vector3(x=0.018, y=0.018, z=0.038)
        m.color = color
        return m

    def _arrow(self, stamp, mid, xyz, vec, color, ns, length=0.10):
        m = Marker()
        m.header = _header(stamp)
        m.ns = ns
        m.id = mid
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale = Vector3(x=0.006, y=0.011, z=0.014)
        m.color = color
        m.points = [
            Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
            Point(
                x=float(xyz[0] + length * vec[0]),
                y=float(xyz[1] + length * vec[1]),
                z=float(xyz[2] + length * vec[2]),
            ),
        ]
        return m

    def _text(self, stamp, mid, xyz, text, color, z=0.028):
        m = Marker()
        m.header = _header(stamp)
        m.ns = "label"
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose = _pose(xyz)
        m.scale = Vector3(z=z)
        m.color = color
        m.text = text
        return m

    def _axis_from(self, block):
        ax = list(block.get("axis") or [0, 0.707, 0.707])
        obj = block.get("object") or {}
        xyz = list(obj.get("xyz") or [0, 0.305, 1.195])
        xyzw = list(obj.get("xyzw") or [0, 0, 0, 1])
        return xyz, xyzw, ax

    def _publish(self):
        a = list(self.data.get("home_a") or [0.0] * 6)
        show = self.data.get("B_FACE4") or {}
        b = list(show.get("joints") or [])
        if len(b) < 6:
            return
        st = DisplayRobotState()
        st.state.joint_state = self._js(a, b, 0.0, 0.083)
        st.state.is_diff = False
        self.s_pub.publish(st)

        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        p1 = list(self.data.get("p1") or [0.0, 0.3, 1.2])
        d1 = list(self.data.get("d1") or [0.0, -0.707, 0.707])
        up = list(self.data.get("up_world") or [0.0, 0.707, 0.707])
        j = list(show.get("joints") or [0.0] * 6)
        deg = [v * 180.0 / 3.141592653589793 for v in j]
        title = "FIXED B_FACE4  J1={:.1f}  J2={:.1f} | NOT EXECUTABLE".format(deg[0], deg[1])
        arr.markers.append(self._text(stamp, 1, [0.0, 0.58, 1.58], title, WHITE, 0.030))
        arr.markers.append(self._arrow(stamp, 2, p1, d1, GREEN, "d1", 0.12))
        arr.markers.append(self._arrow(stamp, 3, p1, up, YELLOW, "up", 0.12))
        arr.markers.append(self._text(stamp, 4, [p1[0], p1[1] - 0.05, p1[2] + 0.07], "P1 D1 up", GREEN))
        block = self.data.get("B_FACE4") or {}
        if (block.get("object") or {}).get("xyz"):
            xyz, xyzw, ax = self._axis_from(block)
            arr.markers.append(self._cyl(stamp, 30, xyz, xyzw, CYAN, "part"))
            arr.markers.append(self._arrow(stamp, 31, xyz, ax, CYAN, "axis", 0.11))
            arr.markers.append(
                self._text(stamp, 32, [xyz[0], xyz[1] + 0.03, xyz[2] + 0.05], "F4", CYAN)
            )
        self.m_pub.publish(arr)


def main():
    rclpy.init()
    node = AxisPreview()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
