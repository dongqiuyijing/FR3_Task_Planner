#!/usr/bin/env python3
"""Show A_FACE1/2, correct B_FACE4/5 mirror, and DUAL-7 part poses. Not executable."""

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


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def qrot(q, v):
    x, y, z, w = q
    qv = (v[0], v[1], v[2], 0.0)
    r = qmul(qmul(q, qv), (-x, -y, -z, w))
    return (r[0], r[1], r[2])


def pose_mul(a_xyz, a_xyzw, b_xyz, b_xyzw):
    rx, ry, rz = qrot(a_xyzw, b_xyz)
    return (
        [a_xyz[0] + rx, a_xyz[1] + ry, a_xyz[2] + rz],
        qmul(a_xyzw, b_xyzw),
    )


def obj_from_tcp(tcp, t_tcp_obj):
    txyz = list((tcp or {}).get("xyz") or [0, 0, 0])
    txyzw = list((tcp or {}).get("xyzw") or [0, 0, 0, 1])
    oxyz = list((t_tcp_obj or {}).get("xyz") or [0, 0, 0.008])
    oxyzw = list((t_tcp_obj or {}).get("xyzw") or [0, 0, -0.707107, 0.707107])
    return pose_mul(txyz, txyzw, oxyz, oxyzw)


class MirrorTargetPreview(Node):
    def __init__(self):
        super().__init__("keypose_b_face_mirror_preview")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("show_b", "DUAL7_I1")
        out = os.path.expanduser(self.get_parameter("output_dir").value)
        if not out:
            out = os.path.expanduser(
                "~/fr_task_ws/src/fr_task_planner/config/keypose_optimization_v1"
            )
        path = os.path.join(out, "b_face45_mirror_targets.yaml")
        with open(path, encoding="utf-8") as handle:
            self.data = yaml.safe_load(handle) or {}
        self.show_b = str(self.get_parameter("show_b").value)
        q = qos()
        self.s_pub = self.create_publisher(DisplayRobotState, "/keypose/preview/robot_state", q)
        self.m_pub = self.create_publisher(MarkerArray, "/keypose/preview/markers", q)
        self.get_logger().info("B_FACE4/5 MIRROR TARGET PREVIEW / NOT EXECUTABLE")
        self.get_logger().info(path)
        self.timer = self.create_timer(0.4, self._publish)

    def _js(self, a, b, qa, qb):
        js = JointState()
        js.header = _header(self.get_clock().now().to_msg())
        js.name = list(ALLJ)
        js.position = [float(v) for v in list(a[:6]) + list(b[:6]) + [float(qa), float(qb)]]
        return js

    def _attached_a(self, obj_xyz, obj_xyzw):
        att = AttachedCollisionObject()
        att.link_name = "arm_a_gripper_tcp"
        att.touch_links = ["arm_a_finger_l", "arm_a_finger_r"]
        obj = CollisionObject()
        obj.id = "small_part"
        obj.header.frame_id = "world"
        obj.operation = CollisionObject.ADD
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.CYLINDER
        prim.dimensions = [0.035, 0.0075]
        obj.primitives.append(prim)
        obj.primitive_poses.append(_pose(obj_xyz, obj_xyzw))
        att.object = obj
        return []

    def _cyl(self, stamp, mid, xyz, xyzw, color, ns, scale=(0.018, 0.018, 0.038)):
        m = Marker()
        m.header = _header(stamp)
        m.ns = ns
        m.id = mid
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose = _pose(xyz, xyzw)
        m.scale = Vector3(x=scale[0], y=scale[1], z=scale[2])
        m.color = color
        return m

    def _arrow(self, stamp, mid, xyz, vec, color, ns):
        m = Marker()
        m.header = _header(stamp)
        m.ns = ns
        m.id = mid
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale = Vector3(x=0.007, y=0.012, z=0.016)
        m.color = color
        start = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
        end = Point(
            x=float(xyz[0] + 0.09 * vec[0]),
            y=float(xyz[1] + 0.09 * vec[1]),
            z=float(xyz[2] + 0.09 * vec[2]),
        )
        m.points = [start, end]
        return m

    def _text(self, stamp, mid, xyz, text, color, ns="label"):
        m = Marker()
        m.header = _header(stamp)
        m.ns = ns
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose = _pose(xyz)
        m.scale = Vector3(z=0.028)
        m.color = color
        m.text = text
        return m

    def _item_obj(self, name):
        item = self.data.get(name) or {}
        obj = item.get("object")
        if obj and obj.get("xyz"):
            return list(obj["xyz"]), list(obj["xyzw"])
        tcp_obj = self.data.get("T_tcpB_object")
        if name.startswith("A_"):
            tcp_obj = self.data.get("T_tcpA_object")
            tcp = item.get("tcp")
            if tcp:
                return obj_from_tcp(tcp, tcp_obj)
        tcp = item.get("tcp")
        return obj_from_tcp(tcp, self.data.get("T_tcpB_object"))

    def _publish(self):
        a = list(self.data.get("A_FACE1_joints") or [0.0] * 6)
        b_src = self.data.get(self.show_b) or self.data.get("DUAL7_I1") or {}
        b = list(b_src.get("joints") or self.data.get("home_b") or [0.0] * 6)
        st = DisplayRobotState()
        st.state.joint_state = self._js(a, b, 0.083, 0.083)
        st.state.is_diff = False
        self.s_pub.publish(st)

        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()
        title = self._text(
            stamp,
            1,
            [0.0, 0.58, 1.58],
            "MIRROR TARGETS | orange=A  cyan=B_mirror  red=DUAL-7 | NOT EXECUTABLE",
            ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),
            "keypose",
        )
        title.scale = Vector3(z=0.036)
        arr.markers.append(title)
        spec = [
            ("A_FACE1", ColorRGBA(r=1.0, g=0.45, b=0.05, a=0.95), "A1", 10),
            ("A_FACE2", ColorRGBA(r=1.0, g=0.85, b=0.15, a=0.85), "A2", 20),
            ("B_FACE4_mirror", ColorRGBA(r=0.1, g=0.85, b=0.95, a=0.95), "B4m +90", 30),
            ("B_FACE5_mirror", ColorRGBA(r=0.15, g=0.35, b=1.0, a=0.95), "B5m +130", 40),
            ("DUAL7_I1", ColorRGBA(r=0.95, g=0.15, b=0.15, a=0.9), "D7 I1 -150", 50),
            ("DUAL7_I2", ColorRGBA(r=0.85, g=0.2, b=0.75, a=0.75), "D7 I2 -150", 60),
        ]
        for name, color, label, base in spec:
            item = self.data.get(name) or {}
            xyz, xyzw = self._item_obj(name)
            arr.markers.append(self._cyl(stamp, base, xyz, xyzw, color, "part"))
            fc = list(item.get("face_center") or xyz)
            fn = list(item.get("face_normal") or [0, -0.707, 0.707])
            fu = list(item.get("face_up") or [1, 0, 0])
            arr.markers.append(self._arrow(stamp, base + 1, fc, fn, color, "normal"))
            up_c = ColorRGBA(r=color.r, g=color.g, b=color.b, a=0.7)
            arr.markers.append(self._arrow(stamp, base + 2, fc, fu, up_c, "up"))
            arr.markers.append(
                self._text(stamp, base + 3, [xyz[0], xyz[1] + 0.04, xyz[2] + 0.05], label, color)
            )
        self.m_pub.publish(arr)


def main():
    rclpy.init()
    node = MirrorTargetPreview()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
