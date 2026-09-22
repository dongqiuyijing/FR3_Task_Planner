#!/usr/bin/env python3
"""Publish one keypose candidate to RViz. Never publishes /joint_states."""

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
GREEN = ColorRGBA(r=0.2, g=0.85, b=0.3, a=1.0)
YELLOW = ColorRGBA(r=1.0, g=0.85, b=0.15, a=1.0)
CYAN = ColorRGBA(r=0.2, g=0.8, b=0.95, a=0.95)
ORANGE = ColorRGBA(r=1.0, g=0.5, b=0.1, a=0.95)
RED = ColorRGBA(r=0.95, g=0.2, b=0.2, a=0.95)
BLUE = ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.95)
TOUR = [
    "A_HOME",
    "A_PREGRASP",
    "A_GRASP",
    "A_LIFT",
    "A_FACE1",
    "A_FACE2",
    "A_FACE3",
    "A_PRE_HANDOVER",
    "A_HANDOVER",
    "B_PRE_HANDOVER",
    "B_HANDOVER",
    "B_FACE4",
    "B_FACE5",
    "B_FACE6",
    "B_HOME",
]


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


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _cand(block, index):
    cands = list((block or {}).get("candidates") or [])
    if not cands:
        return None
    for item in cands:
        if int(item.get("index", 0)) == int(index):
            return item
    if 1 <= int(index) <= len(cands):
        return cands[int(index) - 1]
    return cands[0]


class KeyposePreview(Node):
    def __init__(self):
        super().__init__("keypose_candidate_rviz_preview")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("keypose", "A_FACE2")
        self.declare_parameter("candidate", 1)
        self.declare_parameter("cycle", False)
        self.declare_parameter("dwell_sec", 1.0)
        out = os.path.expanduser(self.get_parameter("output_dir").value)
        if not out:
            out = os.path.expanduser(
                "~/fr_task_ws/src/fr_task_planner/config/keypose_optimization_v1"
            )
        self.cycle = bool(self.get_parameter("cycle").value)
        self.dwell_sec = max(0.2, float(self.get_parameter("dwell_sec").value))
        self.step = 0
        self.step_t = self.get_clock().now()
        self.keypose = TOUR[0] if self.cycle else str(self.get_parameter("keypose").value)
        self.index = 1 if self.cycle else int(self.get_parameter("candidate").value)
        self.index_data = _load(os.path.join(out, "preview_index.yaml"))
        self.arm_a = _load(os.path.join(out, "arm_a_candidates.yaml"))
        self.arm_b = _load(os.path.join(out, "arm_b_candidates.yaml"))
        self.handover = _load(os.path.join(out, "handover_candidates.yaml"))
        self.targets = _load(os.path.join(out, "task_space_targets.yaml"))
        self.home_a = list(self.index_data.get("home_a") or [0.0] * 6)
        self.home_b = list(self.index_data.get("home_b") or [0.0] * 6)
        q = qos()
        self.s_pub = self.create_publisher(DisplayRobotState, "/keypose/preview/robot_state", q)
        self.m_pub = self.create_publisher(MarkerArray, "/keypose/preview/markers", q)
        self.get_logger().info("KEYPOSE V1 RVIZ PREVIEW ONLY / NOT EXECUTABLE")
        if self.cycle:
            self.get_logger().info(
                "tour dwell {:.1f}s: {}".format(self.dwell_sec, " -> ".join(TOUR))
            )
        else:
            self.get_logger().info(f"keypose={self.keypose} candidate={self.index}")
        self.timer = self.create_timer(0.2, self._publish)

    def _pick(self):
        name = self.keypose
        a = list(self.home_a)
        b = list(self.home_b)
        owner = "none"
        obj_xyz = [0.0, 0.4, 0.9]
        obj_xyzw = [0.0, 0.0, 0.0, 1.0]
        qa = float(self.index_data.get("q_a_open", 0.0))
        qb = float(self.index_data.get("q_b_open", 0.0))
        face_c = face_n = face_u = None
        cost = None
        delta = None
        note = ""

        if name in ("HANDOVER", "A_HANDOVER", "B_HANDOVER", "A_PRE_HANDOVER", "B_PRE_HANDOVER"):
            pairs = list(self.handover.get("pairs") or [])
            pair = None
            for item in pairs:
                if int(item.get("index", 0)) == self.index:
                    pair = item
                    break
            if pair is None and pairs:
                pair = pairs[min(self.index, len(pairs)) - 1]
            if pair:
                if name == "A_PRE_HANDOVER":
                    a = list(pair.get("A_PRE_HANDOVER_joints") or a)
                else:
                    a = list(pair.get("A_HANDOVER_joints") or a)
                if name == "B_PRE_HANDOVER":
                    b = list(pair.get("B_PRE_HANDOVER_joints") or b)
                else:
                    b = list(pair.get("B_HANDOVER_joints") or b)
                obj = pair.get("T_world_object") or {}
                obj_xyz = list(obj.get("xyz") or obj_xyz)
                obj_xyzw = list(obj.get("xyzw") or obj_xyzw)
                owner = "A" if name.startswith("A_") or name == "HANDOVER" else "B"
                if name == "B_PRE_HANDOVER":
                    owner = "A"
                qa = float(self.index_data.get("q_a_grasp", 0.083))
                qb = float(self.index_data.get("q_b_hold" if owner == "B" else "q_b_open", 0.0))
                cost = pair.get("cost")
                delta = pair.get("A_delta_from_FACE3")
                note = f"pair {pair.get('index')} shared T_world_object"
        elif name.startswith("A_"):
            block = (self.arm_a.get("keyposes") or {}).get(name) or {}
            item = _cand(block, self.index)
            if item:
                a = list(item.get("joint_values") or a)
                pose = item.get("task_space_pose") or {}
                cost = item.get("cost")
                delta = item.get("delta_joints")
                if name in ("A_PREGRASP",):
                    owner = "none"
                    obj = (self.targets.get("A_GRASP") or {}).get("tcp") or pose
                    obj_xyz = list((obj.get("xyz") if isinstance(obj, dict) else None) or obj_xyz)
                elif name == "A_HOME":
                    owner = "none"
                else:
                    owner = "A"
                    qa = float(self.index_data.get("q_a_grasp", 0.083))
                face_c = item.get("face_center")
                face_n = item.get("face_normal")
                face_u = item.get("face_up")
                tgt = self.targets.get(name) or {}
                objp = tgt.get("object") or {}
                if objp:
                    obj_xyz = list(objp.get("xyz") or obj_xyz)
                    obj_xyzw = list(objp.get("xyzw") or obj_xyzw)
        elif name.startswith("B_"):
            block = (self.arm_b.get("keyposes") or {}).get(name) or {}
            item = _cand(block, self.index)
            if item:
                b = list(item.get("joint_values") or b)
                cost = item.get("cost")
                delta = item.get("delta_joints")
                if name == "B_HOME":
                    owner = "none"
                elif name == "B_PRE_HANDOVER":
                    owner = "A"
                    qa = float(self.index_data.get("q_a_grasp", 0.083))
                else:
                    owner = "B"
                    qb = float(self.index_data.get("q_b_hold", 0.083))
                face_c = item.get("face_center")
                face_n = item.get("face_normal")
                face_u = item.get("face_up")
                tgt = self.targets.get(name) or {}
                objp = tgt.get("object") or {}
                if objp:
                    obj_xyz = list(objp.get("xyz") or obj_xyz)
                    obj_xyzw = list(objp.get("xyzw") or obj_xyzw)
                if name in ("B_HANDOVER", "B_PRE_HANDOVER"):
                    pairs = list(self.handover.get("pairs") or [])
                    if pairs:
                        pair = pairs[min(self.index, len(pairs)) - 1]
                        a = list(pair.get("A_HANDOVER_joints") or a)
                        obj = pair.get("T_world_object") or {}
                        obj_xyz = list(obj.get("xyz") or obj_xyz)
                        obj_xyzw = list(obj.get("xyzw") or obj_xyzw)
        return a, b, qa, qb, owner, obj_xyz, obj_xyzw, face_c, face_n, face_u, cost, delta, note

    def _js(self, a, b, qa, qb):
        js = JointState()
        js.header = _header(self.get_clock().now().to_msg())
        js.name = list(ALLJ)
        js.position = [float(v) for v in list(a[:6]) + list(b[:6]) + [float(qa), float(qb)]]
        return js

    def _attached(self, owner):
        if owner not in ("A", "B"):
            return []
        att = AttachedCollisionObject()
        att.link_name = "arm_a_gripper_tcp" if owner == "A" else "arm_b_gripper_tcp"
        att.touch_links = (
            ["arm_a_finger_l", "arm_a_finger_r"]
            if owner == "A"
            else ["arm_b_finger_l", "arm_b_finger_r"]
        )
        obj = CollisionObject()
        obj.id = "small_part"
        obj.header.frame_id = att.link_name
        obj.operation = CollisionObject.ADD
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.CYLINDER
        prim.dimensions = [0.035, 0.0075]
        obj.primitives.append(prim)
        if owner == "A":
            obj.primitive_poses.append(_pose((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 0.0)))
        else:
            obj.primitive_poses.append(_pose((0.0, 0.0, 0.008), (0.0, 0.0, -0.707107, 0.707107)))
        att.object = obj
        return [att]

    def _arrow(self, stamp, mid, xyz, vec, color, ns="face"):
        m = Marker()
        m.header = _header(stamp)
        m.ns = ns
        m.id = mid
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale = Vector3(x=0.008, y=0.014, z=0.018)
        m.color = color
        start = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
        end = Point(
            x=float(xyz[0] + 0.08 * vec[0]),
            y=float(xyz[1] + 0.08 * vec[1]),
            z=float(xyz[2] + 0.08 * vec[2]),
        )
        m.points = [start, end]
        return m

    def _advance(self):
        if not self.cycle:
            return
        now = self.get_clock().now()
        if (now - self.step_t).nanoseconds < int(self.dwell_sec * 1e9):
            return
        self.step = (self.step + 1) % len(TOUR)
        self.keypose = TOUR[self.step]
        self.index = 1
        self.step_t = now
        self.get_logger().info(f"{self.step + 1}/{len(TOUR)} {self.keypose}")

    def _publish(self):
        self._advance()
        a, b, qa, qb, owner, obj_xyz, obj_xyzw, face_c, face_n, face_u, cost, delta, note = (
            self._pick()
        )
        st = DisplayRobotState()
        st.state.joint_state = self._js(a, b, qa, qb)
        st.state.is_diff = False
        st.state.attached_collision_objects = self._attached(owner)
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
        title.scale = Vector3(z=0.045)
        title.color = WHITE
        if self.cycle:
            title.text = f"{self.step + 1}/{len(TOUR)}  {self.keypose}  | 1s | NOT EXECUTABLE"
        else:
            title.text = f"KEYPOSE PREVIEW ONLY | {self.keypose} candidate {self.index}"
        arr.markers.append(title)
        sub = Marker()
        sub.header = _header(stamp)
        sub.ns = "keypose"
        sub.id = 2
        sub.type = Marker.TEXT_VIEW_FACING
        sub.action = Marker.ADD
        sub.pose = _pose((0.0, 0.55, 1.48))
        sub.scale = Vector3(z=0.032)
        sub.color = GREEN
        cost_s = f"cost={cost:.4f}" if isinstance(cost, (int, float)) else ""
        d_s = ""
        if isinstance(delta, list) and len(delta) >= 6:
            d_s = " dJ=[" + ", ".join(f"{float(v):+.3f}" for v in delta[:6]) + "]"
        sub.text = f"owner={owner} {cost_s}{d_s} {note}"
        arr.markers.append(sub)
        cyl = Marker()
        cyl.header = _header(stamp)
        cyl.ns = "keypose"
        cyl.id = 3
        cyl.type = Marker.CYLINDER
        cyl.action = Marker.ADD
        cyl.pose = _pose(obj_xyz, obj_xyzw)
        cyl.scale = Vector3(x=0.015, y=0.015, z=0.035)
        cyl.color = ORANGE if owner == "A" else (CYAN if owner == "B" else WHITE)
        arr.markers.append(cyl)
        if face_c and face_n:
            arr.markers.append(self._arrow(stamp, 10, face_c, face_n, RED, "face_n"))
        if face_c and face_u:
            arr.markers.append(self._arrow(stamp, 11, face_c, face_u, BLUE, "face_u"))
        self.m_pub.publish(arr)


def main():
    rclpy.init()
    node = KeyposePreview()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
