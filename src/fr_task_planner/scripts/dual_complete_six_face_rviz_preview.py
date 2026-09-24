#!/usr/bin/env python3
"""DUAL-7 complete-task RViz preview.

Publishes /dual7/preview/* and, after taking over DUAL-5-T, /display_planned_path.
Never publishes /joint_states, /tf, /tf_static, or /apply_planning_scene.
Playback timing is VISUALIZATION PLAYBACK TIMING ONLY.
"""

from __future__ import annotations

import sys

import rclpy
import yaml
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    DisplayRobotState,
    DisplayTrajectory,
    RobotState,
    RobotTrajectory,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA, Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

ARM_A = ["arm_a_j1", "arm_a_j2", "arm_a_j3", "arm_a_j4", "arm_a_j5", "arm_a_j6"]
ARM_B = ["arm_b_j1", "arm_b_j2", "arm_b_j3", "arm_b_j4", "arm_b_j5", "arm_b_j6"]
ALLJ = ARM_A + ARM_B + ["arm_a_gripper_joint", "arm_b_gripper_joint"]
WHITE = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
GREEN = ColorRGBA(r=0.2, g=0.85, b=0.3, a=1.0)
YELLOW = ColorRGBA(r=1.0, g=0.85, b=0.15, a=1.0)
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


class Dual7Preview(Node):
    def __init__(self, path: str):
        super().__init__("dual_complete_six_face_rviz_preview")
        self.data = yaml.safe_load(open(path, encoding="utf-8")) or {}
        self.phases = list(self.data.get("phases") or [])
        self.complete = bool(self.data.get("complete"))
        self.frames = self._flatten()
        self.i = 0
        q = qos()
        self.m_pub = self.create_publisher(MarkerArray, "/dual7/preview/markers", q)
        self.s_pub = self.create_publisher(DisplayRobotState, "/dual7/preview/robot_state", q)
        self.t_pub = self.create_publisher(
            DisplayTrajectory, "/dual7/preview/display_planned_path", q
        )
        self.legacy = None
        if self.complete:
            self.legacy = self.create_publisher(DisplayTrajectory, "/display_planned_path", q)
        self.get_logger().info("DUAL-7 COMPLETE TASK PREVIEW")
        self.get_logger().info("RVIZ PREVIEW ONLY / NOT EXECUTABLE")
        self.get_logger().info("VISUALIZATION PLAYBACK TIMING ONLY")
        self.get_logger().info("MODEL PREVIEW ONLY")
        self.get_logger().info(f"frames={len(self.frames)} phases={len(self.phases)} complete={self.complete}")
        self._publish_path()
        self.timer = self.create_timer(0.04, self._tick)
        # Republish slowly so late RViz subscribers see the path, without
        # restarting a 30s Planned Path animation every frame.
        self.path_timer = self.create_timer(8.0, self._publish_path)

    def _flatten(self):
        frames = []
        cur_a = list(self.data.get("home_a") or [0.0] * 6)
        cur_b = list(self.data.get("home_b") or [0.0] * 6)
        owner = "none"
        for p in self.phases:
            wps = p.get("waypoints") or []
            moving = p.get("moving") or "arm_a"
            sa = list(p.get("start_a") or [0.0] * 6)
            sb = list(p.get("start_b") or [0.0] * 6)
            ea = list(p.get("end_a") or sa)
            eb = list(p.get("end_b") or sb)
            if not wps:
                wps = [{"joints": sa if moving == "arm_a" else sb}]
            hold = 18 if moving == "(none)" or "(transfer)" in str(p.get("face")) else 1
            for item in wps:
                q = item.get("joints") if isinstance(item, dict) else item
                frames.append((p, q, moving, sa, sb, ea, eb))
            last = wps[-1]
            q = last.get("joints") if isinstance(last, dict) else last
            for _ in range(hold):
                frames.append((p, q, moving, sa, sb, ea, eb))
        # Frozen KEYPOSE trajectories use `stages` and trajectory `points`,
        # rather than the older DUAL-7 `phases` / `waypoints` schema.
        for stage in self.data.get("stages") or []:
            p = dict(stage)
            sid = str(p.get("id", ""))
            moving = str(p.get("moving", ""))
            if sid == "gripper_close_a":
                owner = "A"
            elif sid in ("attachment_transfer", "gripper_close_b"):
                owner = "B"
            elif sid == "gripper_open_b_place":
                owner = "world"
            p["owner"] = owner
            p["name"] = sid
            p["face"] = sid
            p["object_xyz"] = p.get("object_center_world") or [-0.45, 0.4, 0.7675]
            p["q_a"] = 0.0
            p["q_b"] = 0.0 if owner == "world" else 0.083
            sa, sb = list(cur_a), list(cur_b)
            if moving == "dual":
                pa = [x.get("positions") for x in (p.get("arm_a", {}).get("points") or [])]
                pb = [x.get("positions") for x in (p.get("arm_b", {}).get("points") or [])]
                if not pa or not pb:
                    continue
                # The executor shares one time base but the two controllers
                # may have different point densities.  Sample both paths on
                # a common visual index so RViz visibly moves both arms.
                n = max(len(pa), len(pb))
                for i in range(n):
                    ia = round(i * (len(pa) - 1) / max(1, n - 1))
                    ib = round(i * (len(pb) - 1) / max(1, n - 1))
                    frames.append((p, {"a": pa[ia], "b": pb[ib]}, moving, sa, sb, pa[-1], pb[-1]))
                cur_a, cur_b = list(pa[-1]), list(pb[-1])
                continue
            raw = p.get("points") or []
            wps = [x.get("positions") if isinstance(x, dict) else x for x in raw]
            if not wps:
                wps = [list(cur_a if moving == "arm_a" else cur_b)]
            for q in wps:
                frames.append((p, q, moving, sa, sb, sa, sb))
            if moving == "arm_a":
                cur_a = list(wps[-1])
            elif moving == "arm_b":
                cur_b = list(wps[-1])
        return frames

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
            ["arm_a_finger_l", "arm_a_finger_r"] if owner == "A" else ["arm_b_finger_l", "arm_b_finger_r"]
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

    def _tick(self):
        if not self.frames:
            return
        p, q, moving, sa, sb, ea, eb = self.frames[self.i % len(self.frames)]
        if moving == "arm_a":
            a, b = list(q), list(sb)
        elif moving == "arm_b":
            a, b = list(sa), list(q)
        elif moving == "dual":
            a, b = list(q["a"]), list(q["b"])
        else:
            a, b = list(ea), list(eb)
        st = DisplayRobotState()
        st.state.joint_state = self._js(a, b, p.get("q_a", 0.0), p.get("q_b", 0.083))
        st.state.is_diff = False
        st.state.attached_collision_objects = self._attached(str(p.get("owner")))
        self.s_pub.publish(st)

        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        m = Marker()
        m.header = _header(stamp)
        m.ns = "dual7"
        m.id = 1
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose = _pose((0.0, 0.55, 1.50))
        m.scale = Vector3(z=0.05)
        m.color = WHITE
        m.text = "DUAL-7 MODEL PREVIEW ONLY | CURRENT PHASE: " + str(p.get("name"))
        arr.markers.append(m)
        m2 = Marker()
        m2.header = _header(stamp)
        m2.ns = "dual7"
        m2.id = 2
        m2.type = Marker.TEXT_VIEW_FACING
        m2.action = Marker.ADD
        m2.pose = _pose((0.0, 0.55, 1.43))
        m2.scale = Vector3(z=0.04)
        m2.color = GREEN
        m2.text = (
            "CURRENT OWNER OF small_part: "
            + str(p.get("owner"))
            + " | CURRENT PHYSICAL FACE: "
            + str(p.get("face"))
        )
        arr.markers.append(m2)
        m3 = Marker()
        m3.header = _header(stamp)
        m3.ns = "dual7"
        m3.id = 3
        m3.type = Marker.TEXT_VIEW_FACING
        m3.action = Marker.ADD
        m3.pose = _pose((0.0, 0.55, 1.37))
        m3.scale = Vector3(z=0.035)
        m3.color = YELLOW
        m3.text = "Arm A=left joints  Arm B=right joints  cylinder=small_part  NOT REAL ROBOT"
        arr.markers.append(m3)
        xyz = p.get("object_xyz") or [0.0, 0.4, 0.9]
        cyl = Marker()
        cyl.header = _header(stamp)
        cyl.ns = "dual7"
        cyl.id = 4
        cyl.type = Marker.CYLINDER
        cyl.action = Marker.ADD
        cyl.pose = _pose(xyz)
        cyl.scale = Vector3(x=0.015, y=0.015, z=0.035)
        cyl.color = ORANGE if str(p.get("owner")) == "A" else (CYAN if str(p.get("owner")) == "B" else WHITE)
        arr.markers.append(cyl)
        self.m_pub.publish(arr)
        self.i += 1
        if self.i % len(self.frames) == 0:
            self.get_logger().info("looping complete six-face preview")
            self._publish_path()

    def _publish_path(self):
        jt = JointTrajectory()
        jt.header = _header(self.get_clock().now().to_msg())
        jt.joint_names = list(ALLJ)
        t = 0.0
        start_a, start_b = [0.0] * 6, [0.0] * 6
        first = True
        for p, q, moving, sa, sb, ea, eb in self.frames:
            if moving == "arm_a":
                a, b = list(q), list(sb)
            elif moving == "arm_b":
                a, b = list(sa), list(q)
            else:
                a, b = list(ea), list(eb)
            if first:
                start_a, start_b = list(a), list(b)
                first = False
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in list(a[:6]) + list(b[:6]) + [float(p.get("q_a", 0.0)), float(p.get("q_b", 0.083))]]
            t += 0.04
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            jt.points.append(pt)
        if not jt.points:
            return
        disp = DisplayTrajectory()
        disp.model_id = "fairino3_dual_robot"
        rt = RobotTrajectory()
        rt.joint_trajectory = jt
        disp.trajectory.append(rt)
        start = RobotState()
        start.joint_state = self._js(start_a, start_b, 0.0, 0.0)
        disp.trajectory_start = start
        self.t_pub.publish(disp)
        if self.legacy is not None:
            self.legacy.publish(disp)


def main():
    path = "/tmp/dual7_preview.yaml"
    if len(sys.argv) > 1 and not sys.argv[1].startswith("__"):
        path = sys.argv[1]
    rclpy.init(args=sys.argv)
    node = Dual7Preview(path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
