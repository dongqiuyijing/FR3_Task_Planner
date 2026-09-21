#!/usr/bin/env python3
"""DUAL-5-T RViz preview only.

Publishes independent preview topics under /dual5s/preview/*.
Never publishes /joint_states, /tf, /tf_static, or /apply_planning_scene.
DisplayTrajectory is published only for a collision-validated chain.
"""

from __future__ import annotations

import os
import sys

import rclpy
import yaml
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from moveit_msgs.msg import DisplayRobotState, DisplayTrajectory, ObjectColor, RobotState, RobotTrajectory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

ARM_A = ["arm_a_j1", "arm_a_j2", "arm_a_j3", "arm_a_j4", "arm_a_j5", "arm_a_j6"]
ARM_B = ["arm_b_j1", "arm_b_j2", "arm_b_j3", "arm_b_j4", "arm_b_j5", "arm_b_j6"]
GRIPPERS = ["arm_a_gripper_joint", "arm_b_gripper_joint"]
ALL_JOINTS = ARM_A + ARM_B + GRIPPERS

GREEN = ColorRGBA(r=0.1, g=0.85, b=0.2, a=0.95)
RED = ColorRGBA(r=0.95, g=0.15, b=0.1, a=0.95)
BLUE = ColorRGBA(r=0.15, g=0.45, b=1.0, a=0.95)
YELLOW = ColorRGBA(r=1.0, g=0.85, b=0.1, a=0.95)
GRAY = ColorRGBA(r=0.45, g=0.45, b=0.5, a=0.35)
WHITE = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)


def _qos():
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


def _color_marker(mid, ns, mtype, pose, scale, color, text=""):
    m = Marker()
    m.header.frame_id = "world"
    m.ns = ns
    m.id = mid
    m.type = mtype
    m.action = Marker.ADD
    m.pose = pose
    m.scale = scale
    m.color = color
    m.lifetime.sec = 0
    m.text = text
    return m


def load_preview(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


class Dual5tPreview(Node):
    def __init__(self, yaml_path: str):
        super().__init__("dual_5t_rviz_preview")
        self.yaml_path = yaml_path
        self.data = load_preview(yaml_path)
        self.validated = bool(self.data.get("validated_path"))
        self.candidates = list(self.data.get("candidates") or [])
        self.idx = 0

        qos = _qos()
        self.marker_pub = self.create_publisher(MarkerArray, "/dual5s/preview/markers", qos)
        self.state_pub = self.create_publisher(
            DisplayRobotState, "/dual5s/preview/robot_state", qos
        )
        self.path_pub = self.create_publisher(
            DisplayTrajectory, "/dual5s/preview/display_planned_path", qos
        )
        self.validated_moveit_pub = None
        if self.validated:
            self.validated_moveit_pub = self.create_publisher(
                DisplayTrajectory, "/display_planned_path", qos
            )

        self.get_logger().info("RVIZ PREVIEW ONLY")
        self.get_logger().info("NOT EXECUTABLE")
        self.get_logger().info("NOT REAL ROBOT STATE")
        self.get_logger().info(f"yaml={yaml_path}")
        self.get_logger().info(
            "Validated path published: " + ("YES" if self.validated else "NO")
        )
        if not self.validated:
            self.get_logger().warn("STATIC IK PREVIEW ONLY / NO VALIDATED PATH AVAILABLE")
            self.get_logger().warn("Not publishing a success trajectory on /display_planned_path")

        self.timer = self.create_timer(1.0, self._tick)
        self._tick()

    def _joint_state(self, joints_b):
        js = JointState()
        js.header = _header(self.get_clock().now().to_msg())
        js.name = list(ALL_JOINTS)
        home = list(self.data.get("home_a") or [0.0] * 6)
        qb = list(joints_b or [0.0] * 6)
        if len(home) < 6:
            home += [0.0] * (6 - len(home))
        if len(qb) < 6:
            qb += [0.0] * (6 - len(qb))
        qa = float(self.data.get("q_a", 0.0))
        qb_g = float(self.data.get("q_b", 0.083))
        js.position = list(home[:6]) + list(qb[:6]) + [qa, qb_g]
        return js

    def _markers(self):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        mid = 0

        banner = str(
            self.data.get("banner")
            or "STATIC IK PREVIEW ONLY / NO VALIDATED PATH AVAILABLE"
        )
        arr.markers.append(
            _color_marker(
                mid,
                "legend",
                Marker.TEXT_VIEW_FACING,
                _pose((0.0, 0.55, 1.45)),
                Vector3(x=0.0, y=0.0, z=0.06),
                WHITE,
                "RVIZ PREVIEW ONLY  |  " + banner,
            )
        )
        mid += 1
        arr.markers.append(
            _color_marker(
                mid,
                "legend",
                Marker.TEXT_VIEW_FACING,
                _pose((0.0, 0.55, 1.38)),
                Vector3(x=0.0, y=0.0, z=0.04),
                WHITE,
                "GREEN=collision-free IK  RED=collision IK  BLUE=target pose  NOT real robot",
            )
        )
        mid += 1

        col = self.data.get("column") or {}
        arr.markers.append(
            _color_marker(
                mid,
                "column",
                Marker.CUBE,
                _pose(col.get("xyz") or [0.0, 0.0, 0.7]),
                Vector3(
                    x=float((col.get("size") or [0.2, 0.2, 1.4])[0]),
                    y=float((col.get("size") or [0.2, 0.2, 1.4])[1]),
                    z=float((col.get("size") or [0.2, 0.2, 1.4])[2]),
                ),
                GRAY,
            )
        )
        mid += 1
        arr.markers.append(
            _color_marker(
                mid,
                "column",
                Marker.TEXT_VIEW_FACING,
                _pose((0.0, 0.0, 1.45)),
                Vector3(x=0.0, y=0.0, z=0.04),
                YELLOW,
                "mounting_column",
            )
        )
        mid += 1

        for target in self.data.get("targets") or []:
            xyz = target.get("object_xyz") or [0.0, 0.0, 0.0]
            xyzw = target.get("object_xyzw") or [0.0, 0.0, 0.0, 1.0]
            arr.markers.append(
                _color_marker(
                    mid,
                    "targets",
                    Marker.SPHERE,
                    _pose(xyz, xyzw),
                    Vector3(x=0.03, y=0.03, z=0.03),
                    BLUE,
                )
            )
            mid += 1
            arr.markers.append(
                _color_marker(
                    mid,
                    "targets",
                    Marker.TEXT_VIEW_FACING,
                    _pose((xyz[0], xyz[1], xyz[2] + 0.05), xyzw),
                    Vector3(x=0.0, y=0.0, z=0.035),
                    BLUE,
                    f"{target.get('name')} {target.get('physical_id')} roll={target.get('roll_deg')}",
                )
            )
            mid += 1
            tcp = target.get("tcp_xyz") or xyz
            arr.markers.append(
                _color_marker(
                    mid,
                    "tcp_axes",
                    Marker.SPHERE,
                    _pose(tcp, target.get("tcp_xyzw") or xyzw),
                    Vector3(x=0.018, y=0.018, z=0.018),
                    BLUE,
                )
            )
            mid += 1

        for cand in self.candidates:
            color = GREEN if cand.get("valid") else RED
            label = (
                f"{cand.get('face')} roll={cand.get('roll_deg')} "
                f"{'VALID' if cand.get('valid') else cand.get('status')}"
            )
            if cand.get("collision_pair"):
                label += f" | {cand.get('collision_pair')}"
            # Joint-only candidates: label near P1, offset by roll so they do not stack.
            roll = float(cand.get("roll_deg") or 0.0)
            arr.markers.append(
                _color_marker(
                    mid,
                    "candidates",
                    Marker.TEXT_VIEW_FACING,
                    _pose((0.12, 0.30 + 0.002 * roll, 1.25)),
                    Vector3(x=0.0, y=0.0, z=0.028),
                    color,
                    label,
                )
            )
            mid += 1

        for m in arr.markers:
            m.header.stamp = stamp
        return arr

    def _display_state(self):
        msg = DisplayRobotState()
        msg.state = RobotState()
        joints_b = None
        highlight = []
        if self.candidates:
            cand = self.candidates[self.idx % len(self.candidates)]
            joints_b = cand.get("joints")
            if not cand.get("valid"):
                pair = str(cand.get("collision_pair") or "")
                for token in pair.replace("<->", " ").replace(",", " ").split():
                    if token.startswith("arm_"):
                        highlight.append(token)
            self.get_logger().info(
                f"preview candidate {self.idx % len(self.candidates) + 1}/"
                f"{len(self.candidates)} face={cand.get('face')} "
                f"roll={cand.get('roll_deg')} valid={cand.get('valid')} "
                f"status={cand.get('status')}"
            )
        elif self.data.get("handover_b"):
            joints_b = self.data.get("handover_b")
        msg.state.joint_state = self._joint_state(joints_b)
        msg.state.is_diff = False
        for link in highlight:
            color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.9)
            oc = ObjectColor()
            oc.id = link
            oc.color = color
            msg.highlight_links.append(oc)
        return msg

    def _display_traj(self):
        wps = self.data.get("validated_waypoints") or []
        if not wps:
            return None
        jt = JointTrajectory()
        jt.header = _header(self.get_clock().now().to_msg())
        jt.joint_names = list(ARM_B)
        t = 0.0
        for item in wps:
            joints = item.get("joints") if isinstance(item, dict) else item
            if not joints:
                continue
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in joints]
            t += 0.05
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            jt.points.append(pt)
        if not jt.points:
            return None
        traj = RobotTrajectory()
        traj.joint_trajectory = jt
        disp = DisplayTrajectory()
        disp.model_id = "fairino3_dual_robot"
        disp.trajectory.append(traj)
        start = RobotState()
        start.joint_state = self._joint_state(jt.points[0].positions)
        disp.trajectory_start = start
        return disp

    def _tick(self):
        self.marker_pub.publish(self._markers())
        self.state_pub.publish(self._display_state())
        if self.validated:
            traj = self._display_traj()
            if traj is not None:
                self.path_pub.publish(traj)
                if self.validated_moveit_pub is not None:
                    self.validated_moveit_pub.publish(traj)
        if self.candidates:
            self.idx += 1


def main():
    yaml_path = "/tmp/dual5t_preview.yaml"
    if len(sys.argv) > 1 and not sys.argv[1].startswith("__"):
        yaml_path = sys.argv[1]
    elif os.environ.get("DUAL5T_PREVIEW_YAML"):
        yaml_path = os.environ["DUAL5T_PREVIEW_YAML"]
    rclpy.init(args=sys.argv)
    node = Dual5tPreview(yaml_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
