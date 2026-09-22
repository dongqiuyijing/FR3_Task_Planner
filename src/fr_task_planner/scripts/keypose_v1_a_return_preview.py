#!/usr/bin/env python3
"""Preview only A Handover -> A Home. Not executable."""

import os

import rclpy
import yaml
from geometry_msgs.msg import Point, Quaternion, Vector3
from moveit_msgs.msg import DisplayRobotState
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

A = [f"arm_a_j{i}" for i in range(1, 7)]
B = [f"arm_b_j{i}" for i in range(1, 7)]
ALL = A + B + ["arm_a_gripper_joint", "arm_b_gripper_joint"]


class Preview(Node):
    def __init__(self):
        super().__init__("keypose_v1_a_return_preview")
        path = os.path.expanduser(
            "~/fr_task_ws/src/fr_task_planner/config/keypose_v1_six_face_trajectory.yaml"
        )
        data = yaml.safe_load(open(path, encoding="utf-8"))
        seg = next(s for s in data["stages"] if s.get("id") == "a_handover_to_home")
        self.b = list(seg.get("start_joints") and data["home_b"] or data["home_b"])
        # B stays at handover during this segment. Read it from the previous B segment end.
        bseg = next(s for s in data["stages"] if s.get("id") == "b_pre_to_handover")
        self.b = list(bseg["end_joints"])
        self.pts = [list(p["positions"]) for p in seg["points"]]
        self.i = 0
        q = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub = self.create_publisher(DisplayRobotState, "/keypose/preview/robot_state", q)
        self.mpub = self.create_publisher(MarkerArray, "/keypose/preview/markers", q)
        self.get_logger().info(
            f"A RETURN PREVIEW points={len(self.pts)} method={seg.get('method')} NOT EXECUTABLE"
        )
        self.create_timer(0.15, self.tick)

    def tick(self):
        q = self.pts[self.i]
        js = JointState()
        js.header.frame_id = "world"
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(ALL)
        js.position = q[:6] + self.b[:6] + [0.0, 0.083]
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
        m.text = f"A Handover->Home  {self.i + 1}/{len(self.pts)}  OMPL RRTConnect  NOT EXECUTABLE"
        arr = MarkerArray()
        arr.markers.append(m)
        self.mpub.publish(arr)
        self.i = (self.i + 1) % len(self.pts)


def main():
    rclpy.init()
    node = Preview()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
