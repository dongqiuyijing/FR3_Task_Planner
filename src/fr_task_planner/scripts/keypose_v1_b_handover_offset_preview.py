#!/usr/bin/env python3
"""RViz comparison of frozen and world-offset B_HANDOVER. Never commands hardware."""

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


def robot_state(stamp, a, b, qa, qb):
    msg = DisplayRobotState()
    msg.state.joint_state = JointState(
        header=Header(frame_id="world", stamp=stamp),
        name=list(ALL), position=list(a[:6]) + list(b[:6]) + [qa, qb])
    msg.state.is_diff = False
    return msg


class Preview(Node):
    def __init__(self):
        super().__init__("keypose_v1_b_handover_offset_preview")
        path = os.path.expanduser(
            "~/fr_task_ws/src/fr_task_planner/config/keypose_optimization_v1/"
            "b_handover_offset_preview.yaml")
        with open(path, encoding="utf-8") as stream:
            self.data = yaml.safe_load(stream)
        self.a = list(self.data["A_HANDOVER_joints"])
        self.old = list(self.data["old_B_HANDOVER_joints"])
        self.new = list(self.data["new_B_HANDOVER_joints"])
        self.qa = float(self.data["q_a_grasp"])
        self.qb = float(self.data["q_b_open"])
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST)
        self.new_pub = self.create_publisher(DisplayRobotState, "/keypose/preview/robot_state", qos)
        self.old_pub = self.create_publisher(DisplayRobotState, "/keypose/preview/robot_state_pre", qos)
        self.markers = self.create_publisher(MarkerArray, "/keypose/preview/markers", qos)
        self.create_timer(0.5, self.tick)
        self.get_logger().info("B_HANDOVER OFFSET RViz PREVIEW ONLY; no motion or gripper commands")

    def tick(self):
        stamp = self.get_clock().now().to_msg()
        self.new_pub.publish(robot_state(stamp, self.a, self.new, self.qa, self.qb))
        self.old_pub.publish(robot_state(stamp, self.a, self.old, self.qa, self.qb))
        old = self.data["old_B_HANDOVER_tcp"]["xyz"]
        new = self.data["new_B_HANDOVER_tcp"]["xyz"]
        arr = MarkerArray()
        for marker_id, xyz, color, label in (
                (1, old, ColorRGBA(r=1.0, g=0.55, b=0.1, a=1.0), "frozen B_HANDOVER"),
                (2, new, ColorRGBA(r=0.1, g=1.0, b=0.25, a=1.0), "offset B_HANDOVER")):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = stamp
            marker.ns = "b_handover_tcp"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
            marker.pose.orientation = Quaternion(w=1.0)
            marker.scale = Vector3(x=0.02, y=0.02, z=0.02)
            marker.color = color
            arr.markers.append(marker)
            text = Marker()
            text.header = marker.header
            text.ns = "b_handover_label"
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]) + 0.045)
            text.pose.orientation = Quaternion(w=1.0)
            text.scale.z = 0.025
            text.color = color
            text.text = label
            arr.markers.append(text)
        self.markers.publish(arr)


def main():
    rclpy.init()
    node = Preview()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
