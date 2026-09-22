#!/usr/bin/env python3
"""Show new B_HOME beside unchanged B_PRE_HANDOVER. Not executable."""

import os

import rclpy
import yaml
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from moveit_msgs.msg import DisplayRobotState
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

A = [f"arm_a_j{i}" for i in range(1, 7)]
B = [f"arm_b_j{i}" for i in range(1, 7)]
ALL = A + B + ["arm_a_gripper_joint", "arm_b_gripper_joint"]


def js(stamp, a, b):
    msg = JointState()
    msg.header.frame_id = "world"
    msg.header.stamp = stamp
    msg.name = list(ALL)
    msg.position = list(a[:6]) + list(b[:6]) + [0.0, 0.0]
    return msg


class Preview(Node):
    def __init__(self):
        super().__init__("keypose_v1_b_home_preview")
        path = os.path.expanduser("~/fr_task_ws/src/fr_task_planner/config/keypose_v1_b_home_preview.yaml")
        data = yaml.safe_load(open(path, encoding="utf-8"))
        self.a = list(data["a_home"])
        self.home = list(data["b_home"])
        self.pre = list(data["b_pre"])
        self.obj = list(data["object"])
        q = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(DisplayRobotState, "/keypose/preview/robot_state", q)
        self.pre_pub = self.create_publisher(DisplayRobotState, "/keypose/preview/robot_state_pre", q)
        self.mpub = self.create_publisher(MarkerArray, "/keypose/preview/markers", q)
        self.create_timer(0.5, self.tick)

    def tick(self):
        stamp = self.get_clock().now().to_msg()
        st = DisplayRobotState()
        st.state.joint_state = js(stamp, self.a, self.home)
        st.state.is_diff = False
        self.pub.publish(st)
        pre = DisplayRobotState()
        pre.state.joint_state = js(stamp, self.a, self.pre)
        pre.state.is_diff = False
        self.pre_pub.publish(pre)
        arr = MarkerArray()
        table = Marker()
        table.header.frame_id = "world"
        table.header.stamp = stamp
        table.ns = "table"
        table.id = 1
        table.type = Marker.CUBE
        table.action = Marker.ADD
        table.pose.position = Point(x=0.0, y=0.7, z=0.735)
        table.pose.orientation.w = 1.0
        table.scale = Vector3(x=1.0, y=1.0, z=0.03)
        table.color = ColorRGBA(r=0.45, g=0.45, b=0.45, a=0.35)
        arr.markers.append(table)
        spot = Marker()
        spot.header = table.header
        spot.ns = "placement"
        spot.id = 2
        spot.type = Marker.SPHERE
        spot.action = Marker.ADD
        spot.pose.position = Point(x=-0.45, y=0.40, z=0.750)
        spot.pose.orientation.w = 1.0
        spot.scale = Vector3(x=0.03, y=0.03, z=0.03)
        spot.color = ColorRGBA(r=1.0, g=0.85, b=0.1, a=1.0)
        arr.markers.append(spot)
        cyl = Marker()
        cyl.header = table.header
        cyl.ns = "cylinder"
        cyl.id = 3
        cyl.type = Marker.CYLINDER
        cyl.action = Marker.ADD
        cyl.pose.position = Point(x=self.obj[0], y=self.obj[1], z=self.obj[2])
        cyl.pose.orientation.w = 1.0
        cyl.scale = Vector3(x=0.015, y=0.015, z=0.035)
        cyl.color = ColorRGBA(r=0.1, g=0.85, b=0.95, a=0.9)
        arr.markers.append(cyl)
        label = Marker()
        label.header = table.header
        label.ns = "label"
        label.id = 4
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position = Point(x=0.0, y=0.55, z=1.45)
        label.pose.orientation.w = 1.0
        label.scale.z = 0.04
        label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        label.text = "cyan robot = new B_HOME   ghost = B_PRE_HANDOVER   yellow = placement  NOT EXECUTABLE"
        arr.markers.append(label)
        self.mpub.publish(arr)


def main():
    rclpy.init()
    node = Preview()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
