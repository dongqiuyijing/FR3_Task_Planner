#!/usr/bin/env python3
"""Offline RViz marker preview for the blocked B-place geometry candidate.

Publishes only visualization_msgs/MarkerArray.  It does not create actions,
services, trajectory messages, or gripper commands.
"""
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class PlaceGeometryPreview(Node):
    def __init__(self):
        super().__init__('keypose_v1_b_place_geometry_preview')
        self.pub = self.create_publisher(MarkerArray, '/keypose_v1/b_place_geometry_preview', 1)
        self.timer = self.create_timer(0.5, self.publish)

    @staticmethod
    def cylinder(marker_id, z, r, g, b, label):
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.ns = 'b_place_geometry'
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = -0.35
        marker.pose.position.y = 0.50
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = 0.015
        marker.scale.z = 0.035
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = r, g, b, 0.65
        marker.lifetime.sec = 2
        text = Marker()
        text.header.frame_id = 'world'
        text.ns = 'b_place_geometry_label'
        text.id = marker_id
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = -0.35
        text.pose.position.y = 0.50
        text.pose.position.z = z + 0.030
        text.pose.orientation.w = 1.0
        text.scale.z = 0.025
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        text.text = label
        text.lifetime.sec = 2
        return marker, text

    def publish(self):
        out = MarkerArray()
        table = Marker()
        table.header.frame_id = 'world'
        table.ns = 'b_place_geometry'
        table.id = 10
        table.type = Marker.CUBE
        table.action = Marker.ADD
        table.pose.position.x, table.pose.position.y, table.pose.position.z = -0.35, 0.50, 0.749
        table.pose.orientation.w = 1.0
        table.scale.x, table.scale.y, table.scale.z = 0.30, 0.30, 0.002
        table.color.r, table.color.g, table.color.b, table.color.a = 0.55, 0.35, 0.12, 0.75
        table.lifetime.sec = 2
        out.markers.append(table)
        # Red: user's centre, which puts 2.5 mm of the 35 mm cylinder in table.
        out.markers.extend(self.cylinder(1, 0.765, 1.0, 0.1, 0.1, 'requested center: bottom 0.7475 (collision)'))
        # Green: contact-consistent centre; reference only, never executed.
        out.markers.extend(self.cylinder(2, 0.7675, 0.1, 1.0, 0.1, 'reference only: bottom 0.7500'))
        self.pub.publish(out)


def main():
    rclpy.init()
    node = PlaceGeometryPreview()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
