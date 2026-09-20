#!/usr/bin/env python3
"""Play the saved STEP15 JointTrajectory in Gazebo. No search, no real robot.

Waits only for /fairino3_controller/follow_joint_trajectory, then sends
deployable Home→C segments immediately. Never calls the real gripper.
"""

from __future__ import annotations

import os
import sys
import time

import yaml
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import rclpy


_DEFAULT_TRAJ = os.path.expanduser(
    "~/fr_task_ws/src/fr_task_planner/config/step15_optimized_lift_to_a_trajectory.yaml"
)
_ARM = ["j1", "j2", "j3", "j4", "j5", "j6"]
_DEPLOYABLE = (
    "Home_to_PreGrasp",
    "PreGrasp_to_Grasp",
    "Grasp_to_Lift",
    "Lift_to_A",
    "A_to_B",
    "B_to_C_original_bottom",
)
_REAL_NODE_MARKERS = (
    "fairino_hardware",
    "real_bringup",
    "fr3_real_frozen_trajectory_executor",
)


def _time_s(pt: dict) -> float:
    tfs = pt.get("time_from_start") or {}
    return float(tfs.get("sec", 0)) + 1e-9 * float(tfs.get("nanosec", 0))


def _duration(seconds: float) -> Duration:
    seconds = max(0.0, float(seconds))
    sec = int(seconds)
    nsec = int(round((seconds - sec) * 1e9))
    if nsec >= 1_000_000_000:
        sec += 1
        nsec -= 1_000_000_000
    return Duration(sec=sec, nanosec=nsec)


def _segment_to_msg(seg: dict, scale: float) -> JointTrajectory:
    msg = JointTrajectory()
    msg.joint_names = list(seg.get("joint_names") or _ARM)
    scale = max(1e-3, float(scale))
    for pt in seg.get("points") or []:
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in (pt.get("positions") or [])]
        vels = pt.get("velocities") or []
        if vels:
            point.velocities = [float(v) / scale for v in vels]
        point.time_from_start = _duration(_time_s(pt) * scale)
        msg.points.append(point)
    return msg


class Step15SimPlayback(Node):
    def __init__(self) -> None:
        super().__init__("step15_sim_playback")
        self._declare("trajectory_file", _DEFAULT_TRAJ)
        self._declare("speed_scale", 1.0)
        self._declare("controller_wait_sec", 60.0)
        self._declare("action_name", "/fairino3_controller/follow_joint_trajectory")
        self.arm = ActionClient(
            self, FollowJointTrajectory, self.get_parameter("action_name").get_parameter_value().string_value
        )
        self.attach = self.create_publisher(Empty, "/fr3/grasp/attach", 10)
        self.detach = self.create_publisher(Empty, "/fr3/grasp/detach", 10)
        self.display = self.create_publisher(
            DisplayTrajectory, "/display_planned_path", 1
        )
        self._js_count = 0
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)

    def _on_js(self, _msg: JointState) -> None:
        self._js_count += 1

    def _declare(self, name: str, default) -> None:
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def _abort_if_real(self) -> None:
        sim = False
        if self.has_parameter("use_sim_time"):
            sim = bool(self.get_parameter("use_sim_time").value)
        if not sim:
            raise RuntimeError("STEP15 sim playback requires use_sim_time:=true")
        names = self.get_node_names()
        for name in names:
            lowered = name.lower()
            if any(marker in lowered for marker in _REAL_NODE_MARKERS):
                raise RuntimeError(f"refusing playback: real-stack node present ({name})")

    def _publish_preview(self, segments: list[dict], scale: float) -> None:
        disp = DisplayTrajectory()
        disp.model_id = "fairino3_v6_robot"
        if segments:
            first = _segment_to_msg(segments[0], scale)
            disp.trajectory_start.joint_state.name = list(first.joint_names)
            if first.points:
                disp.trajectory_start.joint_state.position = list(first.points[0].positions)
        for seg in segments:
            rt = RobotTrajectory()
            rt.joint_trajectory = _segment_to_msg(seg, scale)
            disp.trajectory.append(rt)
        self.display.publish(disp)

    def _send(self, traj: JointTrajectory) -> None:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        goal.path_tolerance = [
            JointTolerance(name=name, position=0.35, velocity=1.0) for name in traj.joint_names
        ]
        goal.goal_tolerance = [
            JointTolerance(name=name, position=0.08, velocity=0.15) for name in traj.joint_names
        ]
        goal.goal_time_tolerance = Duration(sec=8)
        future = self.arm.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("Gazebo controller rejected a trajectory segment")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(f"segment failed: {result.error_string}")

    def _burst(self, pub) -> None:
        for _ in range(4):
            pub.publish(Empty())
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)

    def _goto(self, names: list[str], positions: list[float], seconds: float, label: str) -> None:
        traj = JointTrajectory()
        traj.joint_names = list(names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in positions]
        point.time_from_start = _duration(seconds)
        traj.points = [point]
        self.get_logger().info(f"{label} ({seconds:.1f}s)")
        self._send(traj)

    def run(self) -> int:
        self._abort_if_real()
        path = os.path.expanduser(str(self.get_parameter("trajectory_file").value))
        scale = float(self.get_parameter("speed_scale").value)
        wait_sec = float(self.get_parameter("controller_wait_sec").value)
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        home = [float(v) for v in ((data.get("winner") or {}).get("home_joints_rad") or [])]
        segments = [
            seg
            for seg in (data.get("segments") or [])
            if seg.get("deployable") and seg.get("logical_segment") in _DEPLOYABLE
        ]
        if len(home) != 6:
            raise RuntimeError(f"winner.home_joints_rad missing in {path}")
        if len(segments) != 6:
            raise RuntimeError(f"expected 6 deployable segments, got {len(segments)} from {path}")
        self.get_logger().info("STEP15 Gazebo playback. No search. No real robot. No gripper service.")
        self.get_logger().info(f"trajectory={path} scale={scale:.3f}")
        deadline = time.monotonic() + wait_sec
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._js_count >= 8 and self.arm.wait_for_server(timeout_sec=0.05):
                break
            self.get_logger().info(
                f"waiting for live Gazebo joint_states={self._js_count} controller..."
            )
        else:
            raise RuntimeError("timed out waiting for Gazebo follow_joint_trajectory")
        self._abort_if_real()
        self._publish_preview(segments, scale)
        self.get_logger().info("controller ready — return Home, then Home→C")
        self._burst(self.detach)
        self._goto(_ARM, home, max(4.0, 6.0 * scale), "return to Home")
        for seg in segments:
            name = str(seg.get("logical_segment"))
            self.get_logger().info(f"playing {name} ({float(seg.get('duration') or 0.0) * scale:.2f}s)")
            self._send(_segment_to_msg(seg, scale))
            if name == "PreGrasp_to_Grasp":
                self.get_logger().info("Gazebo attach small_part (sim weld only)")
                self._burst(self.attach)
        self.get_logger().info("STEP15 sim playback finished")
        return 0


def main() -> int:
    rclpy.init()
    node = Step15SimPlayback()
    try:
        return node.run()
    except Exception as exc:  # noqa: BLE001
        node.get_logger().error(str(exc))
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
