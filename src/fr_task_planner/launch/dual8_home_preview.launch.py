#!/usr/bin/env python3
"""Publish DUAL-8-HOME RViz preview from the plan-only YAML. Not executable."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    preview_yaml = LaunchConfiguration("preview_yaml").perform(context)
    return [
        LogInfo(
            msg=[
                "DUAL-8-HOME RVIZ PREVIEW ONLY / NOT EXECUTABLE. yaml=",
                preview_yaml,
            ]
        ),
        Node(
            package="fr_task_planner",
            executable="dual_complete_six_face_rviz_preview.py",
            name="dual8_home_rviz_preview",
            output="screen",
            emulate_tty=True,
            arguments=[preview_yaml],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "preview_yaml",
                default_value="/tmp/dual8_home_preview.yaml",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
