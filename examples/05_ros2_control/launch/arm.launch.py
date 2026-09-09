"""Bring up the mock arm, the controller manager, and the two controllers."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

PACKAGE = "jax_arm_controller"


def generate_launch_description():
    """Three nodes: the state publisher, the manager, and one spawner."""
    share = get_package_share_directory(PACKAGE)
    with open(os.path.join(share, "urdf", "arm.urdf")) as handle:
        robot_description = handle.read()
    controllers = os.path.join(share, "config", "controllers.yaml")

    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
                output="both",
            ),
            Node(
                package="controller_manager",
                executable="ros2_control_node",
                parameters=[
                    {"robot_description": robot_description},
                    controllers,
                ],
                output="both",
            ),
            # One spawner for both: the broadcaster is what publishes
            # /joint_states, which is the evidence that the arm moved.
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_state_broadcaster", PACKAGE],
                output="both",
            ),
        ]
    )
