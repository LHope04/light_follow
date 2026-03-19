"""Launch the full intersection pipeline.

Starts both the traffic-light detector and the intersection action manager
so the complete chain

  camera → detect_traffic_light → /detect/traffic_decision
         → intersection_action_manager → /control/moving/state

is active in a single launch invocation.

Usage
-----
  ros2 launch originbot_traffic_light intersection_action.launch.py
  ros2 launch originbot_traffic_light intersection_action.launch.py mode:=card debug:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('originbot_traffic_light')
    detector_params = os.path.join(pkg_share, 'config', 'params.yaml')
    manager_params = os.path.join(pkg_share, 'config', 'intersection_params.yaml')

    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/image_raw',
        description='Input image topic',
    )
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='roi',
        description='Detection mode: roi or card',
    )
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Publish debug image topic (card mode only)',
    )

    detector_node = Node(
        package='originbot_traffic_light',
        executable='traffic_light_detector_node',
        name='traffic_light_detector',
        output='screen',
        parameters=[
            detector_params,
            {
                'image_topic': LaunchConfiguration('image_topic'),
                'mode': LaunchConfiguration('mode'),
                'debug': LaunchConfiguration('debug'),
            },
        ],
    )

    manager_node = Node(
        package='originbot_traffic_light',
        executable='intersection_action_manager_node',
        name='intersection_action_manager',
        output='screen',
        parameters=[manager_params],
    )

    return LaunchDescription([
        image_topic_arg,
        mode_arg,
        debug_arg,
        detector_node,
        manager_node,
    ])
