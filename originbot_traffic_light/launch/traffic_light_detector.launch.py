import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('originbot_traffic_light')
    params_file = os.path.join(pkg_share, 'config', 'params.yaml')

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
        description='Publish debug image topic',
    )

    node = Node(
        package='originbot_traffic_light',
        executable='traffic_light_detector_node',
        name='traffic_light_detector',
        output='screen',
        parameters=[
            params_file,
            {
                'image_topic': LaunchConfiguration('image_topic'),
                'mode': LaunchConfiguration('mode'),
                'debug': LaunchConfiguration('debug'),
            },
        ],
    )

    return LaunchDescription([
        image_topic_arg,
        mode_arg,
        debug_arg,
        node,
    ])
