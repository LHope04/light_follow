import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    PACKAGE_NAME = 'originbot_traffic_light'

    return LaunchDescription([
        Node(
            package=PACKAGE_NAME,
            executable='line_follower_node',
            name='line_follower',
            output='screen'
        ),
        Node(
            package=PACKAGE_NAME,
            executable='traffic_light_detector_node',
            name='traffic_light_detector',
            output='screen',
            parameters=[{
                'image_topic': '/image_raw', 
                'output_topic': '/traffic_light/state',
                'decision_topic': '/detect/traffic_decision',
                # ========== 【核心修改】==========
                'mode': 'roi',   # 保持固定区域(roi)模式，最稳妥
                'debug': True,   # 【关键】必须为 True，才能看到屏幕上的文字！
                'debug_topic': '/traffic_light/debug',
                # ================================
                'publish_unknown': True
            }]
        ),
        Node(
            package=PACKAGE_NAME,
            executable='intersection_action_manager_node',
            name='intersection_action_manager',
            output='screen',
            parameters=[{
                'odom_topic': '/odom',
                'turn_use_line_reacquire': True,
                'line_reacquire_topic': '/line_follow/line_found',
                'turn_min_angle_deg_for_line_exit': 55.0, 
                'turn_max_angle_deg': 105.0 
            }]
        ),
        Node(
            package=PACKAGE_NAME,
            executable='moving_adapter_node',
            name='moving_adapter',
            output='screen'
        )
    ])