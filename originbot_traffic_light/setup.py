import os
from glob import glob
from setuptools import setup

package_name = 'originbot_traffic_light'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='GuYueHome',
    maintainer_email='support@ps-micro.com',
    description='ROI-based traffic light/sign board detection for OriginBot',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'traffic_light_detector_node = originbot_traffic_light.traffic_light_detector_node:main',
            'intersection_action_manager_node = originbot_traffic_light.intersection_action_manager_node:main',
            'moving_adapter_node = originbot_traffic_light.moving_adapter_node:main',
            # [新增] 把我们新写的巡线节点注册进来
            'line_follower_node = originbot_traffic_light.line_follower_node:main', 
        ],
    },
)