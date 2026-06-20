from setuptools import setup
import os
from glob import glob

package_name = 'ram_vi_slam'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rv',
    maintainer_email='rv@todo.todo',
    description='Dense Visual-Inertial SLAM based on ElasticFusion philosophy with ESKF and DINOv2 loop closure',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'slam_node = ram_vi_slam.slam_node:main',
            'offline_runner = ram_vi_slam.offline_runner:main',
        ],
    },
)
