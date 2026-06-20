import os
import numpy as np
import torch
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from ram_vi_slam.tracking import RGBDTracker

bag_path = "/home/rv/RAM_VI_SLAM/slam_benchmark_run1"

# Calibration constants
FX_C = 610.1809082;  FY_C = 610.26391602
CX_C = 337.1600647;  CY_C = 249.06201172
FX_D = 388.2460022;  FY_D = 388.2460022
CX_D = 313.19522095; CY_D = 243.97851562

R_D2C = np.array([
    [ 0.99999636, -0.00241311,  0.00117634],
    [ 0.00241741,  0.99999034, -0.00367048],
    [-0.00116747,  0.00367331,  0.99999255]
])
T_D2C = np.array([0.01454706, 0.00018594, 0.00039981])

tracker = RGBDTracker(FX_C, FY_C, CX_C, CY_C)
K_d_mat = np.array([
    [FX_D, 0.0,  CX_D],
    [0.0,  FY_D, CY_D],
    [0.0,  0.0,  1.0 ]
])
tracker.set_calibration(K_d_mat, R_D2C, T_D2C)

reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
    rosbag2_py.ConverterOptions('cdr', 'cdr')
)
topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

while reader.has_next():
    topic, data, _ = reader.read_next()
    if topic == '/camera/camera/depth/image_rect_raw':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        depth_np = np.frombuffer(msg.data, dtype=np.uint16).reshape((480, 640))
        
        # Test registration
        depth_m = tracker.register_depth(depth_np)
        
        # Print a 10x10 patch in the center
        patch = depth_m[235:245, 315:325]
        print("Center 10x10 patch of registered depth (m):")
        print(patch)
        
        print("Non-zero values in center patch count:", np.sum(patch > 0.0))
        break
