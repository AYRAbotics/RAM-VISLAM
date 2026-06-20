import os
import numpy as np
import open3d as o3d
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import cv2
from ram_vi_slam.tracking import RGBDTracker

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

bag_path = "/home/rv/RAM_VI_SLAM/slam_benchmark_run1"

reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
    rosbag2_py.ConverterOptions('cdr', 'cdr')
)
topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

tracker = RGBDTracker(FX_C, FY_C, CX_C, CY_C)
K_d_mat = np.array([
    [FX_D, 0.0,  CX_D],
    [0.0,  FY_D, CY_D],
    [0.0,  0.0,  1.0 ]
])
tracker.set_calibration(K_d_mat, R_D2C, T_D2C)

latest_color = None
latest_depth = None
latest_color_t = 0
latest_depth_t = 0
frame_count = 0

latest_color_prev = None
depth_m_prev = None
T_wc = np.eye(4)

while reader.has_next() and frame_count < 200:
    topic, data, t_msg = reader.read_next()
    if topic == '/camera/camera/color/image_raw':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        latest_color = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        latest_color = cv2.cvtColor(latest_color, cv2.COLOR_BGR2RGB)
        latest_color_t = t_msg
    elif topic == '/camera/camera/depth/image_rect_raw':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        latest_depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        latest_depth_t = t_msg
        
    if latest_color is not None and latest_depth is not None:
        time_diff = abs(latest_color_t - latest_depth_t) * 1e-6
        if time_diff < 30.0:
            depth_m = tracker.register_depth(latest_depth)
            if frame_count == 0:
                T_wc = np.eye(4)
            else:
                success, T_rel = tracker.align_frames(
                    latest_color_prev, depth_m_prev, latest_color, depth_m, np.eye(4)
                )
                if success:
                    T_wc = T_wc @ T_rel
                if frame_count >= 150:
                    print(f"Frame {frame_count} Pose T_wc position: {T_wc[0:3, 3]}")
            
            latest_color_prev = latest_color.copy()
            depth_m_prev = depth_m.copy()
            frame_count += 1
            latest_color = None
            latest_depth = None
