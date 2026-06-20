import os
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import cv2

bag_path = "/home/rv/RAM_VI_SLAM/slam_benchmark_run1"

reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
    rosbag2_py.ConverterOptions('cdr', 'cdr')
)
topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

frame_count = 0
latest_color = None
latest_depth = None
latest_color_t = 0
latest_depth_t = 0

while reader.has_next() and frame_count < 1000:
    topic, data, t_msg = reader.read_next()
    if topic == '/camera/camera/color/image_raw':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        latest_color = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        latest_color_t = t_msg
    elif topic == '/camera/camera/depth/image_rect_raw':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        latest_depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        latest_depth_t = t_msg
        
    if latest_color is not None and latest_depth is not None:
        time_diff = abs(latest_color_t - latest_depth_t) * 1e-6
        if time_diff < 30.0:
            if frame_count % 150 == 0:
                out_path = f"/home/rv/.gemini/antigravity-ide/brain/e6ac1704-98be-417d-b3b9-73c912205c1b/color_frame_{frame_count}.png"
                cv2.imwrite(out_path, latest_color)
                print(f"Saved {out_path}")
            frame_count += 1
            latest_color = None
            latest_depth = None
