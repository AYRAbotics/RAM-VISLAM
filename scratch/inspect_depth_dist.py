import os
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag_path = "/home/rv/RAM_VI_SLAM/slam_benchmark_run1"

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
        arr = np.frombuffer(msg.data, dtype=np.uint16)
        
        # Calculate distribution
        total = len(arr)
        zeros = np.sum(arr == 0)
        saturated = np.sum(arr == 65535)
        between_0_4m = np.sum((arr > 0) & (arr <= 4000))
        between_4_10m = np.sum((arr > 4000) & (arr <= 10000))
        above_10m = np.sum((arr > 10000) & (arr < 65535))
        
        print("Distribution of depth values:")
        print(f"  Zeros (invalid): {zeros} ({zeros/total*100:.1f}%)")
        print(f"  Saturated (65535): {saturated} ({saturated/total*100:.1f}%)")
        print(f"  0.0m - 4.0m: {between_0_4m} ({between_0_4m/total*100:.1f}%)")
        print(f"  4.0m - 10.0m: {between_4_10m} ({between_4_10m/total*100:.1f}%)")
        print(f"  > 10m: {above_10m} ({above_10m/total*100:.1f}%)")
        break
