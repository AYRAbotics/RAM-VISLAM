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

count = 0
while reader.has_next() and count < 10:
    topic, data, _ = reader.read_next()
    if topic == '/camera/camera/depth/image_rect_raw':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        print("--- Depth Image ---")
        print("Height:", msg.height, "Width:", msg.width)
        print("Encoding:", msg.encoding)
        print("Step:", msg.step)
        arr = np.frombuffer(msg.data, dtype=np.uint16)
        print("Raw data size:", len(msg.data))
        print("Array shape:", arr.shape)
        print("Min:", arr.min(), "Max:", arr.max(), "Mean:", arr.mean())
        count += 1
