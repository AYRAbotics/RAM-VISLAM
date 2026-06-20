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

mags = []
while reader.has_next() and len(mags) < 100:
    topic, data, _ = reader.read_next()
    if topic == '/camera/camera/accel/sample':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        mag = np.linalg.norm(acc)
        mags.append(mag)

print("First 100 Accelerometer Magnitudes:")
print(f"Mean: {np.mean(mags):.4f}")
print(f"Min: {np.min(mags):.4f}")
print(f"Max: {np.max(mags):.4f}")
print(f"Sample values: {mags[:20]}")
