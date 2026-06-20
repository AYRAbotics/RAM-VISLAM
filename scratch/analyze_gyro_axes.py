import rosbag2_py
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag_path = "/home/rv/RAM_VI_SLAM/slam_benchmark_run1"

reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
    rosbag2_py.ConverterOptions('cdr', 'cdr')
)
topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

accel_data = []
gyro_data = []

while reader.has_next():
    topic, data, _ = reader.read_next()
    if topic == '/camera/camera/accel/sample':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        accel_data.append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
    elif topic == '/camera/camera/gyro/sample':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        gyro_data.append([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

accel_data = np.array(accel_data)
gyro_data = np.array(gyro_data)

print(f"Total accel samples: {len(accel_data)}")
print(f"Total gyro samples: {len(gyro_data)}")

print("\nAccelerometer statistics (X, Y, Z):")
print("  Mean: ", np.mean(accel_data, axis=0))
print("  Std:  ", np.std(accel_data, axis=0))
print("  Min:  ", np.min(accel_data, axis=0))
print("  Max:  ", np.max(accel_data, axis=0))

print("\nGyroscope statistics (X, Y, Z):")
print("  Mean: ", np.mean(gyro_data, axis=0))
print("  Std:  ", np.std(gyro_data, axis=0))
print("  Min:  ", np.min(gyro_data, axis=0))
print("  Max:  ", np.max(gyro_data, axis=0))
