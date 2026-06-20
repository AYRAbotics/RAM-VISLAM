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

accel_count = 0
gyro_count = 0

print("Inspecting IMU messages:")
while reader.has_next() and (accel_count < 3 or gyro_count < 3):
    topic, data, _ = reader.read_next()
    if topic == '/camera/camera/accel/sample':
        if accel_count < 3:
            msg = deserialize_message(data, get_message(topic_types[topic]))
            print(f"Accel: frame_id='{msg.header.frame_id}'")
            print(f"  accel = [{msg.linear_acceleration.x}, {msg.linear_acceleration.y}, {msg.linear_acceleration.z}]")
            accel_count += 1
    elif topic == '/camera/camera/gyro/sample':
        if gyro_count < 3:
            msg = deserialize_message(data, get_message(topic_types[topic]))
            print(f"Gyro: frame_id='{msg.header.frame_id}'")
            print(f"  gyro  = [{msg.angular_velocity.x}, {msg.angular_velocity.y}, {msg.angular_velocity.z}]")
            gyro_count += 1
