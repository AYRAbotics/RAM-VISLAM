import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri='/home/rv/RAM_VI_SLAM/slam_benchmark_run1', storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
tps = {t.name: t.type for t in reader.get_all_topics_and_types()}

count = 0
while reader.has_next():
    top, dat, t = reader.read_next()
    if top == '/camera/camera/gyro/sample':
        m = deserialize_message(dat, get_message(tps[top]))
        gx, gy, gz = m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z
        if abs(gy) > 0.5 or abs(gx) > 0.5 or abs(gz) > 0.5:
            print(f"Time {t}: Gyro [{gx:.3f}, {gy:.3f}, {gz:.3f}]")
            count += 1
            if count > 20: break
