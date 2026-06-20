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
    if topic in [
        '/camera/camera/extrinsics/depth_to_accel',
        '/camera/camera/extrinsics/depth_to_gyro',
        '/camera/camera/extrinsics/depth_to_color',
        '/tf_static'
    ]:
        msg_class = get_message(topic_types[topic])
        msg = deserialize_message(data, msg_class)
        print(f"Topic: {topic}")
        print(msg)
        print("-" * 50)
