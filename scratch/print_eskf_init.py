import rosbag2_py, numpy as np, cv2
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri='/home/rv/RAM_VI_SLAM/slam_benchmark_run1', storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
tps = {t.name: t.type for t in reader.get_all_topics_and_types()}

init_acc = []
while reader.has_next():
    top, dat, t = reader.read_next()
    if top == '/camera/camera/accel/sample':
        m = deserialize_message(dat, get_message(tps[top]))
        init_acc.append([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        if len(init_acc) == 50:
            break

mean_acc = np.mean(init_acc, axis=0)
print('Mean of 50 Accel:', mean_acc)
print('Magnitude of Mean:', np.linalg.norm(mean_acc))
