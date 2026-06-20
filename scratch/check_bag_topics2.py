import rosbag2_py

bag_path = "/home/rv/RAM_VI_SLAM/slam_benchmark_run1"

reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
    rosbag2_py.ConverterOptions('cdr', 'cdr')
)

topics = [tp.name for tp in reader.get_all_topics_and_types()]
print("Topics in slam_benchmark_run1:")
for t in sorted(topics):
    print(" ", t)
