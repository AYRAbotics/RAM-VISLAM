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

color_times = []
depth_times = []

count = 0
while reader.has_next() and count < 1000:
    topic, data, t_msg = reader.read_next()
    if topic == '/camera/camera/color/image_raw':
        color_times.append(t_msg)
    elif topic == '/camera/camera/depth/image_rect_raw':
        depth_times.append(t_msg)
    count += 1

print(f"Read {len(color_times)} color frames, {len(depth_times)} depth frames")
if len(color_times) > 0 and len(depth_times) > 0:
    # Print first few timestamps
    print("Color stamps: ", [t * 1e-9 for t in color_times[:5]])
    print("Depth stamps: ", [t * 1e-9 for t in depth_times[:5]])
    
    # Calculate differences between closest frames
    diffs = []
    for ct in color_times[:20]:
        closest_dt = min(depth_times, key=lambda x: abs(x - ct))
        diffs.append(abs(ct - closest_dt) * 1e-6)  # ms
    print("Differences between closest (ms):", [f"{d:.2f}" for d in diffs])
