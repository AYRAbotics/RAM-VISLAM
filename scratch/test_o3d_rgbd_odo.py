import numpy as np
import open3d as o3d
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import cv2

FX_C = 610.1809082;  FY_C = 610.26391602
CX_C = 337.1600647;  CY_C = 249.06201172
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    640, 480, FX_C, FY_C, CX_C, CY_C
)

def get_reader(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('cdr', 'cdr')
    )
    return reader

def image_to_numpy(msg):
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
        if enc == "bgr8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr
    elif enc == "16uc1":
        return np.frombuffer(msg.data, dtype=np.uint16).reshape((h, w))
    return None

reader = get_reader('/home/rv/RAM_VI_SLAM/slam_benchmark_run1')
topic_types = {tp.name: tp.type for tp in reader.get_all_topics_and_types()}

latest_color = None
latest_depth = None
synced_frames = []

while reader.has_next() and len(synced_frames) < 250:
    topic, data, t_msg = reader.read_next()
    msg_type = topic_types.get(topic)
    if msg_type is None:
        continue
    msg_class = get_message(msg_type)
    msg = deserialize_message(data, msg_class)
    
    if topic == '/camera/camera/color/image_raw':
        latest_color = image_to_numpy(msg)
    elif topic == '/camera/camera/depth/image_rect_raw':
        latest_depth = image_to_numpy(msg)
        
    if latest_color is not None and latest_depth is not None:
        synced_frames.append((latest_color.copy(), latest_depth.copy() / 1000.0))
        latest_color = None
        latest_depth = None

# Compare active movement: frame 50 (src) and frame 180 (tgt)
src_rgb, src_depth = synced_frames[50]
tgt_rgb, tgt_depth = synced_frames[180]

src_color_o3d = o3d.geometry.Image(src_rgb)
src_depth_o3d = o3d.geometry.Image(src_depth.astype(np.float32))
tgt_color_o3d = o3d.geometry.Image(tgt_rgb)
tgt_depth_o3d = o3d.geometry.Image(tgt_depth.astype(np.float32))

src_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    src_color_o3d, src_depth_o3d, convert_rgb_to_intensity=True,
    depth_scale=1.0, depth_trunc=8.0
)
tgt_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    tgt_color_o3d, tgt_depth_o3d, convert_rgb_to_intensity=True,
    depth_scale=1.0, depth_trunc=8.0
)

# Compute Odometry
option = o3d.pipelines.odometry.OdometryOption()
success, T_odom, info = o3d.pipelines.odometry.compute_rgbd_odometry(
    src_rgbd, tgt_rgbd, intrinsic, np.eye(4),
    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
    option
)

print("compute_rgbd_odometry success:", success)
print("T_odom translation:", T_odom[0:3, 3])

src_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(src_rgbd, intrinsic)
tgt_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(tgt_rgbd, intrinsic)
tgt_pcd.estimate_normals()

# Test 1: ICP with T_odom as init guess
icp_res1 = o3d.pipelines.registration.registration_icp(
    src_pcd, tgt_pcd, 0.20, T_odom,
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)
print("Test 1 (T_odom as init) fitness:", icp_res1.fitness)
print("Test 1 translation:", icp_res1.transformation[0:3, 3])

# Test 2: ICP with T_odom^-1 as init guess
icp_res2 = o3d.pipelines.registration.registration_icp(
    src_pcd, tgt_pcd, 0.20, np.linalg.inv(T_odom),
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)
print("Test 2 (inv(T_odom) as init) fitness:", icp_res2.fitness)
print("Test 2 translation:", icp_res2.transformation[0:3, 3])
