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

while reader.has_next() and len(synced_frames) < 100:
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

# Frame pairs to test
src_rgb, src_depth = synced_frames[50]
tgt_rgb, tgt_depth = synced_frames[90] # 40 frames gap

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

# Mock a constant velocity initial guess mapping tgt -> src (e.g. slight movement forward)
T_init_guess_tgt_to_src = np.eye(4)
T_init_guess_tgt_to_src[2, 3] = -0.15 # robot predicted moving forward by 15cm

# --- 1. Original Logic ---
# T_init is passed as T_init_guess_tgt_to_src
# In compute_rgbd_odometry, it passes T_init directly (which is tgt -> src)
option = o3d.pipelines.odometry.OdometryOption()
success_orig, T_odom_orig, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
    src_rgbd, tgt_rgbd, intrinsic, T_init_guess_tgt_to_src,
    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
    option
)
# In ICP, it passes np.linalg.inv(T_odom_orig)
src_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(src_rgbd, intrinsic)
tgt_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(tgt_rgbd, intrinsic)
tgt_pcd.estimate_normals()

icp_res_orig = o3d.pipelines.registration.registration_icp(
    src_pcd, tgt_pcd, 0.05, np.linalg.inv(T_odom_orig),
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

print("Original Logic:")
print("Success:", success_orig)
print("ICP Fitness:", icp_res_orig.fitness)
print("ICP Transformation translation:", icp_res_orig.transformation[0:3, 3])

# --- 2. Corrected Logic ---
# Initial guess passed to compute_rgbd_odometry is inv(T_init_guess_tgt_to_src) which is src -> tgt point transform
T_init_guess_src_to_tgt = np.linalg.inv(T_init_guess_tgt_to_src)
success_corr, T_odom_corr, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
    src_rgbd, tgt_rgbd, intrinsic, T_init_guess_src_to_tgt,
    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
    option
)
# In ICP, it passes T_odom_corr directly
icp_res_corr = o3d.pipelines.registration.registration_icp(
    src_pcd, tgt_pcd, 0.05, T_odom_corr,
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

print("\nCorrected Logic:")
print("Success:", success_corr)
print("ICP Fitness:", icp_res_corr.fitness)
print("ICP Transformation translation:", icp_res_corr.transformation[0:3, 3])
