import numpy as np
import open3d as o3d
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import cv2
from ram_vi_slam.tracking import RGBDTracker

FX_C = 610.1809082;  FY_C = 610.26391602
CX_C = 337.1600647;  CY_C = 249.06201172
FX_D = 388.2460022;  FY_D = 388.2460022
CX_D = 313.19522095; CY_D = 243.97851562

R_D2C = np.array([
    [ 0.99999636, -0.00241311,  0.00117634],
    [ 0.00241741,  0.99999034, -0.00367048],
    [-0.00116747,  0.00367331,  0.99999255]
])
T_D2C = np.array([0.01454706, 0.00018594, 0.00039981])

bag_path = "/home/rv/RAM_VI_SLAM/slam_benchmark_run1"

reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
    rosbag2_py.ConverterOptions('cdr', 'cdr')
)
topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

tracker = RGBDTracker(FX_C, FY_C, CX_C, CY_C)
K_d_mat = np.array([
    [FX_D, 0.0,  CX_D],
    [0.0,  FY_D, CY_D],
    [0.0,  0.0,  1.0 ]
])
tracker.set_calibration(K_d_mat, R_D2C, T_D2C)

frames = []
while reader.has_next():
    topic, data, _ = reader.read_next()
    if topic == '/camera/camera/color/image_raw':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        color = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        frames.append(('color', color))
    elif topic == '/camera/camera/depth/image_rect_raw':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        frames.append(('depth', depth))
    
    # Check sync_frames count to break early
    latest_color = None
    latest_depth = None
    temp_sync = 0
    for t, img in frames:
        if t == 'color':
            latest_color = img
        else:
            latest_depth = img
        if latest_color is not None and latest_depth is not None:
            temp_sync += 1
            latest_color = None
            latest_depth = None
    if temp_sync >= 170:
        break

sync_frames = []
latest_color = None
latest_depth = None
for t, img in frames:
    if t == 'color':
        latest_color = img
    else:
        latest_depth = img
    if latest_color is not None and latest_depth is not None:
        sync_frames.append((latest_color, latest_depth))
        latest_color = None
        latest_depth = None

# Let's take frame 150 and 160 to have significant movement
src_color, src_depth = sync_frames[150]
tgt_color, tgt_depth = sync_frames[160]

src_depth_m = tracker.register_depth(src_depth)
tgt_depth_m = tracker.register_depth(tgt_depth)

src_color_o3d = o3d.geometry.Image(src_color)
src_depth_o3d = o3d.geometry.Image(src_depth_m)
tgt_color_o3d = o3d.geometry.Image(tgt_color)
tgt_depth_o3d = o3d.geometry.Image(tgt_depth_m)

src_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    src_color_o3d, src_depth_o3d, convert_rgb_to_intensity=True, depth_scale=1.0, depth_trunc=8.0
)
tgt_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    tgt_color_o3d, tgt_depth_o3d, convert_rgb_to_intensity=True, depth_scale=1.0, depth_trunc=8.0
)

option = o3d.pipelines.odometry.OdometryOption()
option.depth_diff_max = 0.07
option.depth_min = 0.1
option.depth_max = 8.0

success, T_odom, info = o3d.pipelines.odometry.compute_rgbd_odometry(
    src_rgbd, tgt_rgbd, tracker.intrinsic_o3d, np.eye(4),
    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
    option
)

print("compute_rgbd_odometry success:", success)
print("T_odom:\n", T_odom)

src_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(src_rgbd, tracker.intrinsic_o3d)
tgt_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(tgt_rgbd, tracker.intrinsic_o3d)
print("src_pcd points count:", len(src_pcd.points))
print("tgt_pcd points count:", len(tgt_pcd.points))
tgt_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

icp_result = o3d.pipelines.registration.registration_icp(
    src_pcd, tgt_pcd, 0.05, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)
print("icp_result.transformation:\n", icp_result.transformation)
src_pcd_temp1 = o3d.geometry.PointCloud(src_pcd)
src_pcd_temp1.transform(T_odom)
dists1 = src_pcd_temp1.compute_point_cloud_distance(tgt_pcd)
mean_dist1 = np.mean(dists1)

# Calculate alignment error for inv(T_odom)
src_pcd_temp2 = o3d.geometry.PointCloud(src_pcd)
src_pcd_temp2.transform(np.linalg.inv(T_odom))
dists2 = src_pcd_temp2.compute_point_cloud_distance(tgt_pcd)
mean_dist2 = np.mean(dists2)

print(f"Mean distance when applying T_odom to source: {mean_dist1:.6f}")
print(f"Mean distance when applying inv(T_odom) to source: {mean_dist2:.6f}")
if mean_dist1 < mean_dist2:
    print("CONCLUSION: T_odom maps source to target.")
else:
    print("CONCLUSION: T_odom maps target to source.")
