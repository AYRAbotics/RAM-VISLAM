import os
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
from ram_vi_slam.tracking import RGBDTracker
from ram_vi_slam.imu_models import GyroGuidedEstimator
from ram_vi_slam.mapping import SurfelMap

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
R_D2I = np.eye(3)
T_D2I = np.array([-0.00552, 0.0051, 0.01174])

T_d2c = np.eye(4); T_d2c[0:3, 0:3] = R_D2C; T_d2c[0:3, 3] = T_D2C
T_d2i = np.eye(4); T_d2i[0:3, 0:3] = R_D2I; T_d2i[0:3, 3] = T_D2I
T_c2i = T_d2i @ np.linalg.inv(T_d2c)

roll_rad = np.radians(-3.5)
R_corr = Rotation.from_euler('xyz', [roll_rad, 0.0, 0.0]).as_matrix()
T_c2i[0:3, 0:3] = T_c2i[0:3, 0:3] @ R_corr

T_i2c = np.linalg.inv(T_c2i)

bag_path = "/home/rv/RAM_VI_SLAM/slam_benchmark_run1"
reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
    rosbag2_py.ConverterOptions('cdr', 'cdr')
)
topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

tracker = RGBDTracker(FX_C, FY_C, CX_C, CY_C)
K_d_mat = np.array([[FX_D, 0.0,  CX_D], [0.0,  FY_D, CY_D], [0.0,  0.0,  1.0]])
tracker.set_calibration(K_d_mat, R_D2C, T_D2C)

eskf = GyroGuidedEstimator()
surfel_map = SurfelMap(FX_C, FY_C, CX_C, CY_C)

# 1. 3D Open3D Visualizer Setup
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Test Visualizer", width=1024, height=768, visible=False)

# R_w2v
R_w2v = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0]
])
T_w2v = np.eye(4)
T_w2v[0:3, 0:3] = R_w2v

pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

frustum = o3d.geometry.LineSet()
vis.add_geometry(frustum)

base_frustum_vertices = np.array([
    [0, 0, 0],
    [-0.1, -0.075, 0.15],
    [0.1, -0.075, 0.15],
    [0.1, 0.075, 0.15],
    [-0.1, 0.075, 0.15]
])
frustum_lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
frustum_colors = [[0, 1, 0] for _ in range(8)]

latest_color = None
latest_depth = None
latest_color_t = 0
latest_depth_t = 0
frame_count = 0

imu_history = []
last_imu_t = None
curr_acc = np.array([0.0, 0.0, -9.80665])
curr_gyr = np.array([0.0, 0.0, 0.0])
T_wc = np.eye(4)
T_wc_prev = np.eye(4)
last_frame_t = None

print("Running test visualizer...")
while reader.has_next() and frame_count < 100:
    topic, data, t_msg = reader.read_next()
    if topic == '/camera/camera/accel/sample':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        imu_history.append((t_msg, 'accel', acc))
    elif topic == '/camera/camera/gyro/sample':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        gyr = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        imu_history.append((t_msg, 'gyro', gyr))
    elif topic == '/camera/camera/color/image_raw':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        latest_color = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        latest_color = cv2.cvtColor(latest_color, cv2.COLOR_BGR2RGB)
        latest_color_t = t_msg
    elif topic == '/camera/camera/depth/image_rect_raw':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        latest_depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        latest_depth_t = t_msg
        
    if latest_color is not None and latest_depth is not None:
        time_diff = abs(latest_color_t - latest_depth_t) * 1e-6
        if time_diff < 30.0:
            frame_t = min(latest_color_t, latest_depth_t)
            frame_dt = 0.033
            if frame_count > 0 and last_frame_t is not None:
                frame_dt = (frame_t - last_frame_t) * 1e-9
            last_frame_t = frame_t
            
            # Propagate GyroGuidedEstimator
            imu_history.sort(key=lambda x: x[0])
            accel_received = False
            for imu_t, imu_type, imu_val in imu_history:
                if imu_t < frame_t:
                    if imu_type == 'accel':
                        curr_acc = imu_val
                        accel_received = True
                    elif imu_type == 'gyro':
                        curr_gyr = imu_val
                    if last_imu_t is not None and accel_received:
                        dt = (imu_t - last_imu_t) * 1e-9
                        if dt > 0:
                            eskf.predict(dt, curr_acc, curr_gyr)
                    last_imu_t = imu_t
            imu_history = [x for x in imu_history if x[0] >= frame_t]
            
            T_pred = np.eye(4)
            if eskf.is_gravity_initialized:
                T_wi = np.eye(4)
                T_wi[0:3, 0:3] = eskf.R.as_matrix()
                T_wi[0:3, 3] = eskf.p
                T_pred = T_wi @ T_c2i
            
            depth_m = tracker.register_depth(latest_depth)
            
            if frame_count == 0:
                T_wc = T_pred
                track_success = True
            else:
                T_init_rel = np.linalg.inv(T_wc_prev) @ T_pred
                T_init_rel[0:3, 3] = T_init_rel[0:3, 3] * 0.5
                track_success, T_rel = tracker.align_frames(
                    latest_color_prev, depth_m_prev, latest_color, depth_m, T_init_rel
                )
                if track_success:
                    T_wc = T_wc_prev @ T_rel
                else:
                    T_wc = T_pred
                    
            if track_success:
                T_wi = T_wc @ T_i2c
                eskf.update(T_wi[0:3, 3], T_wi[0:3, 0:3], dt=frame_dt)
                T_wi_filt = np.eye(4)
                T_wi_filt[0:3, 0:3] = eskf.R.as_matrix()
                T_wi_filt[0:3, 3] = eskf.p
                T_wc = T_wi_filt @ T_c2i
                
            surfel_map.fuse_frame(latest_color, depth_m, T_wc, frame_count, 0)
            
            # Update visualizer
            T_wc_vis = T_w2v @ T_wc
            R_vis = T_wc_vis[0:3, 0:3]
            t_vis = T_wc_vis[0:3, 3]
            
            # Update Point Cloud
            pos = surfel_map.positions[:surfel_map.active_n].cpu().numpy()
            col = surfel_map.colors[:surfel_map.active_n].cpu().numpy()
            pos_vis = pos @ R_w2v.T
            
            vis.remove_geometry(pcd, reset_bounding_box=False)
            pcd.points = o3d.utility.Vector3dVector(pos_vis.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64))
            vis.add_geometry(pcd, reset_bounding_box=False)
            
            # Update frustum
            transformed_vertices = (base_frustum_vertices @ R_vis.T) + t_vis
            frustum.points = o3d.utility.Vector3dVector(transformed_vertices)
            frustum.lines = o3d.utility.Vector2iVector(frustum_lines)
            frustum.colors = o3d.utility.Vector3dVector(frustum_colors)
            vis.update_geometry(frustum)
            
            view_ctl = vis.get_view_control()
            if view_ctl is not None:
                view_ctl.set_constant_z_near(0.01)
                view_ctl.set_constant_z_far(100.0)
                if frame_count == 1:
                    # View looking from top-front-right (Y is up, -Z is forward)
                    view_ctl.set_front([0.5, 0.6, -0.6]) # positive Y!
                    view_ctl.set_lookat([0.0, 0.0, 0.5])
                    view_ctl.set_up([0.0, 1.0, 0.0])
                    view_ctl.set_zoom(0.8)
                    
            vis.poll_events()
            vis.update_renderer()
            
            latest_color_prev = latest_color.copy()
            depth_m_prev = depth_m.copy()
            T_wc_prev = T_wc.copy()
            frame_count += 1
            latest_color = None
            latest_depth = None

# Capture screen
os.makedirs("/home/rv/.gemini/antigravity-ide/brain/e6ac1704-98be-417d-b3b9-73c912205c1b", exist_ok=True)
screenshot_path = "/home/rv/.gemini/antigravity-ide/brain/e6ac1704-98be-417d-b3b9-73c912205c1b/test_viewpoint_screen.png"
vis.capture_screen_image(screenshot_path)
print(f"Captured screen image to {screenshot_path}")
vis.destroy_window()
