import os
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import cv2
from scipy.spatial.transform import Rotation
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
K_d_mat = np.array([[FX_D, 0.0,  CX_D], [0.0,  FY_D, CY_D], [0.0,  0.0,  1.0]])
tracker.set_calibration(K_d_mat, R_D2C, T_D2C)

latest_color = None
latest_depth = None
latest_color_t = 0
latest_depth_t = 0
frame_count = 0

latest_color_prev = None
depth_m_prev = None

# Gyro integration variables
gyro_history = []
T_wc_prev = np.eye(4)

print("Running comparison on turns...")
while reader.has_next() and frame_count < 300:
    topic, data, t_msg = reader.read_next()
    if topic == '/camera/camera/gyro/sample':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        gyr = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        gyro_history.append((t_msg, gyr))
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
            
            # Find gyro samples in between frames
            curr_gyros = [val for t, val in gyro_history if t < frame_t]
            gyro_history = [x for x in gyro_history if x[0] >= frame_t]
            
            depth_m = tracker.register_depth(latest_depth)
            
            if frame_count > 0:
                # Integrate gyro between last frame and this frame
                # As a simple approximation, sum (gyro * dt)
                # Let's assume average gyro rate or simple sum
                # The total time delta between frames:
                dt = (frame_t - frame_t_prev) * 1e-9
                if len(curr_gyros) > 0:
                    mean_gyro = np.mean(curr_gyros, axis=0)
                else:
                    mean_gyro = np.zeros(3)
                
                gyro_rot = mean_gyro * dt
                gyro_angle = np.linalg.norm(gyro_rot)
                
                # Visual Odometry relative pose (using T_init = Identity)
                success, T_rel = tracker.align_frames(
                    latest_color_prev, depth_m_prev, latest_color, depth_m, np.eye(4)
                )
                
                if success:
                    vo_rotvec = Rotation.from_matrix(T_rel[0:3, 0:3]).as_rotvec()
                    vo_angle = np.linalg.norm(vo_rotvec)
                    vo_trans = np.linalg.norm(T_rel[0:3, 3])
                    
                    # If there is a turn (gyro angle > 0.01 radians or VO angle > 0.5 degrees)
                    if gyro_angle > 0.005 or vo_angle > 0.005:
                        print(f"F{frame_count:03d} (dt={dt:.3f}s):")
                        print(f"  Gyro: RotVec = {gyro_rot}, Angle = {np.degrees(gyro_angle):.3f} deg")
                        print(f"  VO:   RotVec = {vo_rotvec}, Angle = {np.degrees(vo_angle):.3f} deg, Trans = {vo_trans*100:.2f} cm")
                        # Compare direction of Y-axis rotation (yaw)
                        # In optical frame, Y-axis is vertical
                        print(f"  Ratio (Gyro_Y / VO_Y): {mean_gyro[1]*dt / (vo_rotvec[1] + 1e-9):.3f}")
                
            latest_color_prev = latest_color.copy()
            depth_m_prev = depth_m.copy()
            frame_t_prev = frame_t
            frame_count += 1
            latest_color = None
            latest_depth = None
