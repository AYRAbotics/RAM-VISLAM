import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from ram_vi_slam.eskf import ESKF
from ram_vi_slam.tracking import RGBDTracker

# Calibration constants
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

KF_TRANS_THRESH  = 0.08
KF_ROT_THRESH    = 0.08

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
        import cv2
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
        if enc == "bgr8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr
    elif enc == "16uc1":
        return np.frombuffer(msg.data, dtype=np.uint16).reshape((h, w))
    return None

def main():
    bag_path = '/home/rv/RAM_VI_SLAM/slam_benchmark_run1'
    if not os.path.exists(bag_path):
        print(f"Bag path {bag_path} does not exist. Please check configuration.")
        return
        
    reader = get_reader(bag_path)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    T_d2c = np.eye(4); T_d2c[:3, :3] = R_D2C; T_d2c[:3, 3] = T_D2C
    T_d2i = np.eye(4); T_d2i[:3, :3] = R_D2I; T_d2i[:3, 3] = T_D2I
    T_c2i = T_d2i @ np.linalg.inv(T_d2c)
    T_i2c = np.linalg.inv(T_c2i)

    K_d = np.array([[FX_D, 0, CX_D], [0, FY_D, CY_D], [0, 0, 1]], dtype=np.float64)

    eskf = ESKF()
    tracker = RGBDTracker(FX_C, FY_C, CX_C, CY_C)
    tracker.set_calibration(K_d, R_D2C, T_D2C)

    # Message buffers
    latest_color = None
    latest_depth = None
    latest_color_t = 0
    latest_depth_t = 0
    
    imu_history = []
    last_imu_t = None
    
    frame_count = 0
    kf_count = 0
    last_kf_pose = None
    T_wc = np.eye(4)
    
    latest_color_prev = None
    depth_m_prev = None

    print("Analyzing frames 0 to 100...")

    while reader.has_next() and frame_count < 100:
        topic, data, t_msg = reader.read_next()
        msg_type = topic_types.get(topic)
        if msg_type is None:
            continue
        msg_class = get_message(msg_type)
        msg = deserialize_message(data, msg_class)

        if topic == '/camera/camera/accel/sample':
            acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            imu_history.append((t_msg, 'accel', acc))
        elif topic == '/camera/camera/gyro/sample':
            gyr = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
            imu_history.append((t_msg, 'gyro', gyr))
        elif topic == '/camera/camera/color/image_raw':
            latest_color = image_to_numpy(msg)
            latest_color_t = t_msg
        elif topic == '/camera/camera/depth/image_rect_raw':
            latest_depth = image_to_numpy(msg)
            latest_depth_t = t_msg

        if latest_color is not None and latest_depth is not None:
            time_diff = abs(latest_color_t - latest_depth_t) * 1e-6
            if time_diff < 30.0:
                frame_t = min(latest_color_t, latest_depth_t)
                
                # Propagate ESKF
                imu_history.sort(key=lambda x: x[0])
                curr_acc = np.array([0.0, 0.0, -9.80665])
                curr_gyr = np.array([0.0, 0.0, 0.0])
                
                for imu_t, imu_type, imu_val in imu_history:
                    if imu_t < frame_t:
                        if imu_type == 'accel':
                            curr_acc = imu_val
                        elif imu_type == 'gyro':
                            curr_gyr = imu_val
                        
                        if last_imu_t is not None:
                            dt = (imu_t - last_imu_t) * 1e-9
                            if dt > 0:
                                eskf.predict(dt, curr_acc, curr_gyr)
                        last_imu_t = imu_t
                
                imu_history = [x for x in imu_history if x[0] >= frame_t]
                
                # Get tracking initial guess
                T_init = np.eye(4)
                if eskf.is_gravity_initialized:
                    T_wi = np.eye(4)
                    T_wi[0:3, 0:3] = eskf.R.as_matrix()
                    T_wi[0:3, 3] = eskf.p
                    T_init = T_wi @ T_i2c

                # GPU Depth Registration
                depth_m = tracker.register_depth(latest_depth)
                
                if frame_count == 0:
                    T_wc = T_init
                    success = True
                else:
                    success, T_wc = tracker.align_frames(
                        latest_color_prev, depth_m_prev, latest_color, depth_m, T_init
                    )
                
                if success:
                    # Update ESKF
                    T_wi = T_wc @ T_c2i
                    eskf.update(T_wi[0:3, 3], T_wi[0:3, 0:3])
                    
                    is_kf = False
                    if frame_count == 0:
                        is_kf = True
                        last_kf_pose = T_wc.copy()
                        kf_count += 1
                        print(f"Frame {frame_count}: Keyframe 0 spawned (initial frame)")
                    else:
                        T_rel = np.linalg.inv(last_kf_pose) @ T_wc
                        t_dist = np.linalg.norm(T_rel[0:3, 3])
                        r_dist = np.linalg.norm(Rotation.from_matrix(T_rel[0:3, 0:3]).as_rotvec())
                        if t_dist > KF_TRANS_THRESH or r_dist > KF_ROT_THRESH:
                            is_kf = True
                            last_kf_pose = T_wc.copy()
                            print(f"Frame {frame_count}: Keyframe {kf_count} spawned (trans: {t_dist:.3f}m, rot: {r_dist:.3f}rad)")
                            kf_count += 1
                            
                latest_color_prev = latest_color.copy()
                depth_m_prev = depth_m.copy()
                frame_count += 1
                latest_color = None
                latest_depth = None

    print(f"Analysis complete. Total analyzed frames: {frame_count}, total spawned keyframes: {kf_count}")

if __name__ == '__main__':
    main()