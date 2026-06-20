import os
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import cv2
from scipy.spatial.transform import Rotation
from ram_vi_slam.tracking import RGBDTracker
from ram_vi_slam.eskf import ESKF

def run_trajectory(roll_offset=0.0):
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

    T_d2c = np.eye(4); T_d2c[0:3, 0:3] = R_D2C; T_d2c[0:3, 3] = T_D2C
    T_d2i = np.eye(4); T_d2i[0:3, 0:3] = R_D2I; T_d2i[0:3, 3] = T_D2I
    T_c2i = T_d2i @ np.linalg.inv(T_d2c)

    # Apply roll offset (which is camera pitch, rotation around camera X)
    R_corr = Rotation.from_euler('xyz', [np.radians(roll_offset), 0.0, 0.0]).as_matrix()
    T_c2i[0:3, 0:3] = T_c2i[0:3, 0:3] @ R_corr
    T_i2c = np.linalg.inv(T_c2i)

    eskf = ESKF()
    latest_color = None
    latest_depth = None
    latest_color_t = 0
    latest_depth_t = 0
    frame_count = 0

    imu_history = []
    last_imu_t = None
    curr_acc = np.array([0.0, 0.0, -9.80665])
    curr_gyr = np.array([0.0, 0.0, 0.0])
    accel_received = False
    T_wc = np.eye(4)
    T_wc_prev = np.eye(4)

    poses = []

    while reader.has_next() and frame_count < 205:
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
                
                imu_history.sort(key=lambda x: x[0])
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
                    track_success, T_rel = tracker.align_frames(
                        latest_color_prev, depth_m_prev, latest_color, depth_m, T_init_rel
                    )
                    if track_success:
                        T_wc = T_wc_prev @ T_rel
                    else:
                        T_wc = T_pred
                        
                if track_success:
                    T_wi = T_wc @ T_i2c
                    gyro_norm = np.linalg.norm(curr_gyr)
                    rot_scale = 1.0 + 10.0 * gyro_norm
                    pos_scale = 1.0 + 10.0 * gyro_norm
                    eskf.update(T_wi[0:3, 3], T_wi[0:3, 0:3], pos_noise_scale=pos_scale, rot_noise_scale=rot_scale)
                    
                    T_wi_filt = np.eye(4)
                    T_wi_filt[0:3, 0:3] = eskf.R.as_matrix()
                    T_wi_filt[0:3, 3] = eskf.p
                    T_wc = T_wi_filt @ T_c2i
                    
                poses.append((frame_count, T_wc[0, 3], T_wc[1, 3], T_wc[2, 3]))
                
                latest_color_prev = latest_color.copy()
                depth_m_prev = depth_m.copy()
                T_wc_prev = T_wc.copy()
                frame_count += 1
                latest_color = None
                latest_depth = None

    return poses

if __name__ == "__main__":
    for roff in np.arange(-4.0, 3.0, 1.0):
        poses = run_trajectory(roff)
        poses = np.array(poses)
        print(f"Roll Offset (Camera Pitch): {roff:+.1f} deg", flush=True)
        diff200 = poses[200, 1:4] - poses[0, 1:4]
        print(f"  F0->F200 delta: X={diff200[0]:.4f}, Y={diff200[1]:.4f}, Z={diff200[2]:.4f}", flush=True)
