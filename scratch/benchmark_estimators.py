import os
import time
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import cv2
from scipy.spatial.transform import Rotation

from ram_vi_slam.tracking import RGBDTracker
from ram_vi_slam.eskf import ESKF
from ram_vi_slam.imu_models import ComplementaryFilterEstimator, ConstantVelocityEstimator, GyroGuidedEstimator, IMUFusionEstimator

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

T_d2c = np.eye(4); T_d2c[0:3, 0:3] = R_D2C; T_d2c[0:3, 3] = T_D2C
T_d2i = np.eye(4); T_d2i[0:3, 0:3] = R_D2I; T_d2i[0:3, 3] = T_D2I
T_c2i = T_d2i @ np.linalg.inv(T_d2c)

# Apply default offline_runner roll offset of -3.5 degrees
roll_rad = np.radians(-3.5)
R_corr = Rotation.from_euler('xyz', [roll_rad, 0.0, 0.0]).as_matrix()
T_c2i[0:3, 0:3] = T_c2i[0:3, 0:3] @ R_corr
T_i2c = np.linalg.inv(T_c2i)

def get_reader(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('cdr', 'cdr')
    )
    return reader

def run_benchmark(estimator_name, max_frames=500):
    bag_path = "/home/rv/RAM_VI_SLAM/slam_benchmark_run1"
    reader = get_reader(bag_path)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    
    tracker = RGBDTracker(FX_C, FY_C, CX_C, CY_C)
    K_d_mat = np.array([[FX_D, 0.0,  CX_D], [0.0,  FY_D, CY_D], [0.0,  0.0,  1.0]])
    tracker.set_calibration(K_d_mat, R_D2C, T_D2C)
    
    # Instantiate estimator
    if estimator_name == 'eskf':
        estimator = ESKF()
    elif estimator_name == 'complementary':
        estimator = ComplementaryFilterEstimator()
    elif estimator_name == 'const_vel':
        estimator = ConstantVelocityEstimator()
    elif estimator_name == 'gyro_guided':
        estimator = GyroGuidedEstimator()
    elif estimator_name == 'imufusion':
        estimator = IMUFusionEstimator()
    else:
        raise ValueError(f"Unknown estimator: {estimator_name}")
        
    latest_color = None
    latest_depth = None
    latest_color_t = 0
    latest_depth_t = 0
    
    imu_history = []
    last_imu_t = None
    curr_acc = np.array([0.0, 0.0, -9.80665])
    curr_gyr = np.array([0.0, 0.0, 0.0])
    accel_received = False
    
    frame_count = 0
    success_count = 0
    last_frame_t = None
    
    T_wc = np.eye(4)
    T_wc_prev = np.eye(4)
    
    # Store trajectory poses and frame details
    poses = []
    
    start_time = time.time()
    
    while reader.has_next() and frame_count < max_frames:
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
                
                # Propagate estimator using buffered IMU readings
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
                                estimator.predict(dt, curr_acc, curr_gyr)
                        last_imu_t = imu_t
                imu_history = [x for x in imu_history if x[0] >= frame_t]
                
                # Tracking initialization guess from prediction
                T_pred = np.eye(4)
                if estimator.is_gravity_initialized:
                    T_wi = np.eye(4)
                    if hasattr(estimator, 'R'):
                        T_wi[0:3, 0:3] = estimator.R.as_matrix()
                    if hasattr(estimator, 'p'):
                        T_wi[0:3, 3] = estimator.p
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
                        T_delta = np.linalg.inv(T_init_rel) @ T_rel
                        dt_norm = np.linalg.norm(T_delta[0:3, 3])
                        if dt_norm > 0.15:
                            track_success = False
                            
                    if track_success:
                        T_wc = T_wc_prev @ T_rel
                    else:
                        # Fallback
                        if hasattr(estimator, 'v'):
                            estimator.v = estimator.v * 0.2
                        if estimator.is_gravity_initialized:
                            T_wi_prev = T_wc_prev @ T_i2c
                            if hasattr(estimator, 'p'):
                                estimator.p = 0.8 * T_wi_prev[0:3, 3] + 0.2 * estimator.p
                            T_wi = np.eye(4)
                            if hasattr(estimator, 'R'):
                                T_wi[0:3, 0:3] = estimator.R.as_matrix()
                            if hasattr(estimator, 'p'):
                                T_wi[0:3, 3] = estimator.p
                            T_pred = T_wi @ T_c2i
                        T_wc = T_pred
                        
                if track_success:
                    success_count += 1
                    T_wi = T_wc @ T_i2c
                    
                    if estimator_name == 'gyro_guided':
                        estimator.update(T_wi[0:3, 3], T_wi[0:3, 0:3], dt=frame_dt)
                    elif estimator_name == 'imufusion':
                        estimator.update(T_wi[0:3, 3], T_wi[0:3, 0:3], dt=frame_dt)
                    elif estimator_name == 'const_vel':
                        estimator.update(T_wi[0:3, 3], T_wi[0:3, 0:3])
                    elif estimator_name == 'eskf':
                        gyro_norm = np.linalg.norm(curr_gyr)
                        rot_scale = 1.0 + 10.0 * gyro_norm
                        estimator.update(T_wi[0:3, 3], T_wi[0:3, 0:3], pos_noise_scale=1.0, rot_noise_scale=rot_scale)
                    else:
                        # Complementary Filter
                        gyro_norm = np.linalg.norm(curr_gyr)
                        rot_scale = 1.0 + 10.0 * gyro_norm
                        estimator.update(T_wi[0:3, 3], T_wi[0:3, 0:3], pos_noise_scale=1.0, rot_noise_scale=rot_scale, dt=frame_dt)
                        
                    T_wi_filt = np.eye(4)
                    if hasattr(estimator, 'R'):
                        T_wi_filt[0:3, 0:3] = estimator.R.as_matrix()
                    if hasattr(estimator, 'p'):
                        T_wi_filt[0:3, 3] = estimator.p
                    T_wc = T_wi_filt @ T_c2i
                    
                poses.append((frame_count, T_wc.copy(), track_success, np.linalg.norm(curr_gyr)))
                
                latest_color_prev = latest_color.copy()
                depth_m_prev = depth_m.copy()
                T_wc_prev = T_wc.copy()
                frame_count += 1
                latest_color = None
                latest_depth = None
                
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    success_rate = (success_count / frame_count) * 100.0 if frame_count > 0 else 0
    
    # Calculate position statistics during Turn 1 (frames 130-160)
    turn1_poses = [p[1][0:3, 3] for p in poses if 130 <= p[0] <= 160]
    if len(turn1_poses) > 0:
        turn1_poses = np.array(turn1_poses)
        turn1_std = np.std(turn1_poses, axis=0)
        turn1_drift_mag = np.linalg.norm(turn1_poses[-1] - turn1_poses[0])
    else:
        turn1_std = np.zeros(3)
        turn1_drift_mag = 0.0
        
    final_pos = T_wc[0:3, 3]
    
    print(f"Finished {estimator_name}: {frame_count} frames in {elapsed:.2f}s ({fps:.1f} FPS). Success Rate: {success_rate:.1f}%")
    return {
        'name': estimator_name,
        'fps': fps,
        'success_rate': success_rate,
        'fallbacks': frame_count - success_count,
        'final_pos': final_pos,
        'turn1_std_xyz': turn1_std,
        'turn1_drift': turn1_drift_mag
    }

if __name__ == '__main__':
    estimators = ['eskf', 'complementary', 'const_vel', 'gyro_guided', 'imufusion']
    results = []
    print("====================================================")
    print("STARTING ESTIMATOR TRAJECTORY & TRACKING BENCHMARK")
    print("====================================================")
    for est in estimators:
        try:
            res = run_benchmark(est, max_frames=180)
            results.append(res)
        except Exception as e:
            print(f"Failed to benchmark {est}: {e}")
            
    print("\n====================================================")
    print("BENCHMARK SUMMARY (First 180 Frames)")
    print("====================================================")
    print(f"{'Estimator':<18} | {'FPS':<6} | {'Success%':<8} | {'Fallbacks':<9} | {'Turn 1 Drift':<12} | {'Turn 1 Std (X,Y,Z)':<20}")
    print("-" * 85)
    for r in results:
        std_str = f"({r['turn1_std_xyz'][0]:.3f}, {r['turn1_std_xyz'][1]:.3f}, {r['turn1_std_xyz'][2]:.3f})"
        print(f"{r['name']:<18} | {r['fps']:<6.1f} | {r['success_rate']:<7.1f}% | {r['fallbacks']:<9d} | {r['turn1_drift']:<12.3f} | {std_str:<20}")
    print("====================================================")
