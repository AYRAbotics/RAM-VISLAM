import rosbag2_py, numpy as np, cv2, torch
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation
from ram_vi_slam.eskf import ESKF
from ram_vi_slam.tracking import RGBDTracker

tracker = RGBDTracker(640, 480, 'cpu')
K_c = np.array([[386.0, 0, 319.5], [0, 386.0, 239.5], [0, 0, 1]])
tracker.set_intrinsics(K_c)
R_D2I = np.eye(3)
t_D2I = np.array([-0.01174, -0.00552, 0.0051])
R_D2C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
t_D2C = np.array([0.015, 0.0, 0.0])
tracker.set_depth_extrinsics(K_c, R_D2C, t_D2C)
eskf = ESKF()

reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri='/home/rv/RAM_VI_SLAM/slam_benchmark_run1', storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
tps = {t.name: t.type for t in reader.get_all_topics_and_types()}

T_wc = np.eye(4)
T_wc_prev = np.eye(4)
imu_history = []
last_imu_t = None
curr_acc = np.array([0.0, 0.0, -9.8])
curr_gyr = np.array([0.0, 0.0, 0.0])
accel_received = False
latest_color = None
latest_depth = None

frame_count = 0

while reader.has_next():
    top, dat, t = reader.read_next()
    if top == '/camera/camera/accel/sample':
        m = deserialize_message(dat, get_message(tps[top]))
        imu_history.append((t, 'accel', np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])))
    elif top == '/camera/camera/gyro/sample':
        m = deserialize_message(dat, get_message(tps[top]))
        imu_history.append((t, 'gyro', np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])))
    elif top == '/camera/camera/color/image_raw':
        m = deserialize_message(dat, get_message(tps[top]))
        latest_color = np.frombuffer(m.data, dtype=np.uint8).reshape((m.height, m.width, 3))
    elif top == '/camera/camera/aligned_depth_to_color/image_raw':
        m = deserialize_message(dat, get_message(tps[top]))
        latest_depth = np.frombuffer(m.data, dtype=np.uint16).reshape((m.height, m.width))
        
        if latest_color is not None:
            imu_history.sort(key=lambda x: x[0])
            for imu_t, imu_type, imu_val in imu_history:
                if imu_t < t:
                    if imu_type == 'accel':
                        curr_acc = imu_val
                        accel_received = True
                    elif imu_type == 'gyro':
                        curr_gyr = imu_val
                    if last_imu_t is not None and accel_received:
                        dt = (imu_t - last_imu_t) * 1e-9
                        if dt > 0: eskf.predict(dt, curr_acc, curr_gyr)
                    last_imu_t = imu_t
            imu_history = [x for x in imu_history if x[0] >= t]
            
            depth_m = tracker.register_depth(latest_depth)
            
            if not eskf.is_gravity_initialized:
                continue
                
            T_pred = np.eye(4)
            T_pred[0:3, 0:3] = eskf.R.as_matrix()
            T_pred[0:3, 3] = eskf.p
            T_init_rel = np.linalg.inv(T_wc_prev) @ T_pred
            
            track_success, T_rel = tracker.align_frames(latest_color, depth_m, T_init_rel)
            
            T_delta = np.linalg.inv(T_init_rel) @ T_rel
            dt_norm = np.linalg.norm(T_delta[0:3, 3])
            
            if dt_norm > 0.05 or np.linalg.norm(curr_gyr) > 0.3:
                print(f"F{frame_count}: Gyro_mag={np.linalg.norm(curr_gyr):.2f}, track={track_success}, T_delta_trans={dt_norm:.4f}")
            
            if track_success:
                T_wc = T_wc_prev @ T_rel
            else:
                T_wc = T_pred
                
            eskf.update(T_wc[0:3, 3], Rotation.from_matrix(T_wc[0:3, 0:3]))
            T_wc_prev = T_wc
            frame_count += 1
            if frame_count > 600: break
