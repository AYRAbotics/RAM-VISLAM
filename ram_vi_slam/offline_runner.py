import argparse
import time
import os
import sys

# Pre-configure PyTorch allocator conf to reduce fragmentation and avoid GPU OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import signal
import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation

# ROS 2 bag parsing tools
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from .eskf import ESKF
from .imu_models import ComplementaryFilterEstimator, ConstantVelocityEstimator, GyroGuidedEstimator, IMUFusionEstimator
from .tracking import RGBDTracker
from .mapping import SurfelMap
from .loop_closure import LoopDetector
from .pgo import PoseGraphOptimizer
from .diagnostics import metrics_logger
import atexit

_global_state = {
    'surfel_map': None,
    'save_map_path': '/home/rv/RAM_VI_SLAM/output/surfel_map.ply',
    'bag_path': 'slam_benchmark_run1',
    'frame_count': 0,
    'saved': False
}

def cleanup_and_save_map():
    if _global_state.get('saved', False):
        return
    surfel_map = _global_state['surfel_map']
    if surfel_map is not None and surfel_map.active_n > 0:
        print(f"\n[Emergency Save] Exiting runner. Saving current map of {surfel_map.active_n} surfels to prevent data loss...", flush=True)
        try:
            surfel_map.prune_unstable(_global_state['frame_count'], min_weight=3.0)
            surfel_map.merge_voxels(voxel_size=0.01)
            
            save_path = _global_state['save_map_path']
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            surfel_map.export_ply(save_path)
            
            # Save unique timestamped map
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            bag_name = os.path.basename(os.path.normpath(_global_state.get('bag_path', 'slam_benchmark_run1')))
            unique_save_path = f'/home/rv/RAM_VI_SLAM/output/map_{bag_name}_{timestamp}.ply'
            surfel_map.export_ply(unique_save_path)
            
            _global_state['saved'] = True
            print(f"[Emergency Save] Maps saved successfully to {save_path} and {unique_save_path}", flush=True)
        except Exception as save_err:
            print(f"[Emergency Save] Failed to save map: {save_err}", flush=True)

atexit.register(cleanup_and_save_map)

# Calibration constants from bag inspection
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

# Keyframe creation thresholds
KF_TRANS_THRESH = 0.08   # meters (8 cm)
KF_ROT_THRESH   = 0.08   # radians (~4.5 deg)

def get_reader(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('cdr', 'cdr')
    )
    return reader

def image_to_numpy(msg):
    # Standard image msg deserialization to NumPy array
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_path', default='/home/rv/RAM_VI_SLAM/slam_benchmark_run1')
    parser.add_argument('--max_frames', type=int, default=999_999)
    parser.add_argument('--save_map', default='/home/rv/RAM_VI_SLAM/output/surfel_map.ply')
    parser.add_argument('--visualize', action='store_true', help='Show live visualizer (Pangolin replicate)')
    parser.add_argument('--imu_model', default='imufusion', choices=['eskf', 'complementary', 'const_vel', 'gyro_guided', 'imufusion'], help='IMU state estimation model')
    parser.add_argument('--roll-offset', type=float, default=-3.5, help='Roll offset in degrees (X axis)')
    parser.add_argument('--pitch_offset', type=float, default=0.0, help='Camera-IMU pitch offset in degrees')
    parser.add_argument('--yaw_offset', type=float, default=0.0, help='Camera-IMU yaw offset in degrees')
    parser.add_argument('--flat_ground', action=argparse.BooleanOptionalAction, default=True, help='Enforce flat ground constraint (constant height) to eliminate vertical drift/noise')
    parser.add_argument('--diagnostics', action=argparse.BooleanOptionalAction, default=False, help='Enable diagnostics and metrics logging')
    args = parser.parse_args()

    metrics_logger.configure(enabled=args.diagnostics)

    print(f"OfflineRunner: Starting processing on {args.bag_path} using IMU model: {args.imu_model}")
    
    visualizer = None
    if args.visualize:
        from .visualization import SLAMVisualizer
        gravity_aligned = args.imu_model in ['eskf', 'complementary', 'imufusion']
        visualizer = SLAMVisualizer(width=640, height=480, gravity_aligned=gravity_aligned)

    
    # 1. Setup SLAM components
    if args.imu_model == 'eskf':
        eskf = ESKF()
    elif args.imu_model == 'complementary':
        eskf = ComplementaryFilterEstimator()
    elif args.imu_model == 'const_vel':
        eskf = ConstantVelocityEstimator()
    elif args.imu_model == 'gyro_guided':
        eskf = GyroGuidedEstimator()
    elif args.imu_model == 'imufusion':
        eskf = IMUFusionEstimator()
    else:
        raise ValueError(f"Unknown imu_model: {args.imu_model}")
        
    tracker = RGBDTracker(FX_C, FY_C, CX_C, CY_C)
    
    # Extrinsics matrices
    T_d2c = np.eye(4)
    T_d2c[0:3, 0:3] = R_D2C
    T_d2c[0:3, 3] = T_D2C
    
    T_d2i = np.eye(4)
    T_d2i[0:3, 0:3] = R_D2I
    T_d2i[0:3, 3] = T_D2I
    
    # Camera-to-IMU extrinsic transform
    T_c2i = T_d2i @ np.linalg.inv(T_d2c)
    
    # Apply extrinsics calibration offsets
    roll_rad = np.radians(args.roll_offset)
    pitch_rad = np.radians(args.pitch_offset)
    yaw_rad = np.radians(args.yaw_offset)
    R_corr = Rotation.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad]).as_matrix()
    T_c2i[0:3, 0:3] = T_c2i[0:3, 0:3] @ R_corr
    
    T_i2c = np.linalg.inv(T_c2i)
    
    K_d_mat = np.array([
        [FX_D, 0.0,  CX_D],
        [0.0,  FY_D, CY_D],
        [0.0,  0.0,  1.0 ]
    ])
    tracker.set_calibration(K_d_mat, R_D2C, T_D2C)
    
    surfel_map = SurfelMap(FX_C, FY_C, CX_C, CY_C)
    _global_state['surfel_map'] = surfel_map
    _global_state['save_map_path'] = args.save_map
    _global_state['bag_path'] = args.bag_path
    loop_detector = LoopDetector(FX_C, FY_C, CX_C, CY_C)
    pgo = PoseGraphOptimizer()
    
    # 2. Iterate bag messages
    reader = get_reader(args.bag_path)
    topic_types = {tp.name: tp.type for tp in reader.get_all_topics_and_types()}
    
    # Message buffers
    latest_color = None
    latest_depth = None
    latest_color_t = 0
    latest_depth_t = 0
    
    imu_history = []  # tuples of (t, accel, gyro)
    last_imu_t = None
    curr_acc = np.array([0.0, 0.0, -9.80665])
    curr_gyr = np.array([0.0, 0.0, 0.0])
    accel_received = False
    
    frame_count = 0
    kf_count = 0
    last_frame_t = None
    last_kf_pose = None
    
    # Log optimization parameters
    T_wc = np.eye(4)  # Camera pose
    T_wc_prev = np.eye(4)
    
    z_fixed = 0.0
    z_fixed_initialized = False
    start_time = time.time()
    
    exit_requested = [False]
    def sigint_handler(signum, frame):
        if exit_requested[0]:
            print("\nOfflineRunner: Second interrupt. Exiting immediately.", flush=True)
            sys.exit(1)
        print("\nOfflineRunner: Interrupt received. Saving map and exiting after current frame...", flush=True)
        exit_requested[0] = True
    signal.signal(signal.SIGINT, sigint_handler)
    
    loop_idx = 0
    while reader.has_next() and frame_count < args.max_frames and not exit_requested[0]:
        topic, data, t_msg = reader.read_next()
        if visualizer is not None:
            loop_idx += 1
            if loop_idx % 20 == 0:
                visualizer.spin_once()
                
        msg_type = topic_types.get(topic)
        if msg_type is None:
            continue
        msg_class = get_message(msg_type)
        msg = deserialize_message(data, msg_class)
        
        # Buffer IMU readings
        if topic == '/camera/camera/accel/sample':
            acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            imu_history.append((t_msg, 'accel', acc))
        elif topic == '/camera/camera/gyro/sample':
            gyr = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
            imu_history.append((t_msg, 'gyro', gyr))
            
        # Buffer RGB & Depth
        elif topic == '/camera/camera/color/image_raw':
            latest_color = image_to_numpy(msg)
            latest_color_t = t_msg
        elif topic == '/camera/camera/depth/image_rect_raw':
            latest_depth = image_to_numpy(msg)
            latest_depth_t = t_msg
            
        # If we have synchronized color and depth frames
        if latest_color is not None and latest_depth is not None:
            time_diff = abs(latest_color_t - latest_depth_t) * 1e-6  # ms
            if time_diff < 30.0:  # sync within 30 ms
                frame_t = min(latest_color_t, latest_depth_t)
                metrics_logger.start_frame(frame_count, float(frame_t) * 1e-9)
                z_drift = 0.0
                frame_dt = 0.033
                if frame_count > 0 and last_frame_t is not None:
                    frame_dt = (frame_t - last_frame_t) * 1e-9
                last_frame_t = frame_t
                
                # A. Propagate ESKF to the frame timestamp using buffered IMU samples
                imu_history.sort(key=lambda x: x[0])
                
                was_initialized = eskf.is_gravity_initialized
                
                accel_samples = []
                gyro_samples = []
                t_start_imu = time.perf_counter()
                for imu_t, imu_type, imu_val in imu_history:
                    if imu_t < frame_t:
                        if imu_type == 'accel':
                            curr_acc = imu_val
                            accel_received = True
                            accel_samples.append(imu_val)
                        elif imu_type == 'gyro':
                            curr_gyr = imu_val
                            gyro_samples.append(imu_val)
                        
                        if last_imu_t is not None and accel_received:
                            dt = (imu_t - last_imu_t) * 1e-9
                            if dt > 0:
                                eskf.predict(dt, curr_acc, curr_gyr)
                        last_imu_t = imu_t
                t_imu_prop = time.perf_counter() - t_start_imu
                metrics_logger.log("imu_propagation_time", t_imu_prop)
                
                if len(accel_samples) > 0:
                    accel_vars = np.var(accel_samples, axis=0)
                    metrics_logger.log("accel_var_x", float(accel_vars[0]))
                    metrics_logger.log("accel_var_y", float(accel_vars[1]))
                    metrics_logger.log("accel_var_z", float(accel_vars[2]))
                if len(gyro_samples) > 0:
                    gyro_vars = np.var(gyro_samples, axis=0)
                    metrics_logger.log("gyro_var_x", float(gyro_vars[0]))
                    metrics_logger.log("gyro_var_y", float(gyro_vars[1]))
                    metrics_logger.log("gyro_var_z", float(gyro_vars[2]))
                
                # Clean up old IMU history
                imu_history = [x for x in imu_history if x[0] >= frame_t]
                
                # B. Use ESKF pose prediction to initialize tracking guess
                T_pred = np.eye(4)
                if eskf.is_gravity_initialized:
                    # T_wi is the pose of the IMU in world
                    T_wi = np.eye(4)
                    T_wi[0:3, 0:3] = eskf.R.as_matrix()
                    T_wi[0:3, 3] = eskf.p
                    # Convert IMU pose to Camera pose T_wc = T_wi @ T_c2i
                    T_pred = T_wi @ T_c2i
                
                # C. GPU Depth Registration
                depth_m = tracker.register_depth(latest_depth)
                metrics_logger.log("image_width", latest_color.shape[1])
                metrics_logger.log("image_height", latest_color.shape[0])
                metrics_logger.log("valid_depth_pct", float(np.mean(depth_m > 0.1) * 100.0) if depth_m is not None else 0.0)
                
                # D. Alignment / Tracking
                if frame_count == 0:
                    T_wc = T_pred
                    track_success = True
                    T_wi_init = T_wc @ T_i2c
                    z_fixed = T_wi_init[2, 3]
                    
                    flat_ground_active = args.flat_ground
                    if visualizer is not None:
                        flat_ground_active = visualizer.z_lock
                        
                    if flat_ground_active:
                        z_fixed_initialized = True
                        if hasattr(eskf, 'p'):
                            eskf.p[2] = z_fixed
                else:
                    # Compute predicted relative motion from previous frame to current frame
                    T_init_rel = np.linalg.inv(T_wc_prev) @ T_pred
                    # Damp the predicted translation guess by 50% to prevent visual tracking overshoot
                    T_init_rel[0:3, 3] = T_init_rel[0:3, 3] * 0.5
                    
                    # Track frame relative motion
                    track_success, T_rel = tracker.align_frames(latest_color_prev, depth_m_prev, latest_color, depth_m, T_init_rel)
                    
                    if track_success:
                        # Robust rejection: prevent translation-rotation ambiguity or catastrophic visual jumps
                        T_delta = np.linalg.inv(T_init_rel) @ T_rel
                        dt_norm = np.linalg.norm(T_delta[0:3, 3])
                        if dt_norm > 0.15:
                            track_success = False
                    
                    if track_success:
                        T_wc = T_wc_prev @ T_rel
                    else:
                        # Fallback on tracking failure: damp velocity and pull position back to prevent runaway drift
                        if hasattr(eskf, 'v'):
                            eskf.v = eskf.v * 0.2
                        if eskf.is_gravity_initialized:
                            T_wi_prev = T_wc_prev @ T_i2c
                            if hasattr(eskf, 'p'):
                                eskf.p = 0.8 * T_wi_prev[0:3, 3] + 0.2 * eskf.p
                            T_wi = np.eye(4)
                            T_wi[0:3, 0:3] = eskf.R.as_matrix()
                            T_wi[0:3, 3] = eskf.p
                            T_pred = T_wi @ T_c2i
                        T_wc = T_pred
                
                if track_success:
                    # E. Update ESKF nominal state with tracking pose
                    T_wi = T_wc @ T_i2c
                    
                    # Scale measurement noise dynamically based on angular velocity to trust IMU during fast rotation
                    gyro_norm = np.linalg.norm(curr_gyr)
                    rot_scale = 1.0 + 10.0 * gyro_norm
                    # Keep pos_scale constant to let visual tracking correct accelerometer velocity drift!
                    pos_scale = 1.0
                    
                    eskf.update(T_wi[0:3, 3], T_wi[0:3, 0:3], pos_noise_scale=pos_scale, rot_noise_scale=rot_scale, dt=frame_dt)
                    
                    # Feed back the updated ESKF filtered rotation to T_wc
                    if hasattr(eskf, 'R'):
                        T_wc[0:3, 0:3] = eskf.R.as_matrix() @ T_c2i[0:3, 0:3]
                    # Also update the ESKF internal position to match tracking to avoid accelerometer noise injection
                    if hasattr(eskf, 'p'):
                        T_wi = T_wc @ T_i2c
                        eskf.p = T_wi[0:3, 3]
                    
                    # Apply flat ground height constraint to camera pose and ESKF state
                    flat_ground_active = args.flat_ground
                    if visualizer is not None:
                        flat_ground_active = visualizer.z_lock
                    
                    T_wi_curr = T_wc @ T_i2c
                    if flat_ground_active:
                        if not z_fixed_initialized:
                            z_fixed = T_wi_curr[2, 3]
                            z_fixed_initialized = True
                        z_drift = T_wi_curr[2, 3] - z_fixed
                        T_wi_curr[2, 3] = z_fixed
                        T_wc = T_wi_curr @ T_c2i
                        if hasattr(eskf, 'p'):
                            eskf.p[2] = z_fixed
                        if hasattr(eskf, 'v'):
                            eskf.v[2] = 0.0
                    else:
                        z_fixed_initialized = False
                        z_drift = T_wi_curr[2, 3] - z_fixed
                    
                    # F. Manage Mapping & Keyframes
                    is_kf = False
                    if frame_count == 0:
                        is_kf = True
                    else:
                        # Decide if a new keyframe is needed
                        T_rel = np.linalg.inv(last_kf_pose) @ T_wc
                        t_dist = np.linalg.norm(T_rel[0:3, 3])
                        r_dist = np.linalg.norm(Rotation.from_matrix(T_rel[0:3, 0:3]).as_rotvec())
                        if t_dist > KF_TRANS_THRESH or r_dist > KF_ROT_THRESH:
                            is_kf = True
                            
                    metrics_logger.log("kf_inserted", is_kf)
                    if is_kf:
                        kf_id = kf_count
                        kf_count += 1
                        last_kf_pose = T_wc.copy()
                        metrics_logger.log("kf_id", kf_id)
                        
                        # Add keyframe to loop recognition database
                        loop_detector.add_keyframe(kf_id, latest_color, depth_m, T_wc)
                        pgo.add_keyframe(kf_id, T_wc)
                        
                        # G. Check Loop Closures
                        loop_res = loop_detector.detect_loop(kf_id, latest_color, depth_m, T_wc)
                        if loop_res is not None:
                            cand_kf_id, T_cand_curr = loop_res
                            pgo.add_loop_factor(cand_kf_id, kf_id, T_cand_curr)
                            
                            # Optimize graph and propagate deformation to surfels
                            t_start_pgo = time.perf_counter()
                            optimized_poses = pgo.optimize()
                            metrics_logger.log("pgo_time", time.perf_counter() - t_start_pgo)
                            
                            # Vectorized map deformation
                            with torch.no_grad():
                                for k_id, opt_pose in optimized_poses.items():
                                    orig_pose = loop_detector.db[k_id][2]
                                    # Delta transform
                                    T_diff = opt_pose @ np.linalg.inv(orig_pose)
                                    dR = torch.tensor(T_diff[0:3, 0:3], dtype=torch.float32, device=tracker.device)
                                    dt = torch.tensor(T_diff[0:3, 3], dtype=torch.float32, device=tracker.device)
                                    
                                    # Select surfels anchored to this keyframe
                                    s_mask = surfel_map.kf_ids[:surfel_map.active_n] == k_id
                                    if s_mask.any():
                                        surfel_map.positions[:surfel_map.active_n][s_mask] = (
                                            surfel_map.positions[:surfel_map.active_n][s_mask] @ dR.T
                                        ) + dt
                                        surfel_map.normals[:surfel_map.active_n][s_mask] = (
                                            surfel_map.normals[:surfel_map.active_n][s_mask] @ dR.T
                                        )
                                        
                                    # Update recorded pose
                                    loop_detector.db[k_id] = (
                                        loop_detector.db[k_id][0],
                                        loop_detector.db[k_id][1],
                                        opt_pose
                                    )
                                    
                            # Remove duplicates post-closure
                            surfel_map.merge_voxels()
                    
                    # H. Fuse current frame into the global map
                    t_start_mapping = time.perf_counter()
                    surfel_map.fuse_frame(latest_color, depth_m, T_wc, frame_count, kf_count - 1)
                    
                    # Periodic pruning of unstable surfels
                    if frame_count % 30 == 0:
                        surfel_map.prune_unstable(frame_count, min_weight=3.0)
                    t_mapping_time = time.perf_counter() - t_start_mapping
                    metrics_logger.log("mapping_time", t_mapping_time)
                    
                    # Live Visualisation
                    if visualizer is not None:
                        visualizer.update(T_wc, surfel_map, latest_color, depth_m, z_drift=z_drift)
                        if visualizer.save_requested:
                            print("\nOfflineRunner: Save & Exit requested from HUD buttons. Exiting loop...", flush=True)
                            break
                        
                # End frame metrics logging
                metrics_logger.end_frame()

                # Progress logging
                if frame_count % 50 == 0:
                    fps = (frame_count + 1) / (time.time() - start_time)
                    print(f"OfflineRunner: Processed {frame_count} frames, KF: {kf_count}, Surfels: {surfel_map.active_n}, Speed: {fps:.2f} FPS")
                
                # Periodically release PyTorch cached memory to prevent CUDA OOM
                if frame_count % 100 == 0:
                    torch.cuda.empty_cache()
                
                # Keep prev frames for visual tracking
                latest_color_prev = latest_color.copy()
                depth_m_prev = depth_m.copy()
                T_wc_prev = T_wc.copy()
                frame_count += 1
                _global_state['frame_count'] = frame_count
                
                # Periodically save a running checkpoint of the map to disk (non-intrusive)
                if frame_count > 0 and frame_count % 500 == 0:
                    print(f"\nOfflineRunner: Periodically saving running map checkpoint to {args.save_map}...", flush=True)
                    surfel_map.export_ply(args.save_map)
                
                # Clear frame buffers
                latest_color = None
                latest_depth = None
                
    # final downsampling and saving
    surfel_map.prune_unstable(frame_count, min_weight=3.0)
    surfel_map.merge_voxels(voxel_size=0.01)
    
    os.makedirs(os.path.dirname(args.save_map), exist_ok=True)
    surfel_map.export_ply(args.save_map)
    
    # Save unique timestamped map
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_name = os.path.basename(os.path.normpath(args.bag_path))
    unique_save_path = f'/home/rv/RAM_VI_SLAM/output/map_{bag_name}_{timestamp}.ply'
    surfel_map.export_ply(unique_save_path)
    
    _global_state['saved'] = True
    metrics_logger.save_and_close()
    print(f"OfflineRunner: Reconstructed {surfel_map.active_n} surfels in {time.time() - start_time:.2f} seconds. Maps saved to {args.save_map} and {unique_save_path}")
    
    if visualizer is not None:
        visualizer.destroy()

if __name__ == '__main__':
    main()