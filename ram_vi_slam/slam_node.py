import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import time
import message_filters
from queue import Queue
from threading import Thread, Lock

from sensor_msgs.msg import Image, Imu, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial.transform import Rotation
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from .eskf import ESKF
from .imu_models import ComplementaryFilterEstimator, ConstantVelocityEstimator, GyroGuidedEstimator, IMUFusionEstimator
from .tracking import RGBDTracker
from .mapping import SurfelMap
from .loop_closure import LoopDetector
from .pgo import PoseGraphOptimizer
from .diagnostics import metrics_logger

# Extrinsics matrices (same as bag calibration values)
R_D2C = np.array([
    [ 0.99999636, -0.00241311,  0.00117634],
    [ 0.00241741,  0.99999034, -0.00367048],
    [-0.00116747,  0.00367331,  0.99999255]
])
T_D2C = np.array([0.01454706, 0.00018594, 0.00039981])

R_D2I = np.eye(3)
T_D2I = np.array([-0.00552, 0.0051, 0.01174])

# Depth camera intrinsics from bag
FX_D = 388.2460022;  FY_D = 388.2460022
CX_D = 313.19522095; CY_D = 243.97851562

class SlamNode(Node):
    def __init__(self):
        super().__init__('slam_node')
        
        # 1. Thread locks and queues for asynchronous loop closure processing
        self.map_lock = Lock()
        self.loop_queue = Queue()
        
        # 2. Parameters and initialization
        self.declare_parameter('max_depth', 8.0)
        self.max_depth = self.get_parameter('max_depth').get_value()
        self.declare_parameter('visualize', True)
        self.visualize = self.get_parameter('visualize').get_value()
        self.visualizer = None
        
        self.declare_parameter('camera_imu_roll_offset', 0.0)
        self.declare_parameter('camera_imu_pitch_offset', 0.0)
        self.declare_parameter('camera_imu_yaw_offset', 0.0)
        self.roll_off = self.get_parameter('camera_imu_roll_offset').get_value()
        self.pitch_off = self.get_parameter('camera_imu_pitch_offset').get_value()
        self.yaw_off = self.get_parameter('camera_imu_yaw_offset').get_value()
        
        self.declare_parameter('imu_model', 'imufusion')
        self.imu_model = self.get_parameter('imu_model').get_value()
        self.declare_parameter('flat_ground', True)
        self.flat_ground = self.get_parameter('flat_ground').get_value()
        self.declare_parameter('enable_diagnostics', False)
        self.enable_diagnostics = self.get_parameter('enable_diagnostics').get_value()
        self.z_fixed = 0.0
        self.z_fixed_initialized = False
        
        metrics_logger.configure(enabled=self.enable_diagnostics)
        
        if self.imu_model == 'eskf':
            self.eskf = ESKF()
        elif self.imu_model == 'complementary':
            self.eskf = ComplementaryFilterEstimator()
        elif self.imu_model == 'const_vel':
            self.eskf = ConstantVelocityEstimator()
        elif self.imu_model == 'gyro_guided':
            self.eskf = GyroGuidedEstimator()
        elif self.imu_model == 'imufusion':
            self.eskf = IMUFusionEstimator()
        else:
            raise ValueError(f"Unknown imu_model: {self.imu_model}")
        self.tracker = None
        self.surfel_map = None
        self.loop_detector = None
        self.pgo = PoseGraphOptimizer()
        
        # Extrinsics setups
        self.T_d2c = np.eye(4)
        self.T_d2c[0:3, 0:3] = R_D2C
        self.T_d2c[0:3, 3] = T_D2C
        
        self.T_d2i = np.eye(4)
        self.T_d2i[0:3, 0:3] = R_D2I
        self.T_d2i[0:3, 3] = T_D2I
        
        self.T_c2i = self.T_d2i @ np.linalg.inv(self.T_d2c)
        
        # Apply camera-to-IMU roll, pitch, yaw offsets
        roll_rad = np.radians(self.roll_off)
        pitch_rad = np.radians(self.pitch_off)
        yaw_rad = np.radians(self.yaw_off)
        R_corr = Rotation.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad]).as_matrix()
        self.T_c2i[0:3, 0:3] = self.T_c2i[0:3, 0:3] @ R_corr
        
        self.T_i2c = np.linalg.inv(self.T_c2i)
        
        self.kf_count = 0
        self.last_kf_pose = None
        self.frame_count = 0
        self.last_frame_t = None
        
        # Tracking pose state
        self.T_wc = np.eye(4)
        self.T_wc_prev = np.eye(4)
        self.latest_color_prev = None
        self.depth_m_prev = None
        
        # IMU variables
        self.curr_acc = np.array([0.0, 0.0, -9.80665])
        self.curr_gyr = np.array([0.0, 0.0, 0.0])
        self.last_imu_t = None
        self.imu_history = []
        
        # Path for publishing
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'world'
        
        # 3. Subscribers and Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/slam/pose', 10)
        self.path_pub = self.create_publisher(Path, '/slam/path', 10)
        self.cloud_pub = self.create_publisher(PointCloud2, '/slam/map_points', 10)
        
        # Camera Info subscriber to set calibration matrices
        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.on_camera_info, 10
        )
        
        # IMU subscribers using Best Effort QoS with 100 depth queue
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=100
        )
        self.accel_sub = self.create_subscription(
            Imu, '/camera/camera/accel/sample', self.on_accel, imu_qos
        )
        self.gyro_sub = self.create_subscription(
            Imu, '/camera/camera/gyro/sample', self.on_gyro, imu_qos
        )
        
        # Approximate time synchronizer for RGB and Depth topics
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/depth/image_rect_raw')
        
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.03
        )
        self.sync.registerCallback(self.on_frame_pair)
        
        # Background worker thread for Place Recognition and Loop Closures
        self.loop_thread = Thread(target=self.loop_worker, daemon=True)
        self.loop_thread.start()
        
        self.get_logger().info("SlamNode initialized. Waiting for CameraInfo...")

    def on_camera_info(self, msg):
        """Set intrinsics and instantiate tracker/mapper once CameraInfo is received."""
        if self.tracker is not None:
            return
            
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]
        
        self.get_logger().info(f"SlamNode: Received CameraInfo. Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        
        self.tracker = RGBDTracker(fx, fy, cx, cy, width=msg.width, height=msg.height)
        
        # Create corresponding depth camera calibration using the correct depth camera intrinsics
        K_d = np.array([
            [FX_D, 0.0,  CX_D],
            [0.0,  FY_D, CY_D],
            [0.0,  0.0,  1.0 ]
        ])
        self.tracker.set_calibration(K_d, R_D2C, T_D2C)
        
        self.surfel_map = SurfelMap(fx, fy, cx, cy, width=msg.width, height=msg.height)
        self.loop_detector = LoopDetector(fx, fy, cx, cy, width=msg.width, height=msg.height)
        
        if self.visualize and self.visualizer is None:
            from .visualization import SLAMVisualizer
            gravity_aligned = self.imu_model in ['eskf', 'complementary']
            self.visualizer = SLAMVisualizer(width=msg.width, height=msg.height, gravity_aligned=gravity_aligned)


    def on_accel(self, msg):
        acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        t_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.imu_history.append((t_ns, 'accel', acc))

    def on_gyro(self, msg):
        gyr = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        t_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.imu_history.append((t_ns, 'gyro', gyr))

    def on_frame_pair(self, color_msg, depth_msg):
        """Fuses synchronized color and depth frames."""
        if self.tracker is None:
            return
            
        # Parse image stamps
        color_t = color_msg.header.stamp.sec * 1_000_000_000 + color_msg.header.stamp.nanosec
        depth_t = depth_msg.header.stamp.sec * 1_000_000_000 + depth_msg.header.stamp.nanosec
        frame_t = min(color_t, depth_t)
        metrics_logger.start_frame(self.frame_count, float(frame_t) * 1e-9)
        z_drift = 0.0
        
        frame_dt = 0.033
        if self.frame_count > 0 and self.last_frame_t is not None:
            frame_dt = (frame_t - self.last_frame_t) * 1e-9
        self.last_frame_t = frame_t
        
        # Deserialise images to NumPy
        color_np = np.frombuffer(color_msg.data, dtype=np.uint8).reshape((color_msg.height, color_msg.width, 3))
        # Realsense usually uses BGR8, convert to RGB
        color_rgb = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
        
        depth_np = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape((depth_msg.height, depth_msg.width))

        # 1. Propagate ESKF to the current frame timestamp
        self.imu_history.sort(key=lambda x: x[0])
        
        accel_samples = []
        gyro_samples = []
        t_start_imu = time.perf_counter()
        for imu_t, imu_type, imu_val in self.imu_history:
            if imu_t < frame_t:
                if imu_type == 'accel':
                    self.curr_acc = imu_val
                    accel_samples.append(imu_val)
                elif imu_type == 'gyro':
                    self.curr_gyr = imu_val
                    gyro_samples.append(imu_val)
                
                if self.last_imu_t is not None:
                    dt = (imu_t - self.last_imu_t) * 1e-9
                    if dt > 0:
                        self.eskf.predict(dt, self.curr_acc, self.curr_gyr)
                self.last_imu_t = imu_t
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
                
        self.imu_history = [x for x in self.imu_history if x[0] >= frame_t]

        # 2. Get initial tracking guess from ESKF
        T_pred = np.eye(4)
        if self.eskf.is_gravity_initialized:
            T_wi = np.eye(4)
            T_wi[0:3, 0:3] = self.eskf.R.as_matrix()
            T_wi[0:3, 3] = self.eskf.p
            # Convert IMU pose to Camera pose T_wc = T_wi @ T_c2i
            T_pred = T_wi @ self.T_c2i

        # 3. GPU Depth Registration
        depth_m = self.tracker.register_depth(depth_np)
        metrics_logger.log("image_width", color_rgb.shape[1])
        metrics_logger.log("image_height", color_rgb.shape[0])
        metrics_logger.log("valid_depth_pct", float(np.mean(depth_m > 0.1) * 100.0) if depth_m is not None else 0.0)

        # 4. Tracker Alignment
        if self.frame_count == 0:
            self.T_wc = T_pred
            success = True
            T_wi_init = self.T_wc @ self.T_i2c
            self.z_fixed = T_wi_init[2, 3]
            
            flat_ground_active = self.flat_ground
            if self.visualizer is not None:
                flat_ground_active = self.visualizer.z_lock
                
            if flat_ground_active:
                self.z_fixed_initialized = True
                if hasattr(self.eskf, 'p'):
                    self.eskf.p[2] = self.z_fixed
        else:
            # Compute predicted relative motion
            T_init_rel = np.linalg.inv(self.T_wc_prev) @ T_pred
            # Damp the predicted translation guess by 50% to prevent visual tracking overshoot
            T_init_rel[0:3, 3] = T_init_rel[0:3, 3] * 0.5
            
            success, T_rel = self.tracker.align_frames(
                self.latest_color_prev, self.depth_m_prev, color_rgb, depth_m, T_init_rel
            )
            
            if success:
                # Robust rejection: prevent translation-rotation ambiguity or catastrophic visual jumps
                T_delta = np.linalg.inv(T_init_rel) @ T_rel
                dt_norm = np.linalg.norm(T_delta[0:3, 3])
                if dt_norm > 0.15:
                    success = False
            
            if success:
                self.T_wc = self.T_wc_prev @ T_rel
            else:
                # Fallback on tracking failure: damp velocity and pull position back to prevent runaway drift
                if hasattr(self.eskf, 'v'):
                    self.eskf.v = self.eskf.v * 0.2
                if self.eskf.is_gravity_initialized:
                    T_wi_prev = self.T_wc_prev @ self.T_i2c
                    if hasattr(self.eskf, 'p'):
                        self.eskf.p = 0.8 * T_wi_prev[0:3, 3] + 0.2 * self.eskf.p
                    T_wi = np.eye(4)
                    T_wi[0:3, 0:3] = self.eskf.R.as_matrix()
                    T_wi[0:3, 3] = self.eskf.p
                    T_pred = T_wi @ self.T_c2i
                self.T_wc = T_pred

        if success:
            # Update ESKF nominal pose
            T_wi = self.T_wc @ self.T_i2c
            
            # Scale measurement noise dynamically based on angular velocity to trust IMU during fast rotation
            gyro_norm = np.linalg.norm(self.curr_gyr)
            rot_scale = 1.0 + 10.0 * gyro_norm
            # Keep pos_scale constant to let visual tracking correct accelerometer velocity drift!
            pos_scale = 1.0
            
            self.eskf.update(T_wi[0:3, 3], T_wi[0:3, 0:3], pos_noise_scale=pos_scale, rot_noise_scale=rot_scale, dt=frame_dt)
            
            # Feed back the updated ESKF filtered rotation to self.T_wc
            if hasattr(self.eskf, 'R'):
                self.T_wc[0:3, 0:3] = self.eskf.R.as_matrix() @ self.T_c2i[0:3, 0:3]
            # Also update the ESKF internal position to match tracking to avoid accelerometer noise injection
            if hasattr(self.eskf, 'p'):
                T_wi = self.T_wc @ self.T_i2c
                self.eskf.p = T_wi[0:3, 3]
            
            # Apply flat ground height constraint to camera pose and ESKF state
            flat_ground_active = self.flat_ground
            if self.visualizer is not None:
                flat_ground_active = self.visualizer.z_lock
            
            T_wi_curr = self.T_wc @ self.T_i2c
            if flat_ground_active:
                if not self.z_fixed_initialized:
                    self.z_fixed = T_wi_curr[2, 3]
                    self.z_fixed_initialized = True
                z_drift = T_wi_curr[2, 3] - self.z_fixed
                T_wi_curr[2, 3] = self.z_fixed
                self.T_wc = T_wi_curr @ self.T_c2i
                if hasattr(self.eskf, 'p'):
                    self.eskf.p[2] = self.z_fixed
                if hasattr(self.eskf, 'v'):
                    self.eskf.v[2] = 0.0
            else:
                self.z_fixed_initialized = False
                z_drift = T_wi_curr[2, 3] - self.z_fixed
            
            # Keyframe management
            is_kf = False
            if self.frame_count == 0:
                is_kf = True
            else:
                T_rel = np.linalg.inv(self.last_kf_pose) @ self.T_wc
                t_dist = np.linalg.norm(T_rel[0:3, 3])
                r_dist = np.linalg.norm(Rotation.from_matrix(T_rel[0:3, 0:3]).as_rotvec())
                if t_dist > 0.08 or r_dist > 0.08:
                    is_kf = True
                    
            metrics_logger.log("kf_inserted", is_kf)
            if is_kf:
                kf_id = self.kf_count
                self.kf_count += 1
                self.last_kf_pose = self.T_wc.copy()
                metrics_logger.log("kf_id", kf_id)
                
                # Push keyframe to background worker thread queue for place recognition
                self.loop_queue.put((kf_id, color_rgb.copy(), depth_m.copy(), self.T_wc.copy()))

            # 5. Fuse frame into global surfel map
            t_start_mapping = time.perf_counter()
            with self.map_lock:
                self.surfel_map.fuse_frame(color_rgb, depth_m, self.T_wc, self.frame_count, self.kf_count - 1)
                
                if self.frame_count % 30 == 0:
                    self.surfel_map.prune_unstable(self.frame_count, min_weight=3.0)
                    torch.cuda.empty_cache()
            t_mapping_time = time.perf_counter() - t_start_mapping
            metrics_logger.log("mapping_time", t_mapping_time)
                    
            if self.visualizer is not None:
                self.visualizer.update(self.T_wc, self.surfel_map, color_rgb, depth_m, z_drift=z_drift)

            # Publish pose and downsampled cloud
            self.publish_data(color_msg.header.stamp)

        metrics_logger.end_frame()
        self.latest_color_prev = color_rgb.copy()
        self.depth_m_prev = depth_m.copy()
        self.T_wc_prev = self.T_wc.copy()
        self.frame_count += 1

    def loop_worker(self):
        """Asynchronous background loop detection thread."""
        while rclpy.ok():
            try:
                # Blocks until a keyframe is pushed
                kf_id, color, depth, T_wc = self.loop_queue.get()
                
                # Add keyframe to index
                self.loop_detector.add_keyframe(kf_id, color, depth, T_wc)
                self.pgo.add_keyframe(kf_id, T_wc)
                
                # Check for loop closures
                loop_res = self.loop_detector.detect_loop(kf_id, color, depth, T_wc)
                if loop_res is not None:
                    cand_kf_id, T_cand_curr = loop_res
                    self.pgo.add_loop_factor(cand_kf_id, kf_id, T_cand_curr)
                    
                    # Run pose graph optimization
                    t_start_pgo = time.perf_counter()
                    optimized_poses = self.pgo.optimize()
                    metrics_logger.log("pgo_time", time.perf_counter() - t_start_pgo)
                    
                    # Deform global map safely using lock
                    with self.map_lock:
                        for k_id, opt_pose in optimized_poses.items():
                            orig_pose = self.loop_detector.db[k_id][2]
                            T_diff = opt_pose @ np.linalg.inv(orig_pose)
                            
                            dR = torch.tensor(T_diff[0:3, 0:3], dtype=torch.float32, device=DEVICE)
                            dt = torch.tensor(T_diff[0:3, 3], dtype=torch.float32, device=DEVICE)
                            
                            s_mask = self.surfel_map.kf_ids[:self.surfel_map.active_n] == k_id
                            if s_mask.any():
                                self.surfel_map.positions[:self.surfel_map.active_n][s_mask] = (
                                    self.surfel_map.positions[:self.surfel_map.active_n][s_mask] @ dR.T
                                ) + dt
                                self.surfel_map.normals[:self.surfel_map.active_n][s_mask] = (
                                    self.surfel_map.normals[:self.surfel_map.active_n][s_mask] @ dR.T
                                )
                                
                            # Update local database reference pose
                            self.loop_detector.db[k_id] = (
                                self.loop_detector.db[k_id][0],
                                self.loop_detector.db[k_id][1],
                                opt_pose
                            )
                        
                        # Voxel grid filter cleanup
                        self.surfel_map.merge_voxels()
                        
                self.loop_queue.task_done()
            except Exception as e:
                self.get_logger().error(f"LoopWorker error: {e}")
                time.sleep(0.5)

    def publish_data(self, stamp):
        """Publishes camera pose, nav path, and downsampled surfel map points."""
        # 1. Pose
        p = self.T_wc[0:3, 3]
        q = Rotation.from_matrix(self.T_wc[0:3, 0:3]).as_quat()
        
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = float(p[0])
        pose_msg.pose.position.y = float(p[1])
        pose_msg.pose.position.z = float(p[2])
        pose_msg.pose.orientation.x = float(q[0])
        pose_msg.pose.orientation.y = float(q[1])
        pose_msg.pose.orientation.z = float(q[2])
        pose_msg.pose.orientation.w = float(q[3])
        self.pose_pub.publish(pose_msg)
        
        # 2. Path
        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(pose_msg)
        self.path_pub.publish(self.path_msg)
        
        # 3. Point Cloud Map
        if self.surfel_map.active_n > 0:
            # Downsample cloud for online publishing
            indices = np.random.choice(
                self.surfel_map.active_n, 
                size=min(50000, self.surfel_map.active_n), 
                replace=False
            )
            with self.map_lock:
                pos = self.surfel_map.positions[:self.surfel_map.active_n][indices].cpu().numpy()
                col = (self.surfel_map.colors[:self.surfel_map.active_n][indices].cpu().numpy() * 255.0).astype(np.uint8)
            
            # Form fields
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
            ]
            
            points_data = []
            for i in range(len(pos)):
                r, g, b = col[i, 0], col[i, 1], col[i, 2]
                rgb = (int(r) << 16) | (int(g) << 8) | int(b)
                points_data.append([pos[i, 0], pos[i, 1], pos[i, 2], rgb])
                
            cloud_msg = pc2.create_cloud(
                Header(frame_id='world', stamp=stamp), fields, points_data
            )
            self.cloud_pub.publish(cloud_msg)

    def destroy_node(self):
        metrics_logger.save_and_close()
        if self.visualizer is not None:
            self.visualizer.destroy()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("SlamNode: Keyboard interrupt.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()