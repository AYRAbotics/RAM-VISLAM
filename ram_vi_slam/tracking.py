import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RGBDTracker:
    def __init__(self, fx_c, fy_c, cx_c, cy_c, width=640, height=480):
        self.width = width
        self.height = height
        self.fx_c = fx_c
        self.fy_c = fy_c
        self.cx_c = cx_c
        self.cy_c = cy_c
        
        self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx_c, fy_c, cx_c, cy_c
        )
        
        # Extrinsics and intrinsics for depth camera
        self.K_d = None
        self.R_d2c_t = None
        self.t_d2c_t = None
        
        # Device configurations
        self.device = DEVICE

    def set_calibration(self, K_d, R_d2c, t_d2c):
        """Set the depth camera intrinsics and the depth-to-color transform."""
        self.K_d = K_d
        self.R_d2c_t = torch.tensor(R_d2c, dtype=torch.float32, device=self.device)
        self.t_d2c_t = torch.tensor(t_d2c, dtype=torch.float32, device=self.device)

    def register_depth(self, depth_img_np):
        """
        Align the depth image from the depth camera to the color camera coordinate frame.
        Takes numpy depth image (uint16 in mm or float32 in m).
        Returns a numpy depth image (float32 in meters) aligned with the color image.
        """
        if self.K_d is None:
            raise ValueError("Tracker calibration not set!")

        # Convert to tensor and work on GPU
        depth_t = torch.tensor(depth_img_np.astype(np.float32), device=self.device)
        if depth_img_np.dtype == np.uint16:
            depth_t = depth_t / 1000.0  # mm to meters

        H, W = self.height, self.width
        
        # Grid of pixel coordinates in depth frame
        u_grid = torch.arange(W, dtype=torch.float32, device=self.device)
        v_grid = torch.arange(H, dtype=torch.float32, device=self.device)
        v_coords, u_coords = torch.meshgrid(v_grid, u_grid, indexing='ij')

        # Depth intrinsics
        fx_d = self.K_d[0, 0]
        fy_d = self.K_d[1, 1]
        cx_d = self.K_d[0, 2]
        cy_d = self.K_d[1, 2]

        # 1. 3D point cloud in depth frame
        valid_depth = (depth_t > 0.1) & (depth_t < 8.0)
        z_d = depth_t[valid_depth]
        u_d = u_coords[valid_depth]
        v_d = v_coords[valid_depth]

        x_d = (u_d - cx_d) * z_d / fx_d
        y_d = (v_d - cy_d) * z_d / fy_d

        pts_d = torch.stack([x_d, y_d, z_d], dim=-1)  # (N, 3)

        # 2. Transform points to color camera frame: Pc = R_d2c * Pd + t_d2c
        pts_c = pts_d @ self.R_d2c_t.T + self.t_d2c_t

        # 3. Project to color camera pixels
        x_c, y_c, z_c = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
        
        # Only keep points in front of the camera
        valid_zc = z_c > 0.1
        x_c = x_c[valid_zc]
        y_c = y_c[valid_zc]
        z_c = z_c[valid_zc]

        u_c = (x_c * self.fx_c) / z_c + self.cx_c
        v_c = (y_c * self.fy_c) / z_c + self.cy_c

        # 4. Filter indices within boundaries
        u_idx = torch.round(u_c).long()
        v_idx = torch.round(v_c).long()
        valid_idx = (u_idx >= 0) & (u_idx < W) & (v_idx >= 0) & (v_idx < H)

        u_idx = u_idx[valid_idx]
        v_idx = v_idx[valid_idx]
        z_c = z_c[valid_idx]

        # Resolve occlusion by sorting from furthest to closest.
        # When writing to indices, the closest points will overwrite the further ones.
        sort_idx = torch.argsort(z_c, descending=True)
        u_idx = u_idx[sort_idx]
        v_idx = v_idx[sort_idx]
        z_c = z_c[sort_idx]

        # Flatten and write
        flat_idx = v_idx * W + u_idx
        reg_depth_flat = torch.zeros(H * W, dtype=torch.float32, device=self.device)
        reg_depth_flat[flat_idx] = z_c
        
        return reg_depth_flat.view(H, W).cpu().numpy()

    def align_frames(self, src_color_np, src_depth_np, tgt_color_np, tgt_depth_np, T_init=np.eye(4)):
        """
        Compute frame-to-frame tracking transformation from src (source) to tgt (target).
        Colors: (H,W,3) uint8 numpy, Depth: (H,W) float32 in meters.
        Returns: success (bool), 4x4 matrix (np.ndarray)
        """
        # Create Open3D images
        src_color_o3d = o3d.geometry.Image(src_color_np)
        src_depth_o3d = o3d.geometry.Image(src_depth_np)
        tgt_color_o3d = o3d.geometry.Image(tgt_color_np)
        tgt_depth_o3d = o3d.geometry.Image(tgt_depth_np)

        src_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            src_color_o3d, src_depth_o3d, convert_rgb_to_intensity=True,
            depth_scale=1.0, depth_trunc=8.0
        )
        tgt_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            tgt_color_o3d, tgt_depth_o3d, convert_rgb_to_intensity=True,
            depth_scale=1.0, depth_trunc=8.0
        )

        # Convert target -> source camera motion (T_init) to source -> target point transform
        T_init_point = np.linalg.inv(T_init)

        # 1. Open3D Hybrid RGB-D Odometry
        option = o3d.pipelines.odometry.OdometryOption()
        option.depth_diff_max = 0.07
        option.depth_min = 0.1
        option.depth_max = 8.0

        try:
            success, T_odom, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                src_rgbd, tgt_rgbd, self.intrinsic_o3d, T_init_point,
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                option
            )
        except Exception:
            success = False

        if not success:
            T_odom = T_init_point

        # 2. Point-to-Plane ICP refinement
        try:
            src_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(src_rgbd, self.intrinsic_o3d)
            tgt_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(tgt_rgbd, self.intrinsic_o3d)
            
            # Target needs normals for point-to-plane ICP
            tgt_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            
            # registration_icp maps src -> tgt, so initial guess must be T_odom (src -> tgt)
            icp_result = o3d.pipelines.registration.registration_icp(
                src_pcd, tgt_pcd, 0.05, T_odom,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            
            # Verify tracking fitness
            if icp_result.fitness > 0.30:
                T_rel = np.linalg.inv(icp_result.transformation)
                return True, T_rel
        except Exception:
            pass

        # If ICP fails, return success and target -> source camera motion
        T_rel = np.linalg.inv(T_odom)
        return success, T_rel