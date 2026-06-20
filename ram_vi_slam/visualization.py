import numpy as np
import open3d as o3d
import cv2
import torch

class SLAMVisualizer:
    def __init__(self, width=640, height=480, gravity_aligned=False):
        self.width = width
        self.height = height
        
        # 1. 3D Open3D Visualizer Setup
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="RAM-SLAM 3D Map (ElasticFusion Style)", width=1024, height=768)
        
        if gravity_aligned:
            # Coordinate system alignment: World uses ROS standard (+Z UP, -Y FORWARD)
            # Open3D uses (+Y UP, -Z FORWARD). Rotate by -90 degrees around X-axis.
            self.R_w2v = np.array([
                [1.0,  0.0, 0.0],
                [0.0,  0.0, 1.0],
                [0.0, -1.0, 0.0]
            ])
        else:
            # World is aligned with initial camera/IMU frame (+Z FORWARD, +Y DOWN)
            # Map X -> X (right), -Y -> Y (up), Z -> -Z (forward)
            self.R_w2v = np.array([
                [1.0,  0.0,  0.0],
                [0.0, -1.0,  0.0],
                [0.0,  0.0, -1.0]
            ])
            
        self.T_w2v = np.eye(4)
        self.T_w2v[0:3, 0:3] = self.R_w2v


        # Point cloud representing the surfel map
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        
        # Active camera frustum
        self.frustum = o3d.geometry.LineSet()
        self.vis.add_geometry(self.frustum)
        
        # Trajectory path lines
        self.trajectory_pts = []
        self.trajectory = o3d.geometry.LineSet()
        self.vis.add_geometry(self.trajectory)
        
        # Add basic coordinate frame (rotated to match aligned orientation)
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        self.coord_frame.rotate(self.R_w2v, center=(0, 0, 0))
        self.vis.add_geometry(self.coord_frame)
        
        # Define basic template frustum vertices in camera coordinate frame (looking down +Z)
        self.base_frustum_vertices = np.array([
            [0, 0, 0],
            [-0.1, -0.075, 0.15],
            [0.1, -0.075, 0.15],
            [0.1, 0.075, 0.15],
            [-0.1, 0.075, 0.15]
        ])
        self.frustum_lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # center to plane corners
            [1, 2], [2, 3], [3, 4], [4, 1]   # plane border
        ]
        self.frustum_colors = [[0, 255, 102] for _ in range(8)] # Green lines

        # Set constant view clipping planes
        view_ctl = self.vis.get_view_control()
        if view_ctl is not None:
            view_ctl.set_constant_z_near(0.01)
            view_ctl.set_constant_z_far(100.0)

        # Counter to throttle updates
        self.frame_idx = 0

    def update(self, T_wc, surfel_map, latest_color, depth_m):
        """
        Update the 3D visualizer and the 2D tracking HUD.
        Colors: (H,W,3) uint8 RGB, Depth: (H,W) float32 in meters.
        """
        self.frame_idx += 1


        # Transform camera pose to visualizer coordinate system
        T_wc_vis = self.T_w2v @ T_wc
        R_vis = T_wc_vis[0:3, 0:3]
        t_vis = T_wc_vis[0:3, 3]

        # ── 1. Update 3D Point Cloud (Throttled & Downsampled) ──────────────────
        if surfel_map.active_n > 0 and (self.frame_idx % 5 == 0 or self.frame_idx <= 1):
            # Downsample map representation to max 50,000 points to keep UI interaction smooth
            step = max(1, surfel_map.active_n // 50_000)
            
            with torch.no_grad():
                pos = surfel_map.positions[:surfel_map.active_n:step].cpu().numpy()
                col = surfel_map.colors[:surfel_map.active_n:step].cpu().numpy()
                
            # Apply world-to-visualizer rotation to map coordinates
            pos_vis = pos @ self.R_w2v.T
            
            self.vis.remove_geometry(self.pcd, reset_bounding_box=False)
            self.pcd.points = o3d.utility.Vector3dVector(pos_vis.astype(np.float64))
            self.pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64))
            self.vis.add_geometry(self.pcd, reset_bounding_box=False)

        # ── 2. Update Camera Frustum (Real-time updates) ────────────────────────
        transformed_vertices = (self.base_frustum_vertices @ R_vis.T) + t_vis
        
        self.frustum.points = o3d.utility.Vector3dVector(transformed_vertices)
        self.frustum.lines = o3d.utility.Vector2iVector(self.frustum_lines)
        self.frustum.colors = o3d.utility.Vector3dVector(np.array(self.frustum_colors) / 255.0)
        self.vis.update_geometry(self.frustum)

        # ── 3. Update Camera Trajectory Path (Throttled updates) ─────────────────
        self.trajectory_pts.append(t_vis.copy())
        if len(self.trajectory_pts) > 1 and (self.frame_idx % 5 == 0 or self.frame_idx <= 1):
            traj_pts_arr = np.array(self.trajectory_pts)
            lines = [[i, i+1] for i in range(len(self.trajectory_pts) - 1)]
            colors = [[255, 255, 0] for _ in range(len(lines))] # Yellow trajectory
            
            self.vis.remove_geometry(self.trajectory, reset_bounding_box=False)
            self.trajectory.points = o3d.utility.Vector3dVector(traj_pts_arr)
            self.trajectory.lines = o3d.utility.Vector2iVector(lines)
            self.trajectory.colors = o3d.utility.Vector3dVector(np.array(colors) / 255.0)
            self.vis.add_geometry(self.trajectory, reset_bounding_box=False)

        # Enforce constant near and far clipping planes to prevent clipping during movement
        view_ctl = self.vis.get_view_control()
        if view_ctl is not None:
            view_ctl.set_constant_z_near(0.01)
            view_ctl.set_constant_z_far(100.0)

        # Poll visualizer events
        self.vis.poll_events()
        self.vis.update_renderer()

        # Initialize visualizer viewpoint on the first frame
        if self.frame_idx == 1:
            view_ctl = self.vis.get_view_control()
            if view_ctl is not None:
                # Look from front-top-right towards the scene origin with +Y as UP
                view_ctl.set_front([0.5, 0.6, -0.6])
                view_ctl.set_lookat([0.0, 0.2, 0.5])
                view_ctl.set_up([0.0, 1.0, 0.0])
                view_ctl.set_zoom(0.8)

        # ── 4. Update 2D Tracking HUD (RGB + Depth colormap side-by-side) ──────
        # Convert RGB input to BGR for OpenCV
        rgb_bgr = cv2.cvtColor(latest_color, cv2.COLOR_RGB2BGR)
        
        # Normalize depth map to 0-255 range and apply colormap
        depth_norm = np.clip(depth_m / 4.0 * 255.0, 0, 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        
        # Combine side-by-side
        hud = np.hstack([rgb_bgr, depth_colormap])
        
        # Add overlay text
        cv2.putText(hud, f"Surfels: {surfel_map.active_n}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 102), 2)
        cv2.putText(hud, "LIVE RGB", (10, height := self.height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(hud, "DEPTH MAP (METERS)", (self.width + 10, height), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        cv2.imshow("RAM-SLAM Tracking HUD (ElasticFusion)", hud)
        cv2.waitKey(1)

    def spin_once(self):
        """Poll window and GUI events to keep windows responsive."""
        self.vis.poll_events()
        cv2.waitKey(1)

    def destroy(self):
        self.vis.destroy_window()
        cv2.destroyAllWindows()
