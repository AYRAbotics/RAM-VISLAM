import numpy as np
import torch
import open3d as o3d
from .diagnostics import metrics_logger
from .surfel_importance import compute_importance

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SurfelMap:
    def __init__(self, fx, fy, cx, cy, width=640, height=480, max_surfels=12_000_000):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.max_surfels = max_surfels
        
        # 1. State buffers on GPU
        self.positions = torch.zeros((max_surfels, 3), dtype=torch.float32, device=DEVICE)
        self.normals   = torch.zeros((max_surfels, 3), dtype=torch.float32, device=DEVICE)
        self.colors    = torch.zeros((max_surfels, 3), dtype=torch.float32, device=DEVICE)
        self.weights   = torch.zeros(max_surfels,      dtype=torch.float32, device=DEVICE)
        self.radii     = torch.zeros(max_surfels,      dtype=torch.float32, device=DEVICE)
        self.ages      = torch.zeros(max_surfels,      dtype=torch.int32,   device=DEVICE)
        self.kf_ids    = torch.zeros(max_surfels,      dtype=torch.int32,   device=DEVICE)
        
        # New GPU metadata buffers for Adaptive Surfel Importance Framework (Phase 1)
        self.unique_surfel_id = torch.zeros(max_surfels, dtype=torch.int32, device=DEVICE)
        self.creation_frame = torch.zeros(max_surfels, dtype=torch.int32, device=DEVICE)
        self.last_observed_frame = torch.zeros(max_surfels, dtype=torch.int32, device=DEVICE)
        self.observation_count = torch.zeros(max_surfels, dtype=torch.float32, device=DEVICE)
        self.fusion_count = torch.zeros(max_surfels, dtype=torch.float32, device=DEVICE)
        self.confidence_score = torch.zeros(max_surfels, dtype=torch.float32, device=DEVICE)
        self.position_variance = torch.zeros(max_surfels, dtype=torch.float32, device=DEVICE)
        self.normal_variance = torch.zeros(max_surfels, dtype=torch.float32, device=DEVICE)
        self.average_depth_confidence = torch.zeros(max_surfels, dtype=torch.float32, device=DEVICE)
        self.average_icp_fitness = torch.zeros(max_surfels, dtype=torch.float32, device=DEVICE)
        self.average_viewing_angle = torch.zeros(max_surfels, dtype=torch.float32, device=DEVICE)
        self.local_density = torch.zeros(max_surfels, dtype=torch.float32, device=DEVICE)
        self.importance_score = torch.zeros(max_surfels, dtype=torch.float32, device=DEVICE)
        self.total_spawned = 0
        
        self.active_n = 0
        
        # Constant grids for backprojection
        u_grid = torch.arange(width, dtype=torch.float32, device=DEVICE)
        v_grid = torch.arange(height, dtype=torch.float32, device=DEVICE)
        self.vv, self.uu = torch.meshgrid(v_grid, u_grid, indexing='ij')

    def _double_capacity(self):
        new_max = self.max_surfels * 2
        print(f"SurfelMap: Buffer capacity reached. Resizing GPU tensors from {self.max_surfels} to {new_max} surfels...", flush=True)
        
        new_positions = torch.zeros((new_max, 3), dtype=torch.float32, device=DEVICE)
        new_positions[:self.active_n] = self.positions[:self.active_n]
        self.positions = new_positions
        
        new_normals = torch.zeros((new_max, 3), dtype=torch.float32, device=DEVICE)
        new_normals[:self.active_n] = self.normals[:self.active_n]
        self.normals = new_normals
        
        new_colors = torch.zeros((new_max, 3), dtype=torch.float32, device=DEVICE)
        new_colors[:self.active_n] = self.colors[:self.active_n]
        self.colors = new_colors
        
        new_weights = torch.zeros(new_max, dtype=torch.float32, device=DEVICE)
        new_weights[:self.active_n] = self.weights[:self.active_n]
        self.weights = new_weights
        
        new_radii = torch.zeros(new_max, dtype=torch.float32, device=DEVICE)
        new_radii[:self.active_n] = self.radii[:self.active_n]
        self.radii = new_radii
        
        new_ages = torch.zeros(new_max, dtype=torch.int32, device=DEVICE)
        new_ages[:self.active_n] = self.ages[:self.active_n]
        self.ages = new_ages
        
        new_kf_ids = torch.zeros(new_max, dtype=torch.int32, device=DEVICE)
        new_kf_ids[:self.active_n] = self.kf_ids[:self.active_n]
        self.kf_ids = new_kf_ids
        
        new_unique_surfel_id = torch.zeros(new_max, dtype=torch.int32, device=DEVICE)
        new_unique_surfel_id[:self.active_n] = self.unique_surfel_id[:self.active_n]
        self.unique_surfel_id = new_unique_surfel_id
        
        new_creation_frame = torch.zeros(new_max, dtype=torch.int32, device=DEVICE)
        new_creation_frame[:self.active_n] = self.creation_frame[:self.active_n]
        self.creation_frame = new_creation_frame
        
        new_last_observed_frame = torch.zeros(new_max, dtype=torch.int32, device=DEVICE)
        new_last_observed_frame[:self.active_n] = self.last_observed_frame[:self.active_n]
        self.last_observed_frame = new_last_observed_frame
        
        new_observation_count = torch.zeros(new_max, dtype=torch.float32, device=DEVICE)
        new_observation_count[:self.active_n] = self.observation_count[:self.active_n]
        self.observation_count = new_observation_count
        
        new_fusion_count = torch.zeros(new_max, dtype=torch.float32, device=DEVICE)
        new_fusion_count[:self.active_n] = self.fusion_count[:self.active_n]
        self.fusion_count = new_fusion_count
        
        new_confidence_score = torch.zeros(new_max, dtype=torch.float32, device=DEVICE)
        new_confidence_score[:self.active_n] = self.confidence_score[:self.active_n]
        self.confidence_score = new_confidence_score
        
        new_position_variance = torch.zeros(new_max, dtype=torch.float32, device=DEVICE)
        new_position_variance[:self.active_n] = self.position_variance[:self.active_n]
        self.position_variance = new_position_variance
        
        new_normal_variance = torch.zeros(new_max, dtype=torch.float32, device=DEVICE)
        new_normal_variance[:self.active_n] = self.normal_variance[:self.active_n]
        self.normal_variance = new_normal_variance
        
        new_average_depth_confidence = torch.zeros(new_max, dtype=torch.float32, device=DEVICE)
        new_average_depth_confidence[:self.active_n] = self.average_depth_confidence[:self.active_n]
        self.average_depth_confidence = new_average_depth_confidence
        
        new_average_icp_fitness = torch.zeros(new_max, dtype=torch.float32, device=DEVICE)
        new_average_icp_fitness[:self.active_n] = self.average_icp_fitness[:self.active_n]
        self.average_icp_fitness = new_average_icp_fitness
        
        new_average_viewing_angle = torch.zeros(new_max, dtype=torch.float32, device=DEVICE)
        new_average_viewing_angle[:self.active_n] = self.average_viewing_angle[:self.active_n]
        self.average_viewing_angle = new_average_viewing_angle
        
        new_local_density = torch.zeros(new_max, dtype=torch.float32, device=DEVICE)
        new_local_density[:self.active_n] = self.local_density[:self.active_n]
        self.local_density = new_local_density
        
        new_importance_score = torch.zeros(new_max, dtype=torch.float32, device=DEVICE)
        new_importance_score[:self.active_n] = self.importance_score[:self.active_n]
        self.importance_score = new_importance_score
        
        self.max_surfels = new_max

    def update_densities(self, radius=0.05):
        if self.active_n == 0:
            return
        
        # Vectorized voxel-based local density estimation
        positions = self.positions[:self.active_n]
        voxel_coords = torch.round(positions / radius).long()
        
        # Count occurrences of each voxel hash
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_coords, dim=0, return_inverse=True, return_counts=True
        )
        
        # For each surfel, neighbors count = voxel count - 1
        self.local_density[:self.active_n] = torch.clamp(counts[inverse_indices].float() - 1.0, min=0.0)

    def fuse_frame(self, color_img, depth_aligned, T_wc, frame_id, kf_id, icp_fitness=None):
        """
        Fuses a new aligned RGB-D frame into the surfel map.
        T_wc: 4x4 homogenous camera-to-world transform.
        """
        fused_n = 0
        # Transfer current frame to GPU
        color_t = torch.tensor(color_img, dtype=torch.float32, device=DEVICE) / 255.0  # (H, W, 3)
        depth_t = torch.tensor(depth_aligned, dtype=torch.float32, device=DEVICE)       # (H, W)
        
        # Extract R_cw and t_cw from inverse pose
        T_cw = np.linalg.inv(T_wc)
        R_cw = torch.tensor(T_cw[0:3, 0:3], dtype=torch.float32, device=DEVICE)
        t_cw = torch.tensor(T_cw[0:3, 3], dtype=torch.float32, device=DEVICE)
        
        R_wc = torch.tensor(T_wc[0:3, 0:3], dtype=torch.float32, device=DEVICE)
        t_wc = torch.tensor(T_wc[0:3, 3], dtype=torch.float32, device=DEVICE)

        # Get frame's ICP fitness from metrics_logger if not passed
        if icp_fitness is None:
            with metrics_logger.lock:
                icp_fitness = metrics_logger.current_frame_metrics.get("icp_fitness", 1.0)
        if icp_fitness is None:
            icp_fitness = 1.0

        # Compute depth confidence map
        depth_conf_map = torch.exp(-0.25 * depth_t)

        # 1. Backproject frame to 3D camera frame and compute normals
        valid_depth = depth_t > 0.1
        z_cam = torch.where(valid_depth, depth_t, torch.tensor(0.0, device=DEVICE))
        
        x_cam = (self.uu - self.cx) * z_cam / self.fx
        y_cam = (self.vv - self.cy) * z_cam / self.fy
        pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (H, W, 3)
        
        # Compute normals using cross product of spatial neighbors
        pts_right = torch.zeros_like(pts_cam)
        pts_right[:, :-1] = pts_cam[:, 1:]
        pts_right[:, -1] = pts_cam[:, -1]
        
        pts_down = torch.zeros_like(pts_cam)
        pts_down[:-1, :] = pts_cam[1:]
        pts_down[-1, :] = pts_cam[-1]
        
        v1 = pts_right - pts_cam
        v2 = pts_down - pts_cam
        normals_cam = torch.cross(v1, v2, dim=-1)
        norm_len = torch.norm(normals_cam, dim=-1, keepdim=True)
        normals_cam = torch.where(norm_len > 1e-6, normals_cam / norm_len, torch.tensor([0.0, 0.0, -1.0], device=DEVICE))
        
        # Ensure normals point back towards the camera origin
        normals_cam = torch.where(normals_cam[:, :, 2:3] > 0, -normals_cam, normals_cam)
        
        # Transform points and normals to world frame
        pts_world = pts_cam @ R_wc.T + t_wc
        normals_world = normals_cam @ R_wc.T

        # 2. Project existing map surfels to Z-buffer
        index_map = torch.full((self.height, self.width), -1, dtype=torch.long, device=DEVICE)
        
        if self.active_n > 0:
            map_pts = self.positions[:self.active_n]
            # Transform to camera frame
            pts_c = map_pts @ R_cw.T + t_cw
            z_c = pts_c[:, 2]
            
            valid_z = z_c > 0.1
            u_proj = (pts_c[:, 0] * self.fx) / z_c + self.cx
            v_proj = (pts_c[:, 1] * self.fy) / z_c + self.cy
            
            u_idx = torch.round(u_proj).long()
            v_idx = torch.round(v_proj).long()
            
            valid_bounds = (u_idx >= 0) & (u_idx < self.width) & (v_idx >= 0) & (v_idx < self.height)
            valid_surfel_mask = valid_z & valid_bounds
            
            # Retrieve active indices that project into boundaries
            valid_s_idx = torch.where(valid_surfel_mask)[0]
            if len(valid_s_idx) > 0:
                valid_z_vals = z_c[valid_s_idx]
                # Sort descending (furthest first, closest last) so closest overwrites in scatter
                sort_idx = torch.argsort(valid_z_vals, descending=True)
                sorted_s_idx = valid_s_idx[sort_idx]
                
                u_idx_s = u_idx[sorted_s_idx]
                v_idx_s = v_idx[sorted_s_idx]
                
                flat_pixel_idx = v_idx_s * self.width + u_idx_s
                index_map.view(-1)[flat_pixel_idx] = sorted_s_idx

        # 3. Classify into Fusion or Spawn
        has_map_surfel = index_map != -1
        fuse_mask = valid_depth & has_map_surfel
        
        # Fusion compatibility check
        comp_s_idx = torch.tensor([], dtype=torch.long, device=DEVICE)
        if fuse_mask.any():
            s_idx = index_map[fuse_mask]
            
            # Fetch attributes
            m_pos = self.positions[s_idx]
            m_nor = self.normals[s_idx]
            
            f_pos = pts_world[fuse_mask]
            f_nor = normals_world[fuse_mask]
            
            pos_diff = torch.norm(m_pos - f_pos, dim=-1)
            nor_align = torch.sum(m_nor * f_nor, dim=-1)
            
            compatible = (pos_diff < 0.05) & (nor_align > 0.75)
            
            # Increment observation counts and update last observed frames for ALL observed surfels
            observed_s_idx = s_idx
            if len(observed_s_idx) > 0:
                self.observation_count[observed_s_idx] += 1.0
                self.last_observed_frame[observed_s_idx] = frame_id
            
            if compatible.any():
                fused_n = torch.sum(compatible).item()
                comp_s_idx = s_idx[compatible]
                comp_f_pos = f_pos[compatible]
                comp_f_nor = f_nor[compatible]
                comp_f_col = color_t[fuse_mask][compatible]
                comp_f_depth_conf = depth_conf_map[fuse_mask][compatible]
                
                # Increment fusion counts
                self.fusion_count[comp_s_idx] += 1.0
                
                # Compute viewing angle for compatible points
                surf_to_cam = t_wc.unsqueeze(0) - comp_f_pos
                surf_to_cam_norm = torch.norm(surf_to_cam, dim=-1, keepdim=True)
                v = surf_to_cam / (surf_to_cam_norm + 1e-6)
                cos_angle = torch.sum(comp_f_nor * v, dim=-1)
                comp_f_view_angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
                
                # Calculate differences with old means for variance tracking
                diff_pos_old = comp_f_pos - self.positions[comp_s_idx]
                diff_nor_old = comp_f_nor - self.normals[comp_s_idx]
                
                # Perform weighted update of nominal attributes
                w_old = self.weights[comp_s_idx]
                w_new = w_old + 1.0
                
                self.positions[comp_s_idx] = (self.positions[comp_s_idx] * w_old.unsqueeze(-1) + comp_f_pos) / w_new.unsqueeze(-1)
                
                new_nor = (self.normals[comp_s_idx] * w_old.unsqueeze(-1) + comp_f_nor) / w_new.unsqueeze(-1)
                self.normals[comp_s_idx] = new_nor / torch.norm(new_nor, dim=-1, keepdim=True)
                
                # Calculate differences with new means
                diff_pos_new = comp_f_pos - self.positions[comp_s_idx]
                diff_nor_new = comp_f_nor - self.normals[comp_s_idx]
                
                # Update running variances using Welford's formula
                m2_pos_old = self.position_variance[comp_s_idx] * w_old
                m2_pos_new = m2_pos_old + torch.sum(diff_pos_old * diff_pos_new, dim=-1)
                self.position_variance[comp_s_idx] = m2_pos_new / w_new
                
                m2_nor_old = self.normal_variance[comp_s_idx] * w_old
                m2_nor_new = m2_nor_old + torch.sum(diff_nor_old * diff_nor_new, dim=-1)
                self.normal_variance[comp_s_idx] = m2_nor_new / w_new
                
                # Update running averages
                self.average_depth_confidence[comp_s_idx] = (self.average_depth_confidence[comp_s_idx] * w_old + comp_f_depth_conf) / w_new
                self.average_icp_fitness[comp_s_idx] = (self.average_icp_fitness[comp_s_idx] * w_old + icp_fitness) / w_new
                self.average_viewing_angle[comp_s_idx] = (self.average_viewing_angle[comp_s_idx] * w_old + comp_f_view_angle) / w_new
                
                # Update confidence_score field
                self.confidence_score[comp_s_idx] = self.average_depth_confidence[comp_s_idx] * (1.0 - torch.exp(-0.1 * self.observation_count[comp_s_idx]))
                
                self.colors[comp_s_idx] = (self.colors[comp_s_idx] * w_old.unsqueeze(-1) + comp_f_col) / w_new.unsqueeze(-1)
                self.weights[comp_s_idx] = w_new
                self.ages[comp_s_idx] = frame_id

        # 4. Spawning new surfels
        # Spawn if depth is valid and no surfel projected, or surfel was incompatible
        spawn_mask = valid_depth & (~has_map_surfel)
        
        # If the projected surfel is incompatible, spawn a new surfel to capture this surface layer
        if fuse_mask.any():
            incompatible = ~compatible
            if incompatible.any():
                incompatible_pixels = torch.zeros_like(valid_depth)
                incompatible_pixels[fuse_mask] = incompatible
                spawn_mask = spawn_mask | incompatible_pixels
        
        # Add dynamic/behind spawn
        behind_mask = valid_depth & has_map_surfel
        if behind_mask.any():
            s_idx_b = index_map[behind_mask]
            m_z_cam = (self.positions[s_idx_b] @ R_cw.T + t_cw)[:, 2]
            f_z_cam = depth_t[behind_mask]
            is_front = f_z_cam < (m_z_cam - 0.08)  # new geometry in front of current surfel
            spawn_mask[behind_mask] = spawn_mask[behind_mask] | is_front

        spawn_n = torch.sum(spawn_mask).item()
        if spawn_n > 0:
            while self.active_n + spawn_n >= self.max_surfels:
                self._double_capacity()
                
            new_s_indices = torch.arange(self.active_n, self.active_n + spawn_n, device=DEVICE)
            self.positions[new_s_indices] = pts_world[spawn_mask]
            self.normals[new_s_indices]   = normals_world[spawn_mask]
            self.colors[new_s_indices]    = color_t[spawn_mask]
            self.weights[new_s_indices]   = 1.0
            self.radii[new_s_indices]     = 0.01 + 0.01 * (pts_cam[spawn_mask, 2] / 4.0)
            self.ages[new_s_indices]      = frame_id
            self.kf_ids[new_s_indices]    = kf_id
            
            # Spawn metadata initialization
            self.unique_surfel_id[new_s_indices] = torch.arange(self.total_spawned, self.total_spawned + spawn_n, dtype=torch.int32, device=DEVICE)
            self.total_spawned += spawn_n
            
            self.creation_frame[new_s_indices] = frame_id
            self.last_observed_frame[new_s_indices] = frame_id
            self.observation_count[new_s_indices] = 1.0
            self.fusion_count[new_s_indices] = 0.0
            
            spawn_depth_conf = depth_conf_map[spawn_mask]
            self.confidence_score[new_s_indices] = spawn_depth_conf
            self.position_variance[new_s_indices] = 0.0
            self.normal_variance[new_s_indices] = 0.0
            self.average_depth_confidence[new_s_indices] = spawn_depth_conf
            self.average_icp_fitness[new_s_indices] = icp_fitness
            
            # Compute initial viewing angle for spawned surfels
            s_pts = pts_world[spawn_mask]
            s_nor = normals_world[spawn_mask]
            s_surf_to_cam = t_wc.unsqueeze(0) - s_pts
            s_surf_to_cam_norm = torch.norm(s_surf_to_cam, dim=-1, keepdim=True)
            s_v = s_surf_to_cam / (s_surf_to_cam_norm + 1e-6)
            s_cos_angle = torch.sum(s_nor * s_v, dim=-1)
            self.average_viewing_angle[new_s_indices] = torch.acos(torch.clamp(s_cos_angle, -1.0, 1.0))
            
            self.local_density[new_s_indices] = 0.0
            self.importance_score[new_s_indices] = 0.0
            
            self.active_n += spawn_n
            
        # 5. Local density estimation and importance score updates
        self.update_densities(radius=0.05)
        self.importance_score[:self.active_n] = compute_importance(self, current_frame_id=frame_id)
            
        metrics_logger.log("spawned_surfels", spawn_n)
        metrics_logger.log("fused_surfels", fused_n)
        metrics_logger.log("active_surfels", self.active_n)
        
        # 6. Log diagnostics metrics
        if self.active_n > 0:
            importances = self.importance_score[:self.active_n]
            obs_counts = self.observation_count[:self.active_n]
            fus_counts = self.fusion_count[:self.active_n]
            densities = self.local_density[:self.active_n]
            
            metrics_logger.log("mean_importance", float(torch.mean(importances).item()))
            metrics_logger.log("median_importance", float(torch.median(importances).item()))
            metrics_logger.log("max_importance", float(torch.max(importances).item()))
            metrics_logger.log("min_importance", float(torch.min(importances).item()))
            metrics_logger.log("avg_observation_count", float(torch.mean(obs_counts).item()))
            metrics_logger.log("avg_fusion_count", float(torch.mean(fus_counts).item()))
            metrics_logger.log("avg_local_density", float(torch.mean(densities).item()))
        else:
            metrics_logger.log("mean_importance", 0.0)
            metrics_logger.log("median_importance", 0.0)
            metrics_logger.log("max_importance", 0.0)
            metrics_logger.log("min_importance", 0.0)
            metrics_logger.log("avg_observation_count", 0.0)
            metrics_logger.log("avg_fusion_count", 0.0)
            metrics_logger.log("avg_local_density", 0.0)

    def prune_unstable(self, current_frame_id, min_weight=3.0):
        """Remove unstable surfels (e.g., low weight and aged)."""
        if self.active_n == 0:
            return
        
        age_diff = current_frame_id - self.ages[:self.active_n]
        # Keep if weight is high enough, or if it is young (to allow fusion to accumulate weight)
        keep = (self.weights[:self.active_n] >= min_weight) | (age_diff < 30)
        
        keep_n = torch.sum(keep).item()
        if keep_n < self.active_n:
            self.positions[:keep_n] = self.positions[:self.active_n][keep]
            self.normals[:keep_n]   = self.normals[:self.active_n][keep]
            self.colors[:keep_n]    = self.colors[:self.active_n][keep]
            self.weights[:keep_n]   = self.weights[:self.active_n][keep]
            self.radii[:keep_n]     = self.radii[:self.active_n][keep]
            self.ages[:keep_n]      = self.ages[:self.active_n][keep]
            self.kf_ids[:keep_n]    = self.kf_ids[:self.active_n][keep]
            
            # Prune custom metadata fields
            self.unique_surfel_id[:keep_n] = self.unique_surfel_id[:self.active_n][keep]
            self.creation_frame[:keep_n] = self.creation_frame[:self.active_n][keep]
            self.last_observed_frame[:keep_n] = self.last_observed_frame[:self.active_n][keep]
            self.observation_count[:keep_n] = self.observation_count[:self.active_n][keep]
            self.fusion_count[:keep_n] = self.fusion_count[:self.active_n][keep]
            self.confidence_score[:keep_n] = self.confidence_score[:self.active_n][keep]
            self.position_variance[:keep_n] = self.position_variance[:self.active_n][keep]
            self.normal_variance[:keep_n] = self.normal_variance[:self.active_n][keep]
            self.average_depth_confidence[:keep_n] = self.average_depth_confidence[:self.active_n][keep]
            self.average_icp_fitness[:keep_n] = self.average_icp_fitness[:self.active_n][keep]
            self.average_viewing_angle[:keep_n] = self.average_viewing_angle[:self.active_n][keep]
            self.local_density[:keep_n] = self.local_density[:self.active_n][keep]
            self.importance_score[:keep_n] = self.importance_score[:self.active_n][keep]
            
            metrics_logger.log("pruned_surfels", self.active_n - keep_n)
            self.active_n = keep_n
        else:
            metrics_logger.log("pruned_surfels", 0)

    def merge_voxels(self, voxel_size=0.015):
        """Vectorized spatial hash voxel grid filtering to merge duplicate surfels."""
        if self.active_n == 0:
            return
        
        # Round positions to find voxel coordinates
        voxel_coords = torch.round(self.positions[:self.active_n] / voxel_size).long()
        
        # Unique voxels and mapping
        unique_voxels, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
        keep_n = len(unique_voxels)
        
        if keep_n == self.active_n:
            return  # No duplicates to merge
            
        new_positions = torch.zeros((keep_n, 3), dtype=torch.float32, device=DEVICE)
        new_normals   = torch.zeros((keep_n, 3), dtype=torch.float32, device=DEVICE)
        new_colors    = torch.zeros((keep_n, 3), dtype=torch.float32, device=DEVICE)
        new_weights   = torch.zeros(keep_n,      dtype=torch.float32, device=DEVICE)
        new_radii     = torch.zeros(keep_n,      dtype=torch.float32, device=DEVICE)
        new_ages      = torch.zeros(keep_n,      dtype=torch.int32,   device=DEVICE)
        new_kf_ids    = torch.zeros(keep_n,      dtype=torch.int32,   device=DEVICE)
        
        w = self.weights[:self.active_n]
        
        # Accumulate weights
        new_weights.scatter_add_(0, inverse_indices, w)
        
        # Weighted accumulation of positions, normals, colors, and radii
        new_positions.scatter_add_(0, inverse_indices.unsqueeze(-1).expand(-1, 3), self.positions[:self.active_n] * w.unsqueeze(-1))
        new_normals.scatter_add_(0, inverse_indices.unsqueeze(-1).expand(-1, 3), self.normals[:self.active_n] * w.unsqueeze(-1))
        new_colors.scatter_add_(0, inverse_indices.unsqueeze(-1).expand(-1, 3), self.colors[:self.active_n] * w.unsqueeze(-1))
        new_radii.scatter_add_(0, inverse_indices, self.radii[:self.active_n] * w)
        
        # Normalize weighted components
        valid_w = new_weights > 0.0
        new_positions[valid_w] /= new_weights[valid_w].unsqueeze(-1)
        new_normals[valid_w]   = new_normals[valid_w] / torch.norm(new_normals[valid_w], dim=-1, keepdim=True)
        new_colors[valid_w]    /= new_weights[valid_w].unsqueeze(-1)
        new_radii[valid_w]     /= new_weights[valid_w]
        
        # Reduce ages and kf_ids (picking maximum/most recent)
        new_ages.scatter_reduce_(0, inverse_indices, self.ages[:self.active_n], reduce='amax', include_self=False)
        new_kf_ids.scatter_reduce_(0, inverse_indices, self.kf_ids[:self.active_n], reduce='amax', include_self=False)
        
        # Merge custom metadata fields
        new_unique_surfel_id = torch.zeros(keep_n, dtype=torch.int32, device=DEVICE)
        new_creation_frame = torch.zeros(keep_n, dtype=torch.int32, device=DEVICE)
        new_last_observed_frame = torch.zeros(keep_n, dtype=torch.int32, device=DEVICE)
        new_observation_count = torch.zeros(keep_n, dtype=torch.float32, device=DEVICE)
        new_fusion_count = torch.zeros(keep_n, dtype=torch.float32, device=DEVICE)
        new_confidence_score = torch.zeros(keep_n, dtype=torch.float32, device=DEVICE)
        new_position_variance = torch.zeros(keep_n, dtype=torch.float32, device=DEVICE)
        new_normal_variance = torch.zeros(keep_n, dtype=torch.float32, device=DEVICE)
        new_average_depth_confidence = torch.zeros(keep_n, dtype=torch.float32, device=DEVICE)
        new_average_icp_fitness = torch.zeros(keep_n, dtype=torch.float32, device=DEVICE)
        new_average_viewing_angle = torch.zeros(keep_n, dtype=torch.float32, device=DEVICE)
        new_local_density = torch.zeros(keep_n, dtype=torch.float32, device=DEVICE)
        new_importance_score = torch.zeros(keep_n, dtype=torch.float32, device=DEVICE)

        # Merge ID and frames using reduction
        new_unique_surfel_id.scatter_reduce_(0, inverse_indices, self.unique_surfel_id[:self.active_n], reduce='amax', include_self=False)
        new_creation_frame.scatter_reduce_(0, inverse_indices, self.creation_frame[:self.active_n], reduce='amin', include_self=False)
        new_last_observed_frame.scatter_reduce_(0, inverse_indices, self.last_observed_frame[:self.active_n], reduce='amax', include_self=False)
        
        # Sum counts
        new_observation_count.scatter_add_(0, inverse_indices, self.observation_count[:self.active_n])
        new_fusion_count.scatter_add_(0, inverse_indices, self.fusion_count[:self.active_n])
        
        # Weighted averages for stats
        new_confidence_score.scatter_add_(0, inverse_indices, self.confidence_score[:self.active_n] * w)
        new_average_depth_confidence.scatter_add_(0, inverse_indices, self.average_depth_confidence[:self.active_n] * w)
        new_average_icp_fitness.scatter_add_(0, inverse_indices, self.average_icp_fitness[:self.active_n] * w)
        new_average_viewing_angle.scatter_add_(0, inverse_indices, self.average_viewing_angle[:self.active_n] * w)
        
        new_confidence_score[valid_w] /= new_weights[valid_w]
        new_average_depth_confidence[valid_w] /= new_weights[valid_w]
        new_average_icp_fitness[valid_w] /= new_weights[valid_w]
        new_average_viewing_angle[valid_w] /= new_weights[valid_w]
        
        # Variances and other metrics
        new_position_variance.scatter_reduce_(0, inverse_indices, self.position_variance[:self.active_n], reduce='amax', include_self=False)
        new_normal_variance.scatter_reduce_(0, inverse_indices, self.normal_variance[:self.active_n], reduce='amax', include_self=False)
        new_local_density.scatter_reduce_(0, inverse_indices, self.local_density[:self.active_n], reduce='amax', include_self=False)
        new_importance_score.scatter_reduce_(0, inverse_indices, self.importance_score[:self.active_n], reduce='amax', include_self=False)
        
        # Overwrite buffers
        self.positions[:keep_n] = new_positions
        self.normals[:keep_n]   = new_normals
        self.colors[:keep_n]    = new_colors
        self.weights[:keep_n]   = new_weights
        self.radii[:keep_n]     = new_radii
        self.ages[:keep_n]      = new_ages
        self.kf_ids[:keep_n]    = new_kf_ids
        
        self.unique_surfel_id[:keep_n] = new_unique_surfel_id
        self.creation_frame[:keep_n] = new_creation_frame
        self.last_observed_frame[:keep_n] = new_last_observed_frame
        self.observation_count[:keep_n] = new_observation_count
        self.fusion_count[:keep_n] = new_fusion_count
        self.confidence_score[:keep_n] = new_confidence_score
        self.position_variance[:keep_n] = new_position_variance
        self.normal_variance[:keep_n] = new_normal_variance
        self.average_depth_confidence[:keep_n] = new_average_depth_confidence
        self.average_icp_fitness[:keep_n] = new_average_icp_fitness
        self.average_viewing_angle[:keep_n] = new_average_viewing_angle
        self.local_density[:keep_n] = new_local_density
        self.importance_score[:keep_n] = new_importance_score
        
        self.active_n = keep_n

    def export_ply(self, filepath):
        """Export the active surfels as a PLY mesh file."""
        if self.active_n == 0:
            print("SurfelMap: Map is empty, skipping PLY export.")
            return
            
        pos = self.positions[:self.active_n].cpu().numpy()
        nor = self.normals[:self.active_n].cpu().numpy()
        col = (self.colors[:self.active_n].cpu().numpy() * 255.0).astype(np.uint8)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos.astype(np.float64))
        pcd.normals = o3d.utility.Vector3dVector(nor.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64) / 255.0)
        
        o3d.io.write_point_cloud(filepath, pcd, write_ascii=False)
        print(f"SurfelMap: Exported {self.active_n} surfels to {filepath}")