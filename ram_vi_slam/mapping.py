import numpy as np
import torch
import open3d as o3d

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
        
        self.max_surfels = new_max

    def fuse_frame(self, color_img, depth_aligned, T_wc, frame_id, kf_id):
        """
        Fuses a new aligned RGB-D frame into the surfel map.
        T_wc: 4x4 homogenous camera-to-world transform.
        """
        # Transfer current frame to GPU
        color_t = torch.tensor(color_img, dtype=torch.float32, device=DEVICE) / 255.0  # (H, W, 3)
        depth_t = torch.tensor(depth_aligned, dtype=torch.float32, device=DEVICE)       # (H, W)
        
        # Extract R_cw and t_cw from inverse pose
        T_cw = np.linalg.inv(T_wc)
        R_cw = torch.tensor(T_cw[0:3, 0:3], dtype=torch.float32, device=DEVICE)
        t_cw = torch.tensor(T_cw[0:3, 3], dtype=torch.float32, device=DEVICE)
        
        R_wc = torch.tensor(T_wc[0:3, 0:3], dtype=torch.float32, device=DEVICE)
        t_wc = torch.tensor(T_wc[0:3, 3], dtype=torch.float32, device=DEVICE)

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
            
            if compatible.any():
                comp_s_idx = s_idx[compatible]
                comp_f_pos = f_pos[compatible]
                comp_f_nor = f_nor[compatible]
                comp_f_col = color_t[fuse_mask][compatible]
                
                # Perform weighted update
                w_old = self.weights[comp_s_idx]
                w_new = w_old + 1.0
                
                self.positions[comp_s_idx] = (self.positions[comp_s_idx] * w_old.unsqueeze(-1) + comp_f_pos) / w_new.unsqueeze(-1)
                
                new_nor = (self.normals[comp_s_idx] * w_old.unsqueeze(-1) + comp_f_nor) / w_new.unsqueeze(-1)
                self.normals[comp_s_idx] = new_nor / torch.norm(new_nor, dim=-1, keepdim=True)
                
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
            self.active_n += spawn_n

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
            self.active_n = keep_n

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
        
        # Overwrite buffers
        self.positions[:keep_n] = new_positions
        self.normals[:keep_n]   = new_normals
        self.colors[:keep_n]    = new_colors
        self.weights[:keep_n]   = new_weights
        self.radii[:keep_n]     = new_radii
        self.ages[:keep_n]      = new_ages
        self.kf_ids[:keep_n]    = new_kf_ids
        
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