#!/usr/bin/env python3
"""
RAM-VI-S Structural Plane Detection & Adaptive Confidence Module
Extracts 3D planar primitives from depth maps, tracks persistent plane landmarks,
classifies planes using ESKF IMU gravity, and computes soft adaptive confidence scores.
"""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

class PlaneLandmark:
    """Dataclass representing a persistent structural 3D plane landmark in world frame."""
    def __init__(self, plane_id, normal_w, d_w, plane_type="UNKNOWN", area=0.0, inliers=0, rmse=0.0, frame_id=0):
        self.plane_id = plane_id
        self.normal_w = normal_w / np.linalg.norm(normal_w)  # 3D unit normal in world frame
        self.d_w = float(d_w)                                 # Signed distance from origin
        self.plane_type = plane_type                         # FLOOR, WALL, CEILING, UNKNOWN
        self.area = float(area)
        self.inliers = int(inliers)
        self.rmse = float(rmse)
        self.observation_count = 1
        self.first_observed_frame = frame_id
        self.last_observed_frame = frame_id
        self.confidence = 0.5

    def update(self, normal_w, d_w, area, inliers, rmse, frame_id):
        """Updates landmark parameters with a new compatible observation."""
        alpha = 0.3  # Exponential moving average factor
        normal_w = normal_w / np.linalg.norm(normal_w)
        if np.dot(self.normal_w, normal_w) < 0:
            normal_w = -normal_w
            d_w = -d_w

        self.normal_w = (1 - alpha) * self.normal_w + alpha * normal_w
        self.normal_w /= np.linalg.norm(self.normal_w)
        self.d_w = (1 - alpha) * self.d_w + alpha * d_w
        self.area = max(self.area, area)
        self.inliers += inliers
        self.rmse = (1 - alpha) * self.rmse + alpha * rmse
        self.observation_count += 1
        self.last_observed_frame = frame_id

class StructuralPlaneDetector:
    """Detects, tracks, and classifies structural plane landmarks using IMU gravity."""
    def __init__(self, fx=525.0, fy=525.0, cx=319.5, cy=239.5, max_planes=5):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.max_planes = max_planes
        
        self.landmarks = {}  # plane_id -> PlaneLandmark
        self.next_plane_id = 0

    def depth_to_point_cloud(self, depth_m):
        """Converts depth map to Open3D PointCloud."""
        h, w = depth_m.shape
        ys, xs = np.where(depth_m > 0.2)  # Ignore close noise < 0.2m
        zs = depth_m[ys, xs]
        
        # Subsample to max 10,000 points for real-time 30FPS efficiency
        if len(zs) > 10000:
            idx = np.random.choice(len(zs), 10000, replace=False)
            ys, xs, zs = ys[idx], xs[idx], zs[idx]

        px = (xs - self.cx) * zs / self.fx
        py = (ys - self.cy) * zs / self.fy
        pts = np.vstack((px, py, zs)).T

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    def classify_plane_with_gravity(self, normal_w, g_unit=np.array([0.0, 0.0, -1.0])):
        """
        Classifies plane using IMU gravity vector g_unit in world frame.
        g_unit is typically [0, 0, -1] in gravity-aligned world frame.
        """
        align_dot = np.dot(normal_w, g_unit)
        
        # Floor normal points UP (against gravity vector [0, 0, -1] -> dot ~ -1.0 if g=[0,0,-1])
        if align_dot < -0.85:
            return "FLOOR"
        elif align_dot > 0.85:
            return "CEILING"
        elif abs(align_dot) < 0.25:
            return "WALL"
        else:
            return "UNKNOWN"

    def compute_adaptive_confidence(self, landmark, g_unit=np.array([0.0, 0.0, -1.0])):
        """
        Computes soft structural confidence C_struct in [0, 1].
        High score if plane is large, persistent, low RMSE, and gravity-aligned.
        """
        # Area score (max at 2.0 m^2)
        w_area = min(1.0, landmark.area / 2.0)
        
        # RMSE score (max at RMSE <= 0.01m)
        w_rmse = max(0.0, 1.0 - (landmark.rmse / 0.05))
        
        # Persistence score (max at N_obs >= 5)
        w_persist = min(1.0, landmark.observation_count / 5.0)
        
        # Gravity alignment score
        align_dot = np.dot(landmark.normal_w, g_unit)
        if landmark.plane_type == "FLOOR" or landmark.plane_type == "CEILING":
            w_grav = abs(align_dot)
        elif landmark.plane_type == "WALL":
            w_grav = 1.0 - abs(align_dot)
        else:
            w_grav = 0.1

        # Composite soft weight
        c_struct = 0.3 * w_area + 0.3 * w_rmse + 0.2 * w_persist + 0.2 * w_grav
        return float(np.clip(c_struct, 0.0, 1.0))

    def detect_and_track_planes(self, depth_m, T_wc, frame_id, g_unit=np.array([0.0, 0.0, -1.0])):
        """
        Extracts dominant 3D planes from depth image, matches them with persistent landmarks,
        and returns active structural landmarks with confidence scores.
        """
        pcd = self.depth_to_point_cloud(depth_m)
        if len(pcd.points) < 100:
            return []

        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]

        detected_planes = []
        remaining_pcd = pcd

        for _ in range(self.max_planes):
            if len(remaining_pcd.points) < 100:
                break
            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=0.03, ransac_n=3, num_iterations=100
            )
            if len(inliers) < 150:
                break

            a, b, c, d_c = plane_model
            n_c = np.array([a, b, c])
            norm = np.linalg.norm(n_c)
            if norm == 0:
                continue
            n_c = n_c / norm
            d_c = d_c / norm

            # Transform plane equation to world frame
            n_w = R_wc @ n_c
            d_w = d_c - np.dot(n_w, t_wc)

            # Estimate plane area approximation
            inlier_pts = np.asarray(remaining_pcd.select_by_index(inliers).points)
            rmse = float(np.sqrt(np.mean((inlier_pts @ n_c + d_c)**2)))
            area = float(len(inliers) * 0.001)  # approx area scale

            # Classify plane type using IMU gravity
            ptype = self.classify_plane_with_gravity(n_w, g_unit)

            detected_planes.append({
                "normal_w": n_w,
                "d_w": d_w,
                "plane_type": ptype,
                "area": area,
                "inliers": len(inliers),
                "rmse": rmse
            })

            # Remove inliers for next plane iteration
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

        # Track and associate detected planes with existing landmarks
        active_landmarks = []
        for det in detected_planes:
            n_w = det["normal_w"]
            d_w = det["d_w"]
            
            matched_id = None
            for lm_id, lm in self.landmarks.items():
                # Check normal alignment and distance difference
                normal_sim = abs(np.dot(lm.normal_w, n_w))
                dist_diff = abs(lm.d_w - d_w)
                if normal_sim > 0.92 and dist_diff < 0.15:
                    matched_id = lm_id
                    break

            if matched_id is not None:
                lm = self.landmarks[matched_id]
                lm.update(n_w, d_w, det["area"], det["inliers"], det["rmse"], frame_id)
            else:
                lm_id = self.next_plane_id
                self.next_plane_id += 1
                lm = PlaneLandmark(lm_id, n_w, d_w, det["plane_type"], det["area"], det["inliers"], det["rmse"], frame_id)
                self.landmarks[lm_id] = lm

            lm.confidence = self.compute_adaptive_confidence(lm, g_unit)
            active_landmarks.append(lm)

        return active_landmarks
