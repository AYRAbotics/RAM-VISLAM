import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import faiss
import open3d as o3d
from .diagnostics import metrics_logger

class DINOv2Extractor:
    def __init__(self):
        # Run on CPU to keep all GPU memory free for the surfel mapping
        self.device = torch.device('cpu')
        print("DINOv2: Loading model vit_s_14 on CPU...")
        # Load local or hub model
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, cv_img_rgb):
        """Extract L2-normalized 384-dimensional descriptor from RGB image."""
        pil_img = Image.fromarray(cv_img_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        emb = self.model(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

class LoopDetector:
    def __init__(self, fx, fy, cx, cy, width=640, height=480):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )
        
        self.extractor = DINOv2Extractor()
        # FAISS Index for Inner Product search (cosine similarity on normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(384)
        
        self.kf_list = []  # mapping FAISS row idx -> kf_id
        self.db = {}       # kf_id -> (color_img, depth_img, T_wc)
        
        # Thresholds
        self.min_id_diff = 40  # ignore candidate keyframes within last 40 keyframes
        self.sim_threshold = 0.90 # DINOv2 similarity threshold

    def add_keyframe(self, kf_id, color_img_rgb, depth_aligned, T_wc):
        """Extract feature and add keyframe to index and database."""
        desc = self.extractor.extract(color_img_rgb)
        desc_np = np.expand_dims(desc, axis=0).astype(np.float32)
        
        self.faiss_index.add(desc_np)
        self.kf_list.append(kf_id)
        
        # Save downsampled images to minimize RAM storage
        small_color = cv2.resize(color_img_rgb, (320, 240))
        small_depth = cv2.resize(depth_aligned, (320, 240))
        self.db[kf_id] = (small_color, small_depth, T_wc.copy())

    def detect_loop(self, kf_id, color_img_rgb, depth_aligned, T_wc):
        """Query index for loop closure candidates and verify geometrically using ICP."""
        metrics_logger.log("loop_candidate_found", False)
        metrics_logger.log("loop_accepted", False)
        metrics_logger.log("loop_similarity", None)

        if len(self.kf_list) < self.min_id_diff:
            return None
            
        desc = self.extractor.extract(color_img_rgb)
        desc_np = np.expand_dims(desc, axis=0).astype(np.float32)
        
        # Search FAISS index
        k_search = min(10, len(self.kf_list))
        scores, indices = self.faiss_index.search(desc_np, k_search)
        
        scores = scores[0]
        indices = indices[0]
        
        for sim, idx in zip(scores, indices):
            if idx == -1:
                continue
                
            candidate_kf_id = self.kf_list[idx]
            
            # Check temporal distance constraint
            if kf_id - candidate_kf_id < self.min_id_diff:
                continue
                
            # Check similarity threshold
            if sim < self.sim_threshold:
                continue
                
            metrics_logger.log("loop_candidate_found", True)
            metrics_logger.log("loop_similarity", float(sim))
            
            # Run geometric verification using Open3D Point-to-Plane ICP
            print(f"LoopDetector: Candidate loop found between KF {kf_id} and KF {candidate_kf_id} (sim: {sim:.4f})")
            
            cand_color, cand_depth, cand_T_wc = self.db[candidate_kf_id]
            
            # Reconstruct Open3D point clouds at resized resolution (320x240)
            small_fx = self.fx * 0.5
            small_fy = self.fy * 0.5
            small_cx = self.cx * 0.5
            small_cy = self.cy * 0.5
            small_intrinsic = o3d.camera.PinholeCameraIntrinsic(320, 240, small_fx, small_fy, small_cx, small_cy)
            
            # Current frame downsampled for ICP speed
            curr_color_small = cv2.resize(color_img_rgb, (320, 240))
            curr_depth_small = cv2.resize(depth_aligned, (320, 240))
            
            # Create RGBD images
            cand_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(cand_color), o3d.geometry.Image(cand_depth),
                convert_rgb_to_intensity=True, depth_scale=1.0, depth_trunc=8.0
            )
            curr_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(curr_color_small), o3d.geometry.Image(curr_depth_small),
                convert_rgb_to_intensity=True, depth_scale=1.0, depth_trunc=8.0
            )
            
            cand_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(cand_rgbd, small_intrinsic)
            curr_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(curr_rgbd, small_intrinsic)
            
            curr_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            
            # Initial guess: relative pose based on current camera odometry
            T_init = np.linalg.inv(cand_T_wc) @ T_wc
            # relative pose mapping candidate (src) -> current (tgt)
            T_init_icp = np.linalg.inv(T_init)
            
            try:
                icp_res = o3d.pipelines.registration.registration_icp(
                    cand_pcd, curr_pcd, 0.08, T_init_icp,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
                
                # Check verification criteria
                if icp_res.fitness > 0.45 and icp_res.inlier_rmse < 0.035:
                    print(f"LoopDetector: Loop verified! ICP fitness: {icp_res.fitness:.4f}, RMSE: {icp_res.inlier_rmse:.4f}")
                    metrics_logger.log("loop_accepted", True)
                    # Return current -> candidate transform by inverting the src -> tgt result
                    return candidate_kf_id, np.linalg.inv(icp_res.transformation)
            except Exception as e:
                print(f"LoopDetector: Verification failed: {e}")
                
        return None