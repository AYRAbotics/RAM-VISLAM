"""
RGB-Guided Depth Edge Refinement Module (Advancement 1)
Joint Bilateral Filtering and Flying Pixel Suppression using PyTorch GPU tensors.
"""

import torch
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JointBilateralDepthFilter:
    """
    GPU-accelerated Joint Bilateral Depth Filter.
    Uses RGB color gradients to guide depth edge sharpening and suppress depth dilation
    around thin structures (chair legs, pillars, window frames, table edges).
    """
    def __init__(self, radius=2, sigma_spatial=3.0, sigma_color=0.15, max_depth_diff=0.08, device=DEVICE):
        self.radius = radius
        self.sigma_spatial = sigma_spatial
        self.sigma_color = sigma_color
        self.max_depth_diff = max_depth_diff
        self.device = device
        
        # Precompute spatial Gaussian weights
        shifts = []
        spatial_weights = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shifts.append((dy, dx))
                dist_sq = dy * dy + dx * dx
                w_s = np.exp(-dist_sq / (2.0 * sigma_spatial * sigma_spatial))
                spatial_weights.append(w_s)
                
        self.shifts = shifts
        self.spatial_weights = torch.tensor(spatial_weights, dtype=torch.float32, device=self.device).view(-1, 1, 1)

    def filter(self, depth_t: torch.Tensor, color_t: torch.Tensor) -> torch.Tensor:
        """
        Apply Joint Bilateral Depth Filtering.
        Args:
            depth_t: (H, W) float32 tensor on GPU (depth in meters).
            color_t: (H, W, 3) float32 tensor on GPU (RGB normalized [0, 1]).
        Returns:
            filtered_depth: (H, W) float32 tensor on GPU.
        """
        H, W = depth_t.shape
        r = self.radius

        # Pad depth and color tensors
        depth_pad = F.pad(depth_t.unsqueeze(0).unsqueeze(0), (r, r, r, r), mode='replicate').squeeze(0).squeeze(0)
        color_pad = F.pad(color_t.permute(2, 0, 1).unsqueeze(0), (r, r, r, r), mode='replicate').squeeze(0).permute(1, 2, 0)

        # Center references
        center_color = color_pad[r:H+r, r:W+r]  # (H, W, 3)
        center_depth = depth_pad[r:H+r, r:W+r]  # (H, W)

        # Vectorized accumulation across the window
        weight_sum = torch.zeros((H, W), dtype=torch.float32, device=self.device)
        depth_sum = torch.zeros((H, W), dtype=torch.float32, device=self.device)

        # Thresholds
        two_sigma_color_sq = 2.0 * (self.sigma_color ** 2)

        for i, (dy, dx) in enumerate(self.shifts):
            w_spatial = self.spatial_weights[i]
            
            # Neighbor slices
            y_start = r + dy
            x_start = r + dx
            neighbor_depth = depth_pad[y_start:y_start+H, x_start:x_start+W]
            neighbor_color = color_pad[y_start:y_start+H, x_start:x_start+W]

            # Color difference (photometric weight)
            color_diff_sq = torch.sum((center_color - neighbor_color) ** 2, dim=-1)
            w_color = torch.exp(-color_diff_sq / two_sigma_color_sq)

            # Depth difference gating: do not blend across true depth discontinuities (> 12cm)
            depth_diff = torch.abs(center_depth - neighbor_depth)
            valid_neighbor = (neighbor_depth > 0.1) & (depth_diff < 0.12)

            # Combined weight
            w = w_spatial * w_color * valid_neighbor.float()

            weight_sum += w
            depth_sum += w * neighbor_depth

        # Normalize filtered depth where valid bilateral support exists, otherwise preserve original depth
        has_support = weight_sum > 0.15
        filtered_depth = torch.where(has_support, depth_sum / torch.clamp(weight_sum, min=1e-4), center_depth)

        return filtered_depth


# Module-level cached filter instance for high-speed execution
_cached_filter = None

def refine_depth_with_rgb_guidance(depth_np: np.ndarray, color_np: np.ndarray, device=DEVICE) -> np.ndarray:
    """
    Convenience functional interface for numpy arrays.
    Args:
        depth_np: (H, W) float32 in meters.
        color_np: (H, W, 3) uint8 or float32.
    Returns:
        refined_depth_np: (H, W) float32 in meters.
    """
    global _cached_filter
    if _cached_filter is None:
        _cached_filter = JointBilateralDepthFilter(device=device)

    # Convert to GPU tensors
    depth_t = torch.tensor(depth_np, dtype=torch.float32, device=device)
    if color_np.dtype == np.uint8:
        color_t = torch.tensor(color_np.astype(np.float32) / 255.0, dtype=torch.float32, device=device)
    else:
        color_t = torch.tensor(color_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        filtered_t = _cached_filter.filter(depth_t, color_t)

    return filtered_t.cpu().numpy()
