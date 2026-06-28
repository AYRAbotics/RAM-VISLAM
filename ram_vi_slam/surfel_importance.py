import numpy as np
import torch

def compute_importance(surfel, current_frame_id=None):
    """
    Computes the importance score of a surfel.
    Supports both a single surfel (as a dict or object) and a batch of surfels (as a SurfelMap or dict of tensors).
    """
    # Check if we are dealing with PyTorch tensors (vectorized mode)
    is_tensor = False
    if hasattr(surfel, 'positions') or (isinstance(surfel, dict) and any(isinstance(v, torch.Tensor) for v in surfel.values())):
        is_tensor = True
        
    if is_tensor:
        return _compute_importance_tensor(surfel, current_frame_id)
    else:
        return _compute_importance_scalar(surfel, current_frame_id)

def _compute_importance_scalar(surfel, current_frame_id):
    # Retrieve properties from single surfel dict or object
    def get_val(obj, key, default=0.0):
        if isinstance(obj, dict):
            return obj.get(key, default)
        else:
            return getattr(obj, key, default)

    obs_count = float(get_val(surfel, 'observation_count', 0.0))
    fus_count = float(get_val(surfel, 'fusion_count', 0.0))
    conf_score = float(get_val(surfel, 'confidence_score', 0.0))
    icp_fit = float(get_val(surfel, 'average_icp_fitness', 0.0))
    depth_conf = float(get_val(surfel, 'average_depth_confidence', 0.0))
    pos_var = float(get_val(surfel, 'position_variance', 0.0))
    nor_var = float(get_val(surfel, 'normal_variance', 0.0))
    view_ang = float(get_val(surfel, 'average_viewing_angle', 0.0))
    last_obs = int(get_val(surfel, 'last_observed_frame', 0))
    
    # 1. Observation Score
    obs_score = 1.0 - np.exp(-0.05 * obs_count)
    
    # 2. Fusion Score
    fus_score = 1.0 - np.exp(-0.1 * fus_count)
    
    # 3. Confidence Score: conf_score (already in [0, 1])
    
    # 4. ICP Quality: icp_fit (already in [0, 1])
    
    # 5. Depth Confidence: depth_conf (already in [0, 1])
    
    # 6. Surface Stability: inverse mapping of position & normal variance
    stability = np.exp(-1000.0 * pos_var - 10.0 * nor_var)
    
    # 7. Viewing Diversity: combines observation count with viewing angle
    view_div = min(1.0, view_ang / 1.5708) * (1.0 - np.exp(-0.05 * obs_count))
    
    # Weighted sum
    raw_score = (0.30 * obs_score + 
                 0.20 * fus_score + 
                 0.15 * conf_score + 
                 0.10 * icp_fit + 
                 0.10 * depth_conf + 
                 0.10 * stability + 
                 0.05 * view_div)
                 
    # Penalties
    penalty = 0.0
    if current_frame_id is not None:
        idle_time = max(0, current_frame_id - last_obs)
        penalty += 0.002 * max(0.0, idle_time - 50.0)
        
    penalty += 5.0 * pos_var
    penalty += 1.0 * nor_var
    
    importance = np.clip(raw_score - penalty, 0.0, 1.0)
    return float(importance)

def _compute_importance_tensor(surfel_map, current_frame_id):
    if hasattr(surfel_map, 'positions'):
        # It's a SurfelMap instance
        obs_count = surfel_map.observation_count[:surfel_map.active_n]
        fus_count = surfel_map.fusion_count[:surfel_map.active_n]
        conf_score = surfel_map.confidence_score[:surfel_map.active_n]
        icp_fit = surfel_map.average_icp_fitness[:surfel_map.active_n]
        depth_conf = surfel_map.average_depth_confidence[:surfel_map.active_n]
        pos_var = surfel_map.position_variance[:surfel_map.active_n]
        nor_var = surfel_map.normal_variance[:surfel_map.active_n]
        view_ang = surfel_map.average_viewing_angle[:surfel_map.active_n]
        last_obs = surfel_map.last_observed_frame[:surfel_map.active_n]
    else:
        # It's a dictionary of tensors
        obs_count = surfel_map['observation_count']
        fus_count = surfel_map['fusion_count']
        conf_score = surfel_map['confidence_score']
        icp_fit = surfel_map['average_icp_fitness']
        depth_conf = surfel_map['average_depth_confidence']
        pos_var = surfel_map['position_variance']
        nor_var = surfel_map['normal_variance']
        view_ang = surfel_map['average_viewing_angle']
        last_obs = surfel_map['last_observed_frame']
        
    # Vectorized calculations in PyTorch
    obs_score = 1.0 - torch.exp(-0.05 * obs_count)
    fus_score = 1.0 - torch.exp(-0.1 * fus_count)
    
    stability = torch.exp(-1000.0 * pos_var - 10.0 * nor_var)
    view_div = torch.clamp(view_ang / 1.5708, max=1.0) * (1.0 - torch.exp(-0.05 * obs_count))
    
    raw_score = (0.30 * obs_score + 
                 0.20 * fus_score + 
                 0.15 * conf_score + 
                 0.10 * icp_fit + 
                 0.10 * depth_conf + 
                 0.10 * stability + 
                 0.05 * view_div)
                 
    penalty = torch.zeros_like(raw_score)
    if current_frame_id is not None:
        idle_time = torch.clamp(current_frame_id - last_obs, min=0)
        penalty += 0.002 * torch.clamp(idle_time - 50.0, min=0.0)
        
    penalty += 5.0 * pos_var
    penalty += 1.0 * nor_var
    
    importance = torch.clamp(raw_score - penalty, 0.0, 1.0)
    return importance
