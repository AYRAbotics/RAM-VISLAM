import os
import numpy as np
import torch
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from ram_vi_slam.tracking import RGBDTracker

bag_path = "/home/rv/RAM_VI_SLAM/slam_benchmark_run1"

# Calibration constants
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

tracker = RGBDTracker(FX_C, FY_C, CX_C, CY_C)
K_d_mat = np.array([
    [FX_D, 0.0,  CX_D],
    [0.0,  FY_D, CY_D],
    [0.0,  0.0,  1.0 ]
])
tracker.set_calibration(K_d_mat, R_D2C, T_D2C)

reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
    rosbag2_py.ConverterOptions('cdr', 'cdr')
)
topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

while reader.has_next():
    topic, data, _ = reader.read_next()
    if topic == '/camera/camera/depth/image_rect_raw':
        msg = deserialize_message(data, get_message(topic_types[topic]))
        depth_np = np.frombuffer(msg.data, dtype=np.uint16).reshape((480, 640))
        
        # Run step by step to see where points are lost
        depth_t = torch.tensor(depth_np.astype(np.float32), device=tracker.device)
        depth_t = depth_t / 1000.0
        
        u_grid = torch.arange(640, dtype=torch.float32, device=tracker.device)
        v_grid = torch.arange(480, dtype=torch.float32, device=tracker.device)
        v_coords, u_coords = torch.meshgrid(v_grid, u_grid, indexing='ij')
        
        valid_depth = (depth_t > 0.1) & (depth_t < 10.0) # ignore saturated/far values
        z_d = depth_t[valid_depth]
        u_d = u_coords[valid_depth]
        v_d = v_coords[valid_depth]
        
        x_d = (u_d - CX_D) * z_d / FX_D
        y_d = (v_d - CY_D) * z_d / FY_D
        pts_d = torch.stack([x_d, y_d, z_d], dim=-1)
        
        pts_c = pts_d @ tracker.R_d2c_t.T + tracker.t_d2c_t
        x_c, y_c, z_c = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
        
        u_c = (x_c * FX_C) / z_c + CX_C
        v_c = (y_c * FY_C) / z_c + CY_C
        
        u_idx = torch.round(u_c).long()
        v_idx = torch.round(v_c).long()
        
        valid_bounds = (u_idx >= 0) & (u_idx < 640) & (v_idx >= 0) & (v_idx < 480)
        
        print("Detailed Step Count:")
        print("  Total valid depth pixels within 0.1m - 10m:", torch.sum(valid_depth).item())
        print("  Projected points within image boundaries:", torch.sum(valid_bounds).item())
        print("  Projected points outside boundaries:", torch.sum(~valid_bounds).item())
        if torch.sum(~valid_bounds).item() > 0:
            print("  Sample outside u:", u_idx[~valid_bounds][:10].cpu().numpy())
            print("  Sample outside v:", v_idx[~valid_bounds][:10].cpu().numpy())
        break
