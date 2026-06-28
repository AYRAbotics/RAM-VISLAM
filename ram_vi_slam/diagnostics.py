import time
import csv
import json
import os
import threading
from datetime import datetime

class MetricsLogger:
    def __init__(self):
        self.enabled = False
        self.log_dir = "output"
        self.current_frame_metrics = {}
        self.history = []
        self.csv_path = None
        self.json_path = None
        self.lock = threading.Lock()
        self.frame_start_time = None
        
        # Comprehensive headers/keys for tracking
        self.headers = [
            "frame_id", "timestamp", "total_frame_time", "fps",
            "tracking_success", "rgbd_odom_error", "rgbd_odom_time",
            "icp_fitness", "icp_rmse", "icp_correspondences", "icp_time",
            "eskf_innovation_norm", "accel_var_x", "accel_var_y", "accel_var_z",
            "gyro_var_x", "gyro_var_y", "gyro_var_z", "imu_propagation_time",
            "valid_depth_pct", "num_features", "image_width", "image_height",
            "active_surfels", "spawned_surfels", "fused_surfels", "pruned_surfels", "mapping_time",
            "kf_inserted", "kf_id", "loop_candidate_found", "loop_accepted",
            "loop_similarity", "pgo_time",
            "mean_importance", "median_importance", "max_importance", "min_importance",
            "avg_observation_count", "avg_fusion_count", "avg_local_density",
            "cpu_usage", "gpu_memory_used_mb", "ram_usage_mb"
        ]

    def configure(self, enabled=False, log_dir="output"):
        with self.lock:
            self.enabled = enabled
            self.log_dir = log_dir
            if not enabled:
                return
                
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_path = os.path.join(log_dir, f"diagnostics_{timestamp}.csv")
            self.json_path = os.path.join(log_dir, f"diagnostics_{timestamp}.json")
            
            # Initialize CSV file with headers
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                
            self.history = []
            print(f"[Diagnostics] Logging configured. Output target: {self.csv_path}")

    def start_frame(self, frame_id, timestamp):
        if not self.enabled:
            return
        with self.lock:
            self.current_frame_metrics = {k: None for k in self.headers}
            self.current_frame_metrics["frame_id"] = frame_id
            self.current_frame_metrics["timestamp"] = timestamp
            self.frame_start_time = time.perf_counter()

    def log(self, key, value):
        if not self.enabled:
            return
        with self.lock:
            if key in self.current_frame_metrics:
                self.current_frame_metrics[key] = value

    def end_frame(self):
        if not self.enabled:
            return
        with self.lock:
            # End total time
            if self.frame_start_time is not None:
                total_time = time.perf_counter() - self.frame_start_time
                self.current_frame_metrics["total_frame_time"] = total_time
                self.current_frame_metrics["fps"] = 1.0 / total_time if total_time > 0 else 0.0
            
            # Record system performance (RAM, CPU)
            try:
                import psutil
                process = psutil.Process(os.getpid())
                self.current_frame_metrics["ram_usage_mb"] = process.memory_info().rss / (1024 * 1024)
                self.current_frame_metrics["cpu_usage"] = psutil.cpu_percent()
            except Exception:
                pass
                
            # GPU Usage (optional) via torch
            try:
                import torch
                if torch.cuda.is_available():
                    self.current_frame_metrics["gpu_memory_used_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                pass
                
            # Append to history and write row to CSV immediately to prevent losing data on crash
            self.history.append(self.current_frame_metrics.copy())
            
            row = [self.current_frame_metrics.get(h) for h in self.headers]
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

    def save_and_close(self):
        if not self.enabled:
            return
        with self.lock:
            # Write history to JSON as structured metadata
            try:
                with open(self.json_path, 'w') as f:
                    json.dump(self.history, f, indent=4)
                print(f"\n[Diagnostics] Diagnostics saved successfully to:\n - CSV: {self.csv_path}\n - JSON: {self.json_path}")
            except Exception as e:
                print(f"\n[Diagnostics] Error saving JSON diagnostics: {e}")

# Global singleton instance
metrics_logger = MetricsLogger()
