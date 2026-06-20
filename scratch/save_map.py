import sys
import os

def save_running_map():
    print("Inject: Looking for surfel_map in running frames...", flush=True)
    found = False
    for tid, frame in sys._current_frames().items():
        curr = frame
        while curr:
            if curr.f_code.co_name == 'main':
                locals_dict = curr.f_locals
                if 'surfel_map' in locals_dict:
                    surfel_map = locals_dict['surfel_map']
                    frame_count = locals_dict.get('frame_count', 0)
                    args = locals_dict.get('args', None)
                    save_path = args.save_map if args else "/home/rv/RAM_VI_SLAM/output/surfel_map.ply"
                    
                    print(f"Inject: Found surfel_map with {surfel_map.active_n} active surfels.", flush=True)
                    print("Inject: Pruning unstable surfels...", flush=True)
                    surfel_map.prune_unstable(frame_count, min_weight=3.0)
                    print("Inject: Merging voxels...", flush=True)
                    surfel_map.merge_voxels(voxel_size=0.01)
                    print(f"Inject: Exporting PLY to {save_path}...", flush=True)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    surfel_map.export_ply(save_path)
                    print("Inject: Successfully saved surfel map to file!", flush=True)
                    found = True
                    break
            curr = curr.f_back
        if found:
            break
    if not found:
        print("Inject: ERROR: surfel_map not found in frames!", flush=True)

try:
    save_running_map()
except Exception as e:
    print(f"Inject: Exception occurred during map save: {e}", flush=True)
