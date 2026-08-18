#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog

def draw_centered_text(img, text, cx, cy, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.5, color=(30, 30, 30), thickness=4):
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    tx = cx - text_size[0] // 2
    ty = cy + text_size[1] // 2
    cv2.putText(img, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)

def main():
    # 1. Load startup image
    startup_path = '/home/rv/RAM_VI_SLAM/1.png'
    if not os.path.exists(startup_path):
        print(f"Error: Startup image not found at {startup_path}")
        sys.exit(1)
        
    img_orig = cv2.imread(startup_path)
    
    # Draw mode labels onto original image (centered inside the white pills)
    draw_centered_text(img_orig, "LIVE MODE", 400, 426)
    draw_centered_text(img_orig, "BAG MODE", 1483, 426)
    
    # Scale down by 50% for display
    img_base = cv2.resize(img_orig, (960, 540))
    
    selected_mode = [None] # Use list to modify inside callback
    hover_state = [None]
    advanced_mode = [False]
    
    def on_mouse(event, x, y, flags, param):
        # Coordinates scaled by 0.5:
        # Left pill: x in [97, 303], y in [186, 239]
        if 97 <= x <= 303 and 186 <= y <= 239:
            hover_state[0] = "live"
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_mode[0] = "live"
        # Right pill: x in [638, 844], y in [186, 239]
        elif 638 <= x <= 844 and 186 <= y <= 239:
            hover_state[0] = "bag"
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_mode[0] = "bag"
        # Advanced mode toggle pill: bottom center
        elif 380 <= x <= 580 and 480 <= y <= 520:
            hover_state[0] = "adv_toggle"
            if event == cv2.EVENT_LBUTTONDOWN:
                advanced_mode[0] = not advanced_mode[0]
        else:
            hover_state[0] = None
                
    window_name = "AYRAbotics - RAM-VI-S SLAM Launcher"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    
    print("Launcher: Waiting for mode selection in GUI...", flush=True)
    print("  - Click 'LIVE MODE' or press 'L' for Live Camera Mode")
    print("  - Click 'BAG MODE' or press 'B' for Bag Mode")
    print("  - Press 'A' or click the toggle at bottom for Advanced Mode (Edge Filter + Adaptive Radii)")
    
    while selected_mode[0] is None:
        img_draw = img_base.copy()
        
        # Draw Advanced Mode status pill at bottom center
        adv_color = (0, 200, 0) if advanced_mode[0] else (100, 100, 100)
        adv_text = "ADVANCED MODE: ON" if advanced_mode[0] else "ADVANCED MODE: OFF (Press 'A' to toggle)"
        cv2.rectangle(img_draw, (360, 485), (600, 525), (40, 40, 40), -1)
        cv2.rectangle(img_draw, (360, 485), (600, 525), adv_color, 2, cv2.LINE_AA)
        cv2.putText(img_draw, adv_text, (370, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.45, adv_color, 1, cv2.LINE_AA)
        
        if hover_state[0] == "live":
            cv2.rectangle(img_draw, (95, 184), (305, 241), (0, 220, 255), 3, cv2.LINE_AA)
        elif hover_state[0] == "bag":
            cv2.rectangle(img_draw, (636, 184), (846, 241), (0, 220, 255), 3, cv2.LINE_AA)
            
        cv2.imshow(window_name, img_draw)
        key = cv2.waitKey(30)
        
        if key in [ord('b'), ord('B')]:
            selected_mode[0] = "bag"
        elif key in [ord('l'), ord('L')]:
            selected_mode[0] = "live"
        elif key in [ord('a'), ord('A'), ord('e'), ord('E')]:
            advanced_mode[0] = not advanced_mode[0]
        elif key == 27 or key in [ord('q'), ord('Q')] or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Launcher: Closed by user.")
            cv2.destroyAllWindows()
            sys.exit(0)
            
    cv2.destroyWindow(window_name)
    
    # 2. Handle selected mode
    if selected_mode[0] == "bag":
        print("Launcher: BAG MODE selected. Opening folder dialog...", flush=True)
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        bag_path = filedialog.askdirectory(initialdir="/home/rv/RAM_VI_SLAM", title="Select ROS 2 Bag Folder")
        root.destroy()
        
        if not bag_path:
            print("Launcher: No bag folder selected. Exiting.")
            sys.exit(0)
            
        adv_str = " (with Advancements 1 & 2: Edge Filter & Adaptive Radii)" if advanced_mode[0] else " (Baseline Mode)"
        print(f"Launcher: Starting offline simulation on bag: {bag_path}{adv_str}", flush=True)
        
        cmd = [sys.executable, "-m", "ram_vi_slam.offline_runner", "--bag_path", bag_path, "--diagnostics", "--visualize"]
        if advanced_mode[0]:
            cmd.extend(["--enable_depth_filter", "--adaptive_radii"])
            
        env = os.environ.copy()
        env["PYTHONPATH"] = f"/home/rv/RAM_VI_SLAM:{env.get('PYTHONPATH', '')}"
        subprocess.run(cmd, cwd="/home/rv/RAM_VI_SLAM", env=env)
        
    elif selected_mode[0] == "live":
        print("Launcher: LIVE MODE selected. Waiting for camera topics to start...", flush=True)
        cmd = [sys.executable, "-m", "ram_vi_slam.slam_node"]
        if advanced_mode[0]:
            cmd.extend(["--ros-args", "-p", "enable_depth_filter:=true", "-p", "adaptive_radii:=true"])
            
        env = os.environ.copy()
        env["PYTHONPATH"] = f"/home/rv/RAM_VI_SLAM:{env.get('PYTHONPATH', '')}"
        subprocess.run(cmd, cwd="/home/rv/RAM_VI_SLAM", env=env)
        
    # 3. Automatically render multi-view 3D point-cloud evaluation
    ply_path = '/home/rv/RAM_VI_SLAM/output/surfel_map.ply'
    eval_img_path = '/home/rv/RAM_VI_SLAM/output/surfel_reconstruction_comparison.png'
    
    if os.path.exists(ply_path):
        print(f"\nLauncher: Rendering multi-view 3D Surfel Point Cloud from {ply_path}...", flush=True)
        try:
            render_script = '/home/rv/RAM_VI_SLAM/scratch/render_surfel_eval.py'
            if os.path.exists(render_script):
                subprocess.run([sys.executable, render_script], cwd="/home/rv/RAM_VI_SLAM")
        except Exception as e:
            print(f"Launcher: Warning - could not render evaluation views: {e}", flush=True)

    # 4. Display Reconstructed Result Window & Finish Screen
    if os.path.exists(eval_img_path):
        img_eval = cv2.imread(eval_img_path)
        if img_eval is not None:
            eval_window = "AYRAbotics - Reconstructed 3D Surfel Point Cloud"
            cv2.namedWindow(eval_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(eval_window, 1280, 720)
            cv2.imshow(eval_window, img_eval)
            
    exit_path = '/home/rv/RAM_VI_SLAM/2.png'
    if os.path.exists(exit_path):
        img_exit = cv2.imread(exit_path)
        img_exit_display = cv2.resize(img_exit, (960, 540))
        
        finish_window = "AYRAbotics - Simulation Finished"
        cv2.namedWindow(finish_window)
        
        closed = [False]
        def on_mouse_exit(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                closed[0] = True
        cv2.setMouseCallback(finish_window, on_mouse_exit)
        
        print("\nLauncher: Simulation ended. Reconstructed 3D point cloud is displayed.", flush=True)
        print("  - Press 'V' to open the Interactive 3D Orbit Viewer", flush=True)
        print("  - Click or press any other key / ESC to Exit\n", flush=True)
        
        while not closed[0]:
            cv2.imshow(finish_window, img_exit_display)
            key = cv2.waitKey(30)
            
            # Press 'V' to launch Open3D interactive viewer
            if key in [ord('v'), ord('V')]:
                if os.path.exists(ply_path):
                    print("Launcher: Launching interactive Open3D viewer (Press 'Q' inside viewer to return)...", flush=True)
                    try:
                        import open3d as o3d
                        pcd = o3d.io.read_point_cloud(ply_path)
                        o3d.visualization.draw_geometries([pcd], window_name="RAM-VI-S 3D Surfel Map (Interactive)")
                    except Exception as e:
                        print(f"Viewer error: {e}", flush=True)
            elif key != -1 or cv2.getWindowProperty(finish_window, cv2.WND_PROP_VISIBLE) < 1:
                break
                
        cv2.destroyAllWindows()
        print("Launcher: Exiting launcher. Thank you!", flush=True)

if __name__ == '__main__':
    main()
