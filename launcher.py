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
        
    img = cv2.imread(startup_path)
    
    # Draw mode labels onto original image (centered inside the white pills)
    draw_centered_text(img, "LIVE MODE", 400, 426)
    draw_centered_text(img, "BAG MODE", 1483, 426)
    
    # Scale down by 50% for display
    scale = 0.5
    img_display = cv2.resize(img, (960, 540))
    
    selected_mode = [None] # Use list to modify inside callback
    
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Coordinates scaled by 0.5:
            # Left pill: x in [97, 303], y in [186, 239]
            if 97 <= x <= 303 and 186 <= y <= 239:
                selected_mode[0] = "live"
            # Right pill: x in [638, 844], y in [186, 239]
            elif 638 <= x <= 844 and 186 <= y <= 239:
                selected_mode[0] = "bag"
                
    window_name = "AYRAbotics - RAM-VI SLAM Launcher"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    
    print("Launcher: Waiting for mode selection in GUI...", flush=True)
    while selected_mode[0] is None:
        cv2.imshow(window_name, img_display)
        # Handle GUI events and check if window was closed
        key = cv2.waitKey(30)
        if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Launcher: Closed by user.")
            cv2.destroyAllWindows()
            sys.exit(0)
            
    cv2.destroyWindow(window_name)
    
    # 2. Handle selected mode
    if selected_mode[0] == "bag":
        print("Launcher: BAG MODE selected. Opening folder dialog...", flush=True)
        # Tkinter folder selector
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True) # Bring dialog to front
        bag_path = filedialog.askdirectory(initialdir="/home/rv/RAM_VI_SLAM", title="Select ROS 2 Bag Folder")
        root.destroy()
        
        if not bag_path:
            print("Launcher: No bag folder selected. Exiting.")
            sys.exit(0)
            
        print(f"Launcher: Starting offline simulation on bag: {bag_path}", flush=True)
        cmd = [sys.executable, "-m", "ram_vi_slam.offline_runner", "--bag_path", bag_path, "--diagnostics", "--visualize"]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"/home/rv/RAM_VI_SLAM:{env.get('PYTHONPATH', '')}"
        subprocess.run(cmd, cwd="/home/rv/RAM_VI_SLAM", env=env)
        
    elif selected_mode[0] == "live":
        print("Launcher: LIVE MODE selected. Waiting for camera topics to start...", flush=True)
        cmd = [sys.executable, "-m", "ram_vi_slam.slam_node"]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"/home/rv/RAM_VI_SLAM:{env.get('PYTHONPATH', '')}"
        subprocess.run(cmd, cwd="/home/rv/RAM_VI_SLAM", env=env)
        
    # 3. Load exit/finish image
    exit_path = '/home/rv/RAM_VI_SLAM/2.png'
    if os.path.exists(exit_path):
        img_exit = cv2.imread(exit_path)
        img_exit_display = cv2.resize(img_exit, (960, 540))
        
        finish_window = "AYRAbotics - Simulation Finished"
        cv2.namedWindow(finish_window)
        
        # Simple mouse callback to close on click
        closed = [False]
        def on_mouse_exit(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                closed[0] = True
        cv2.setMouseCallback(finish_window, on_mouse_exit)
        
        print("Launcher: Simulation ended. Displaying finish screen. Click or press any key to exit...", flush=True)
        while not closed[0]:
            cv2.imshow(finish_window, img_exit_display)
            key = cv2.waitKey(30)
            # Break if any key is pressed or window is closed or mouse is clicked
            if key != -1 or cv2.getWindowProperty(finish_window, cv2.WND_PROP_VISIBLE) < 1:
                break
                
        cv2.destroyAllWindows()
        print("Launcher: Exiting launcher. Thank you!", flush=True)

if __name__ == '__main__':
    main()
