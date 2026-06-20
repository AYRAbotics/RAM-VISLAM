import rosbag2_py, numpy as np, cv2
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from ram_vi_slam.tracking import RGBDTracker

FX_C = 610.1809; FY_C = 610.2639; CX_C = 337.16; CY_C = 249.06
FX_D = 388.246; FY_D = 388.246; CX_D = 313.195; CY_D = 243.978
R_D2C = np.array([[ 0.99999636, -0.00241311,  0.00117634],
                  [ 0.00241741,  0.99999034, -0.00367048],
                  [-0.00116747,  0.00367331,  0.99999255]])
T_D2C = np.array([0.01454706, 0.00018594, 0.00039981])

tracker = RGBDTracker(FX_C, FY_C, CX_C, CY_C)
K_d = np.array([[FX_D, 0, CX_D], [0, FY_D, CY_D], [0, 0, 1]])
tracker.set_calibration(K_d, R_D2C, T_D2C)

reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri='/home/rv/RAM_VI_SLAM/slam_benchmark_run1', storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
tps = {t.name: t.type for t in reader.get_all_topics_and_types()}

prev_color = None; prev_depth = None; lc_t=0; ld_t=0
fc = 0
sum_t = np.zeros(3)

while reader.has_next() and fc < 50:
    top, dat, t = reader.read_next()
    if top == '/camera/camera/color/image_raw':
        m = deserialize_message(dat, get_message(tps[top]))
        c = np.frombuffer(m.data, dtype=np.uint8).reshape((m.height, m.width, 3))
        lc = cv2.cvtColor(c, cv2.COLOR_BGR2RGB); lc_t = t
    elif top == '/camera/camera/depth/image_rect_raw':
        m = deserialize_message(dat, get_message(tps[top]))
        ld = np.frombuffer(m.data, dtype=np.uint16).reshape((m.height, m.width))
        ld_t = t
    
    if 'lc' in locals() and 'ld' in locals() and lc is not None and ld is not None:
        if abs(lc_t - ld_t) * 1e-6 < 30.0:
            dm = tracker.register_depth(ld)
            if fc > 0:
                s, T = tracker.align_frames(prev_color, prev_depth, lc, dm, np.eye(4))
                if s:
                    sum_t += T[0:3, 3]
                    print(f"F{fc} T_rel trans:", T[0:3, 3])
            prev_color = lc.copy(); prev_depth = dm.copy()
            fc += 1
            lc = None; ld = None
print("Total relative translation (Camera frame):", sum_t)
