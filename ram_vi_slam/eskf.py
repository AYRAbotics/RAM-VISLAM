import numpy as np
from scipy.spatial.transform import Rotation

def skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])

class ESKF:
    def __init__(self, 
                 accel_noise=0.08,      # m/s^2/sqrt(Hz)
                 gyro_noise=0.005,      # rad/s/sqrt(Hz)
                 accel_bias_noise=0.001, # m/s^3/sqrt(Hz)
                 gyro_bias_noise=0.0001, # rad/s^2/sqrt(Hz)
                 meas_pos_noise=0.01,   # m
                 meas_rot_noise=0.015):  # rad
        
        # Nominal state: position, velocity, orientation, biases
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.R = Rotation.identity()
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        
        # 15x15 State covariance matrix [p, v, theta, ba, bg]
        self.P = np.eye(15) * 1e-4
        
        # Noise parameters
        self.sigma_a = accel_noise
        self.sigma_g = gyro_noise
        self.sigma_ba = accel_bias_noise
        self.sigma_bg = gyro_bias_noise
        
        self.meas_pos_noise = meas_pos_noise
        self.meas_rot_noise = meas_rot_noise
        
        # Gravity vector in world frame
        self.g_w = np.array([0.0, 0.0, 9.80665])
        self.is_gravity_initialized = False
        self.init_accel_buffer = []
        
        # Last update time
        self.last_t = None

    def initialize_gravity(self, init_accel):
        """Initialize gravity vector from initial accelerometer readings."""
        mean_accel = np.mean(init_accel, axis=0)
        mag = np.linalg.norm(mean_accel)
        if mag < 8.0 or mag > 11.5:
            # Shake detected, retry initialization
            self.init_accel_buffer = []
            return False
            
        self.g_w = np.array([0.0, 0.0, -mag])
        
        # Initialize orientation from gravity vector
        ax, ay, az = mean_accel
        roll = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        self.R = Rotation.from_euler('xyz', [roll, pitch, 0.0])
        
        self.is_gravity_initialized = True
        print(f"ESKF: Gravity initialized. Magnitude: {mag:.4f} m/s^2. Initial RPY: {np.degrees([roll, pitch, 0.0])}")
        return True

    def predict(self, dt, accel, gyro):
        """Propagate state and covariance using IMU readings."""
        if not self.is_gravity_initialized:
            # Buffer acceleration to initialize gravity
            self.init_accel_buffer.append(accel)
            if len(self.init_accel_buffer) >= 50:
                self.initialize_gravity(self.init_accel_buffer)
            return

        # Cap dt to avoid integration spikes under large time gaps
        dt = min(dt, 0.05)

        # Correct IMU measurements with current bias estimates
        a_corr = accel - self.ba
        w_corr = gyro - self.bg
        
        # 1. State Propagation (Nominal State)
        R_curr = self.R.as_matrix()
        
        # Propagate position and velocity
        self.p += self.v * dt + 0.5 * (R_curr @ a_corr + self.g_w) * (dt ** 2)
        # Apply velocity damping to prevent runaway velocity from noisy accelerometer
        self.v = self.v * np.exp(-2.0 * dt) + (R_curr @ a_corr + self.g_w) * dt
        
        # Propagate orientation
        rot_step = Rotation.from_rotvec(w_corr * dt)
        self.R = self.R * rot_step
        
        # 2. Covariance Propagation
        # State Transition Matrix Fx
        Fx = np.eye(15)
        Fx[0:3, 3:6] = np.eye(3) * dt
        Fx[3:6, 6:9] = -R_curr @ skew(a_corr) * dt
        Fx[3:6, 9:12] = -R_curr * dt
        Fx[6:9, 6:9] = rot_step.as_matrix().T
        Fx[6:9, 12:15] = -np.eye(3) * dt
        
        # Process noise covariance Q
        Q = np.zeros((15, 15))
        Q[0:3, 0:3] = np.eye(3) * (self.sigma_a ** 2) * (dt ** 2)
        Q[3:6, 3:6] = np.eye(3) * (self.sigma_a ** 2) * dt
        Q[6:9, 6:9] = np.eye(3) * (self.sigma_g ** 2) * dt
        Q[9:12, 9:12] = np.eye(3) * (self.sigma_ba ** 2) * dt
        Q[12:15, 12:15] = np.eye(3) * (self.sigma_bg ** 2) * dt
        
        self.P = Fx @ self.P @ Fx.T + Q

    def update(self, p_meas, R_meas, pos_noise_scale=1.0, rot_noise_scale=1.0):
        """Correct state using camera tracking update (pose of IMU in world)."""
        if not self.is_gravity_initialized:
            return

        # Residuals
        r_p = p_meas - self.p
        r_theta = (self.R.inv() * Rotation.from_matrix(R_meas)).as_rotvec()
        y = np.concatenate([r_p, r_theta])
        
        # Measurement matrix H
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)
        
        # Measurement noise covariance V
        V = np.eye(6)
        V[0:3, 0:3] = np.eye(3) * ((self.meas_pos_noise * pos_noise_scale) ** 2)
        V[3:6, 3:6] = np.eye(3) * ((self.meas_rot_noise * rot_noise_scale) ** 2)
        
        # Kalman gain
        S = H @ self.P @ H.T + V
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Correct state
        dx = K @ y
        
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.R = self.R * Rotation.from_rotvec(dx[6:9])
        self.ba += dx[9:12]
        self.bg += dx[12:15]
        
        # Update covariance using Joseph form for numerical stability
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ V @ K.T