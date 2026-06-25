import numpy as np
from scipy.spatial.transform import Rotation
import imufusion


class ComplementaryFilterEstimator:
    def __init__(self, alpha=0.98, gravity_mag=9.80665):
        self.alpha = alpha
        self.g = gravity_mag
        
        # State: position, velocity, orientation (Rotation)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.R = Rotation.identity()
        
        # Gravity direction initialization
        self.is_gravity_initialized = False
        self.init_accel_buffer = []
        self.g_w = np.array([0.0, 0.0, -self.g])
        
        # Keep track of Euler angles for complementary filter (roll, pitch, yaw)
        self.rpy = np.zeros(3)

    def initialize_gravity(self, init_accel):
        mean_accel = np.mean(init_accel, axis=0)
        mag = np.linalg.norm(mean_accel)
        if mag < 8.0 or mag > 11.5:
            self.init_accel_buffer = []
            return False
            
        self.g_w = np.array([0.0, 0.0, -mag])
        
        # Initialize orientation from gravity vector
        ax, ay, az = mean_accel
        roll = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        self.rpy = np.array([roll, pitch, 0.0])
        self.R = Rotation.from_euler('xyz', self.rpy)
        
        self.is_gravity_initialized = True
        print(f"ComplementaryFilter: Gravity initialized to {self.g_w}. Initial RPY: {np.degrees(self.rpy)}")
        return True

    def predict(self, dt, accel, gyro):
        if not self.is_gravity_initialized:
            self.init_accel_buffer.append(accel)
            if len(self.init_accel_buffer) >= 50:
                self.initialize_gravity(self.init_accel_buffer)
            return

        # Cap dt to avoid integration spikes under large time gaps
        dt = min(dt, 0.05)

        # 1. Orientation update using Complementary Filter
        ax, ay, az = accel
        acc_roll = np.arctan2(ay, az)
        acc_pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        
        # Gyro integration step
        gyro_roll_rate = gyro[0]
        gyro_pitch_rate = gyro[1]
        gyro_yaw_rate = gyro[2]
        
        # Complementary filter formulas
        new_roll = self.alpha * (self.rpy[0] + gyro_roll_rate * dt) + (1.0 - self.alpha) * acc_roll
        new_pitch = self.alpha * (self.rpy[1] + gyro_pitch_rate * dt) + (1.0 - self.alpha) * acc_pitch
        new_yaw = self.rpy[2] + gyro_yaw_rate * dt  # Yaw cannot be corrected with accelerometer
        
        self.rpy = np.array([new_roll, new_pitch, new_yaw])
        self.R = Rotation.from_euler('xyz', self.rpy)
        
        # 2. Position and Velocity propagation using double integration
        R_curr = self.R.as_matrix()
        # acceleration in world = R_body_to_world * acceleration_body + gravity_world
        # raw accel measurements include gravity counterforce (upward proper acceleration), so:
        a_w = R_curr @ accel + self.g_w
        
        self.p += self.v * dt + 0.5 * a_w * (dt ** 2)
        self.v += a_w * dt

    def update(self, p_meas, R_meas, *args, **kwargs):
        """Correct pose using tracking updates."""
        if not self.is_gravity_initialized:
            return
            
        # Simple proportional update to state
        self.p = 0.8 * self.p + 0.2 * p_meas
        self.R = Rotation.from_matrix(R_meas)
        self.rpy = self.R.as_euler('xyz')
        
        # Reset velocity to be consistent with position change
        # (A simple first-order velocity approximation)
        self.v = np.zeros(3)


class ConstantVelocityEstimator:
    def __init__(self):
        self.p = np.zeros(3)
        self.R = Rotation.identity()
        self.v = np.zeros(3)       # Linear velocity
        self.omega = np.zeros(3)   # Angular velocity
        
        # Dummy variables to match ESKF interface
        self.is_gravity_initialized = True

    def predict(self, dt, accel=None, gyro=None):
        """Propagate state using estimated constant velocity."""
        self.p += self.v * dt
        rot_step = Rotation.from_rotvec(self.omega * dt)
        self.R = self.R * rot_step

    def update(self, p_meas, R_meas, *args, **kwargs):
        """Update state and recalculate velocity estimates based on observation delta."""
        # Note: We don't have dt here, but we can compute instantaneous velocity on next step.
        # Or estimate it using a moving average
        dp = p_meas - self.p
        self.p = p_meas.copy()
        
        R_diff = self.R.inv() * Rotation.from_matrix(R_meas)
        self.R = Rotation.from_matrix(R_meas.copy())
        
        # Estimate velocities (assuming standard frame rate ~0.033s for scaling, 
        # actual velocity is refined when tracking feeds delta time).
        # We can cap velocity to prevent extreme spikes on tracking glitches
        dt_est = 0.033
        inst_v = dp / dt_est
        inst_omega = R_diff.as_rotvec() / dt_est
        
        self.v = 0.7 * self.v + 0.3 * inst_v
        self.omega = 0.7 * self.omega + 0.3 * inst_omega


class GyroGuidedEstimator:
    def __init__(self):
        # State: position, orientation (Rotation), and velocity
        self.p = np.zeros(3)
        self.R = Rotation.identity()
        self.v = np.zeros(3)
        
        # Interface compatibility flag
        self.is_gravity_initialized = True

    def predict(self, dt, accel=None, gyro=None):
        """Propagate state using velocity for translation and gyroscope for rotation."""
        # Propagate position
        self.p += self.v * dt
        
        # Viscous velocity damping to avoid translation drift/overshoot on small motions
        self.v = self.v * np.exp(-1.5 * dt)
        
        # Propagate orientation directly using high-rate gyroscope data
        if gyro is not None:
            rot_step = Rotation.from_rotvec(gyro * dt)
            self.R = self.R * rot_step

    def update(self, p_meas, R_meas, *args, **kwargs):
        """Correct pose using visual tracking measurements and update velocity estimates."""
        p_prev = self.p.copy()
        
        # Visual odometry drives the position and orientation directly
        self.p = p_meas.copy()
        self.R = Rotation.from_matrix(R_meas)
        
        # Estimate linear velocity from position delta (average frame time ~0.033s if dt not provided)
        dt_est = kwargs.get('dt', 0.033)
        if dt_est <= 0:
            dt_est = 0.033
            
        dp = self.p - p_prev
        inst_v = dp / dt_est
        
        # Cap instantaneous velocity to 1.5 m/s to filter out visual tracking glitches
        inst_v_norm = np.linalg.norm(inst_v)
        if inst_v_norm > 1.5:
            inst_v = (inst_v / inst_v_norm) * 1.5
            
        self.v = 0.6 * self.v + 0.4 * inst_v


class IMUFusionEstimator:
    def __init__(self):
        # State: position, orientation (Rotation), and velocity
        self.p = np.zeros(3)
        self.R = Rotation.identity()
        self.v = np.zeros(3)
        
        # Initialize imufusion AHRS
        self.ahrs = imufusion.Ahrs()
        
        # Interface compatibility flag
        self.is_gravity_initialized = True

    def predict(self, dt, accel=None, gyro=None):
        """Propagate state using velocity for translation and imufusion for orientation."""
        # 1. Propagate position
        self.p += self.v * dt
        
        # Viscous velocity damping to avoid translation drift/overshoot on small motions
        self.v = self.v * np.exp(-1.5 * dt)
        
        # 2. Propagate orientation using imufusion
        if gyro is not None and accel is not None:
            # Convert gyro from rad/s to degrees/s
            gyro_deg = gyro * (180.0 / np.pi)
            # Convert accel from m/s^2 to g
            accel_g = accel / 9.80665
            
            # Update AHRS
            self.ahrs.update_no_magnetometer(gyro_deg, accel_g, dt)
            
            # Retrieve quaternion in [w, x, y, z] format and convert to scipy Rotation [x, y, z, w]
            w, x, y, z = self.ahrs.quaternion.wxyz
            self.R = Rotation.from_quat([x, y, z, w])

    def update(self, p_meas, R_meas, *args, **kwargs):
        """Correct pose using visual tracking measurements and update velocity estimates."""
        p_prev = self.p.copy()
        
        # Visual odometry drives the position and orientation directly
        self.p = p_meas.copy()
        self.R = Rotation.from_matrix(R_meas)
        
        # Estimate linear velocity from position delta (average frame time ~0.033s if dt not provided)
        dt_est = kwargs.get('dt', 0.033)
        if dt_est <= 0:
            dt_est = 0.033
            
        dp = self.p - p_prev
        inst_v = dp / dt_est
        
        # Cap instantaneous velocity to 1.5 m/s to filter out visual tracking glitches
        inst_v_norm = np.linalg.norm(inst_v)
        if inst_v_norm > 1.5:
            inst_v = (inst_v / inst_v_norm) * 1.5
            
        self.v = 0.6 * self.v + 0.4 * inst_v

