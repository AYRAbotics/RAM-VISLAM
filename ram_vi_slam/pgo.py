import numpy as np
import gtsam

class PoseGraphOptimizer:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.poses = {}  # kf_id -> 4x4 numpy pose T_wc
        
        # Noise models
        # Prior noise (extremely tight to pin down the coordinate origin)
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]) # rot_x,y,z, pos_x,y,z sigmas
        )
        # Odometry noise model
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])
        )
        # Loop closure noise model (slightly higher confidence/tighter than odom)
        self.loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([5e-4, 5e-4, 5e-4, 5e-3, 5e-3, 5e-3])
        )

    def np_to_pose3(self, T):
        """Convert a 4x4 homogeneous matrix to a gtsam.Pose3."""
        R = T[0:3, 0:3]
        t = T[0:3, 3]
        return gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t[0], t[1], t[2]))

    def pose3_to_np(self, pose3):
        """Convert a gtsam.Pose3 to a 4x4 homogeneous matrix."""
        T = np.eye(4)
        T[0:3, 0:3] = pose3.rotation().matrix()
        t = pose3.translation()
        T[0:3, 3] = [t[0], t[1], t[2]]
        return T

    def add_keyframe(self, kf_id, pose):
        """Add a keyframe pose node and connect to the previous node with an odometry factor."""
        gtsam_pose = self.np_to_pose3(pose)
        self.initial.insert(kf_id, gtsam_pose)
        self.poses[kf_id] = pose.copy()
        
        if kf_id == 0:
            # First keyframe gets anchored
            self.graph.add(gtsam.PriorFactorPose3(0, gtsam_pose, self.prior_noise))
        else:
            # Connect consecutive keyframes
            T_prev = self.poses[kf_id - 1]
            T_rel = np.linalg.inv(T_prev) @ pose
            self.graph.add(
                gtsam.BetweenFactorPose3(
                    kf_id - 1, kf_id, self.np_to_pose3(T_rel), self.odom_noise
                )
            )

    def add_loop_factor(self, kf_src, kf_dst, T_rel):
        """Add a loop closure constraint between kf_src and kf_dst."""
        # Note: T_rel is the pose of kf_dst in kf_src frame
        gtsam_rel = self.np_to_pose3(T_rel)
        self.graph.add(
            gtsam.BetweenFactorPose3(
                kf_src, kf_dst, gtsam_rel, self.loop_noise
            )
        )
        print(f"PGO: Added loop constraint between KF {kf_src} -> KF {kf_dst}")

    def optimize(self):
        """Run GTSAM Levenberg-Marquardt optimizer."""
        try:
            print("PGO: Running pose graph optimization...")
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial)
            result = optimizer.optimize()
            
            # Extract optimized poses
            optimized_poses = {}
            for kf_id in self.poses.keys():
                opt_pose3 = result.atPose3(kf_id)
                opt_pose_np = self.pose3_to_np(opt_pose3)
                optimized_poses[kf_id] = opt_pose_np
                
                # Update our pose representation and initial guess
                self.poses[kf_id] = opt_pose_np
                # Clear and insert to update values for future optimizations
                self.initial.update(kf_id, opt_pose3)
                
            print("PGO: Trajectory optimization complete.")
            return optimized_poses
        except Exception as e:
            print(f"PGO: Optimization failed: {e}")
            return self.poses