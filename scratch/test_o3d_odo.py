import open3d as o3d
import numpy as np

# Create two dummy point clouds with a known translation
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(np.array([[0, 0, 1]], dtype=float))

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(np.array([[0, 0, 2]], dtype=float)) # shifted by +1 in Z

# Estimate normals for point-to-plane ICP
pcd1.estimate_normals()
pcd2.estimate_normals()

# Run ICP from pcd1 (source) to pcd2 (target) with identity initial guess
res = o3d.pipelines.registration.registration_icp(
    pcd1, pcd2, 5.0, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

print("ICP transformation (pcd1 -> pcd2):")
print(res.transformation)

# Transform pcd1 with res.transformation
pcd1_trans = pcd1.clone().transform(res.transformation)
print("pcd1 translated point:", np.asarray(pcd1_trans.points))
