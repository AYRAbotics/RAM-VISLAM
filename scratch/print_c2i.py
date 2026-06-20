import numpy as np
from scipy.spatial.transform import Rotation

R_D2C = np.array([
    [ 0.99999636, -0.00241311,  0.00117634],
    [ 0.00241741,  0.99999034, -0.00367048],
    [-0.00116747,  0.00367331,  0.99999255]
])
T_D2C = np.array([0.01454706, 0.00018594, 0.00039981])

R_D2I = np.eye(3)
T_D2I = np.array([-0.00552, 0.0051, 0.01174])

T_d2c = np.eye(4)
T_d2c[0:3, 0:3] = R_D2C
T_d2c[0:3, 3] = T_D2C

T_d2i = np.eye(4)
T_d2i[0:3, 0:3] = R_D2I
T_d2i[0:3, 3] = T_D2I

T_c2i = T_d2i @ np.linalg.inv(T_d2c)

# Print before correction
print("T_c2i before roll offset:")
print(T_c2i)
print("Rotation angles (xyz degrees):", Rotation.from_matrix(T_c2i[0:3, 0:3]).as_euler('xyz', degrees=True))

# Print after correction
roll_rad = np.radians(-3.5)
R_corr = Rotation.from_euler('xyz', [roll_rad, 0.0, 0.0]).as_matrix()
T_c2i_corr = T_c2i.copy()
T_c2i_corr[0:3, 0:3] = T_c2i_corr[0:3, 0:3] @ R_corr
print("\nT_c2i after roll offset -3.5:")
print(T_c2i_corr)
print("Rotation angles (xyz degrees):", Rotation.from_matrix(T_c2i_corr[0:3, 0:3]).as_euler('xyz', degrees=True))
