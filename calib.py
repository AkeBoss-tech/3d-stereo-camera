import numpy as np

# Example data from your JSON for camera 1 and camera 2 (assuming same calibration for simplicity)
""" camera_matrix = np.array([
    [953.0615149722063, 0, 670.5135610507407],
    [0, 959.3798799023834, 366.76133360900457],
    [0, 0, 1]
])

distortion_coefficients = np.array([0.03843197821443714, -0.13773331981711934, -0.0004603474945041381, 0.002767313475790507, 0.092320474121936]) """

camera_matrix = np.array([
        [
            929.0767864652852,
            0,
            631.3577879062562
        ],
        [
            0,
            926.8831686779699,
            361.2536918659137
        ],
        [
            0,
            0,
            1
        ]
    ])
distortion_coefficients = np.array([
        0.1089513793964826,
        -0.22901263789728088,
        0.0012388242047717127,
        -0.0005792140975913752,
        0.08701326622031279
    ])

# Rotation and translation between the two cameras (example values)
R = np.eye(3)  # Assuming no rotation for simplicity
T = np.array([-0.1, 0, 0])  # Assuming a 10 cm translation along the x-axis

# Save to .npz file
np.savez('calibration_data.npz', K1=camera_matrix, D1=distortion_coefficients, K2=camera_matrix, D2=distortion_coefficients, R=R, T=T)
