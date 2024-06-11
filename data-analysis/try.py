import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Path to the depth map image in the output folder
depth_map_path = "output/depth_GLPN/depth_left_image.jpg"
# depth_map_path = "output/depth_image_0.png"

# Load the depth map image
depth_image = Image.open(depth_map_path)
depth_map = np.array(depth_image)

# Camera calibration parameters
K = [
    [953.0615149722063, 0, 670.5135610507407],
    [0, 959.3798799023834, 366.76133360900457],
    [0, 0, 1]
]
fx = K[0][0]  # Focal length in pixels along x-axis
fy = K[1][1]  # Focal length in pixels along y-axis
cx = K[0][2]  # Principal point x-coordinate
cy = K[1][2]  # Principal point y-coordinate

# Define a function to calculate distances for all pixels
def calculate_distances(depth_map, fx, fy, cx, cy):
    distances = np.zeros_like(depth_map, dtype=float)
    for y in range(depth_map.shape[0]):
        for x in range(depth_map.shape[1]):
            depth_value = depth_map[y, x]
            distances[y, x] = depth_value * np.sqrt((x - cx)**2 / fx**2 + (y - cy)**2 / fy**2 + 1)
    return distances

# Calculate distances
distances = calculate_distances(depth_map, fx, fy, cx, cy)

# Create a new matplotlib image where the scale shows the distance away from the object in units
plt.imshow(distances, cmap='viridis')
plt.colorbar(label='Distance (units)')
plt.title('Distance Map')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.show()