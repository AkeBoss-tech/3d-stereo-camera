import cv2
import numpy as np

# Load calibration data
calib_data = np.load('calibration_data.npz')

# Load the matrices
K = calib_data['K1']  # Assuming K1 and K2 are the same
D = calib_data['D1']  # Assuming D1 and D2 are the same
R = calib_data['R']
T = calib_data['T']

# Load stereo images (left and right)
img_left = cv2.imread('output/left_image.jpg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('output/right_image.jpg', cv2.IMREAD_GRAYSCALE)

# Image size
image_size = (img_left.shape[1], img_left.shape[0])

# Rectify the images
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K, D,
                                                  K, D,
                                                  image_size, R, T, alpha=0)

map1x, map1y = cv2.initUndistortRectifyMap(K, D, R1, P1, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K, D, R2, P2, image_size, cv2.CV_32FC1)

rectified_left = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
rectified_right = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)

# Compute the disparity map
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=21)
disparity_map = stereo.compute(rectified_left, rectified_right)

# Convert disparity map to float32
disparity_map = disparity_map.astype(np.float32) / 16.0

# Avoid division by zero by setting disparity values of 0 to a small number
disparity_map[disparity_map == 0] = 0.1

# Compute the depth map using the disparity-to-depth formula
focal_length = P1[0, 0]  # Focal length from the projection matrix P1
baseline = np.linalg.norm(T)  # Baseline is the norm of the translation vector T

depth_map = (focal_length * baseline) / disparity_map

# Normalize depth map for visualization (optional)
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map_normalized = np.uint8(depth_map_normalized)

# Save or display the depth map
cv2.imwrite('depth_map.png', depth_map_normalized)
cv2.imshow('Depth Map', depth_map_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np

# Load calibration data
calib_data = np.load('calibration_data.npz')

# Load the matrices
K = calib_data['K1']  # Assuming K1 and K2 are the same
D = calib_data['D1']  # Assuming D1 and D2 are the same
R = calib_data['R']
T = calib_data['T']

# Load stereo images (left and right)
img_left = cv2.imread('output/left_image.jpg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('output/right_image.jpg', cv2.IMREAD_GRAYSCALE)

# Image size
image_size = (img_left.shape[1], img_left.shape[0])

# Rectify the images
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K, D,
                                                  K, D,
                                                  image_size, R, T, alpha=0)

map1x, map1y = cv2.initUndistortRectifyMap(K, D, R1, P1, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K, D, R2, P2, image_size, cv2.CV_32FC1)

rectified_left = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
rectified_right = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)

# Compute the disparity map
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=21)
disparity_map = stereo.compute(rectified_left, rectified_right)

# Convert disparity map to float32
disparity_map = disparity_map.astype(np.float32) / 16.0

# Avoid division by zero by setting disparity values of 0 to a small number
disparity_map[disparity_map == 0] = 0.1

# Compute the depth map using the disparity-to-depth formula
focal_length = P1[0, 0]  # Focal length from the projection matrix P1
baseline = np.linalg.norm(T)  # Baseline is the norm of the translation vector T

depth_map = (focal_length * baseline) / disparity_map

# Normalize depth map for visualization (optional)
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map_normalized = np.uint8(depth_map_normalized)

# Save or display the depth map
cv2.imwrite('depth_map.png', depth_map_normalized)
cv2.imshow('Depth Map', depth_map_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()