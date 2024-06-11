import cv2
import numpy as np

# Load calibration data
calib_data = np.load('calibration_data.npz')

# Load the matrices
K1 = calib_data['K1']
D1 = calib_data['D1']
K2 = calib_data['K2']
D2 = calib_data['D2']
R = calib_data['R']
T = calib_data['T']

# Load stereo images
img_left = cv2.imread('pictures/left_image.jpg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('pictures/right_image.jpg', cv2.IMREAD_GRAYSCALE)

# Stereo rectification
h, w = img_left.shape[:2]

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T)

# Compute the undistortion and rectification transformation map
left_map1, left_map2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_16SC2)

# Apply the rectification maps
rectified_left = cv2.remap(img_left, left_map1, left_map2, cv2.INTER_LINEAR)
rectified_right = cv2.remap(img_right, right_map1, right_map2, cv2.INTER_LINEAR)

# StereoBM matcher
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
disparity = stereo.compute(rectified_left, rectified_right)

# Normalize the disparity map for display
disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)

# Display the disparity map
cv2.imshow('Disparity', disparity)
cv2.imwrite('depth2.png', disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Reproject image to 3D
depth_map = cv2.reprojectImageTo3D(disparity, Q)

cv2.imshow('Depth Map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Display the depth map
depth_display = depth_map[:, :, 2]
depth_display = cv2.normalize(depth_display, depth_display, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
depth_display = np.uint8(depth_display)

cv2.imshow('Depth Map', depth_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save depth map to file
np.save('depth_map.npy', depth_map)
