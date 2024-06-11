import cv2
import numpy as np

# Load calibration data
calib_data = np.load('calibration_data.npz')

# Load the matrices
K = calib_data['K1']  # Assuming K1 and K2 are the same
D = calib_data['D1']  # Assuming D1 and D2 are the same
R = calib_data['R']
T = calib_data['T']

# Load stereo images
img_left = cv2.imread('pictures/left_image.jpg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('pictures/right_image.jpg', cv2.IMREAD_GRAYSCALE)
# Check image sizes
print("Image shape:", img_left.shape, img_right.shape)

# Stereo rectification
h, w = img_left.shape[:2]
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, D, K, D, (w, h), R, T)

# Compute the undistortion and rectification transformation map
map1x, map1y = cv2.initUndistortRectifyMap(K, D, R1, P1, (w, h), cv2.CV_16SC2)
map2x, map2y = cv2.initUndistortRectifyMap(K, D, R2, P2, (w, h), cv2.CV_16SC2)

# Apply the rectification maps
rectified_left = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
rectified_right = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)

# Create StereoSGBM matcher
minDisparity = 0
numDisparities = 16 * 6  # Must be divisible by 16
blockSize = 5
P1 = 8 * 3 * blockSize ** 2
P2 = 32 * 3 * blockSize ** 2
disp12MaxDiff = 1
uniquenessRatio = 10
speckleWindowSize = 100
speckleRange = 32

stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
                               numDisparities=numDisparities,
                               blockSize=blockSize,
                               P1=P1,
                               P2=P2,
                               disp12MaxDiff=disp12MaxDiff,
                               uniquenessRatio=uniquenessRatio,
                               speckleWindowSize=speckleWindowSize,
                               speckleRange=speckleRange)

disparity = stereo.compute(rectified_left, rectified_right)

# Check disparity map values
print("Disparity map stats - Min:", np.min(disparity), "Max:", np.max(disparity), "Mean:", np.mean(disparity))

# Filter out invalid disparity values
disparity[disparity < 0] = 0

# Normalize the disparity map for display
disparity_display = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_display = np.uint8(disparity_display)

# Display the disparity map
cv2.imshow('Disparity', disparity_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Reproject image to 3D
try:
    depth_map = cv2.reprojectImageTo3D(disparity, Q)
    cv2.imshow('depth wepth', depth_map)
    cv2.waitKey(0)

    # Replace NaNs and infs with zero
    depth_map[np.isnan(depth_map)] = 0
    depth_map[np.isinf(depth_map)] = 0

    # Check depth map values
    print("Depth map stats - Min:", np.min(depth_map), "Max:", np.max(depth_map), "Mean:", np.mean(depth_map))
except Exception as e:
    print("Error during reprojectImageTo3D:", e)

# Display the depth map
try:
    depth_display = depth_map[:, :, 2]
    depth_display = cv2.normalize(depth_display, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_display = np.uint8(depth_display)

    cv2.imshow('Depth Map', depth_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:
    print("Error displaying depth map:", e)

# Save depth map to file
np.save('depth_map.npy', depth_map)