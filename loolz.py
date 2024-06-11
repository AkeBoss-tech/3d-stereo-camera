import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load calibration data
calib_data = np.load('calibration_data.npz')

# Load the matrices
K = calib_data['K1']  # Assuming K1 and K2 are the same
D = calib_data['D1']  # Assuming D1 and D2 are the same
R = calib_data['R']
T = calib_data['T']

# Baseline in meters or centimeters
baseline = np.linalg.norm(T)  # This will give the baseline in the unit used in T

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
minDisparity = 2
numDisparities = 16 * 6  # Must be divisible by 16
blockSize = 9
P1 = 8 *10 * 3 * blockSize ** 2
P2 = 32 *10 * 3 * blockSize ** 2
disp12MaxDiff = 1
uniquenessRatio = 10
speckleWindowSize = 200
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

# Reproject image to 3D
try:
    depth_map = cv2.reprojectImageTo3D(disparity, Q)

    # Replace NaNs and infs with zero
    depth_map[np.isnan(depth_map)] = 0
    depth_map[np.isinf(depth_map)] = 0

    # Extract the Z values (depth)
    depth_values = depth_map[:, :, 2]

    # Check depth map values
    print("Depth map stats - Min:", np.min(depth_values), "Max:", np.max(depth_values), "Mean:", np.mean(depth_values))
except Exception as e:
    print("Error during reprojectImageTo3D:", e)

# Define the hover event handler
def hover(event):
    if event.inaxes == ax:
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < depth_values.shape[1] and 0 <= y < depth_values.shape[0]:
            depth = depth_values[y, x]
            ax.set_title(f'Depth: {depth:.2f} meters')
        else:
            ax.set_title('')

# Create the depth map plot
fig, ax = plt.subplots()
cax = ax.imshow(depth_values, cmap='jet')
cbar = fig.colorbar(cax, ax=ax, label='Depth (meters)')
plt.title('Depth Map')

# Connect the hover event to the event handler function
fig.canvas.mpl_connect('motion_notify_event', hover)

plt.show()