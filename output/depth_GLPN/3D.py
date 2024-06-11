import os
import numpy as np
import cv2
import plotly.graph_objects as go
from PIL import Image

# Directory containing depth map images and corresponding original images
depth_map_dir = "output/depth_GLPN"
original_image_dir = "./pictures"

# Ensure output directory for 3D plots exists
output_dir = "output/depth_GLPN_3d"
os.makedirs(output_dir, exist_ok=True)

# Process each depth map image
for file in os.listdir(depth_map_dir):
    if file.endswith((".jpg", ".jpeg", ".png")):
        print(f"Processing {file}")

        # Load the depth map image
        depth_image_path = os.path.join(depth_map_dir, file)
        depth_image = Image.open(depth_image_path)
        depth_array = np.array(depth_image)

        # Normalize depth values for visualization
        depth_normalized = depth_array / 255.0

        # Load the corresponding original image
        original_image_path = os.path.join(original_image_dir, file[6:])
        if not os.path.exists(original_image_path):
            print(f"Original image for {file} not found. {original_image_path}")
            continue

        original_image = Image.open(original_image_path).convert('RGB')
        original_array = np.array(original_image)

        # Resize original image to match depth map dimensions if necessary
        if original_array.shape[:2] != depth_array.shape:
            original_array = cv2.resize(original_array, (depth_array.shape[1], depth_array.shape[0]))

        # Generate grid for plotting
        height, width = depth_normalized.shape
        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)
        x, y = np.meshgrid(x, y)

        # Normalize RGB values for visualization
        rgb_normalized = original_array / 255.0

        # Flatten RGB channels for use with surfacecolor
        surfacecolor = np.zeros((height, width), dtype=float)
        for i in range(height):
            for j in range(width):
                surfacecolor[i, j] = (rgb_normalized[i, j, 0] * 0.299 + 
                                      rgb_normalized[i, j, 1] * 0.587 + 
                                      rgb_normalized[i, j, 2] * 0.114)

        # Create a 3D surface plot
        fig = go.Figure(data=[go.Surface(z=depth_normalized, x=x, y=y, surfacecolor=surfacecolor, colorscale='viridis')])
        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Depth'
        ))

        # Save the 3D plot as an interactive HTML file
        plot_output_path = os.path.join(output_dir, f"depth_3d_{os.path.splitext(file)[0]}.html")
        fig.write_html(plot_output_path)

print("3D depth map visualization with color completed.")
