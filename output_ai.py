# create a python script to read through the pictures folder and
# run them through the depth model and object detection model
# save these two images to a folder named output

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from ultralytics import YOLO
from PIL import Image
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go


device = None

def load_depth_model():
    global device
    """Loads the DPT-Hybrid-Midas depth estimation model."""
    image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)  # Move model to GPU

    return image_processor, model

""" object_model = YOLO("yolov8m.pt") """
image_processor, depth_model = load_depth_model()

# Loop through all the pictures
for file in os.listdir("pictures"):
    if not "pro" in file:
        continue
    if "trash" in file: 
        continue
    print(file)
    # load the image
    image = cv2.imread(f"pictures/{file}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # run the depth model
    # Convert frame to PIL Image (required by DPT model)
    frame_pil = Image.fromarray(image)

    # Prepare image for the model
    inputs = image_processor(images=frame_pil, return_tensors="pt").to(device)  # Move input to GPU

    # Enable cuDNN (optional)
    # torch.backends.cudnn.benchmark = True  # Uncomment if applicable

    # Run inference using PyTorch
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original frame size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.shape[:2][::-1],  # Get height, width from frame shape
        mode="bilinear",
        align_corners=False,
    )

    # Process and visualize the depth map
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")

    # stretch the formatted image size to the original image size
    formatted = cv2.resize(formatted, (image.shape[1], image.shape[0]))

    # save the formatted image
    cv2.imwrite(f"output/depth_{file}", formatted)

    """# Create a 3D matplotlib plot of the depth map
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    # Generate a grid of coordinates
    x = np.linspace(0, formatted.shape[1] - 1, formatted.shape[1])
    y = np.linspace(0, formatted.shape[0] - 1, formatted.shape[0])
    x, y = np.meshgrid(x, y)

    # Normalize the depth map for better visualization
    z = formatted / 255.0 * np.max(output)

    # Plot the surface with texture from the original image
    ax.plot_surface(x, y, z, facecolors=image_rgb / 255, rstride=1, cstride=1, antialiased=True)

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')

    # Save the 3D plot as an image
    plt.savefig(f"output/depth_3d_{file}")

    # Close the plot to free memory
    plt.close(fig) """
    # Create a 3D plot with Plotly
    x = np.linspace(0, formatted.shape[1] - 1, formatted.shape[1])
    y = np.linspace(0, formatted.shape[0] - 1, formatted.shape[0])
    x, y = np.meshgrid(x, y)

    z = formatted / 255.0 * np.max(output)

    # Convert image RGB values to a normalized range [0, 1]
    image_rgb_normalized = image_rgb / 255.0

    # Flatten the RGB channels for use with surfacecolor
    surfacecolor = np.zeros((formatted.shape[0], formatted.shape[1]), dtype=float)
    for i in range(formatted.shape[0]):
        for j in range(formatted.shape[1]):
            surfacecolor[i, j] = (image_rgb_normalized[i, j, 0] * 0.299 + 
                                  image_rgb_normalized[i, j, 1] * 0.587 + 
                                  image_rgb_normalized[i, j, 2] * 0.114)

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, surfacecolor=surfacecolor, colorscale='viridis')])
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Depth'
    ))


    # Save the 3D plot as an interactive HTML file
    fig.write_html(f"output/depth_3d_{file.split('.')[0]}.html")

    # run YOLOv8
    """ results = object_model.predict(f"pictures/{file}", show=False)
    print(len(results)) """
