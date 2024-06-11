import os
import torch
import numpy as np
from PIL import Image
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import cv2

# Load the model and processor
processor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# Ensure output directory exists
os.makedirs("output/depth_GLPN", exist_ok=True)

# Process images
for file in os.listdir("pictures"):
    if file.endswith((".jpg", ".jpeg", ".png")):
        if not "pro" in file:
            continue
        print(f"Processing {file}")

        # Load image
        image_path = os.path.join("pictures", file)
        image = Image.open(image_path)

        # Prepare image for the model
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # Visualize the prediction
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)

        # Save the depth map
        depth_output_path = os.path.join("output/depth_GLPN", f"depth_{file}")
        depth.save(depth_output_path)

print("Processing completed for GLPN model.")
