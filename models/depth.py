import cv2
import numpy as np
import requests
import torch
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation

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

# Initialize camera
cap = cv2.VideoCapture(1)

# Load depth estimation model
image_processor, model = load_depth_model()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to RGB (assuming model expects RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert frame to PIL Image (required by DPT model)
    frame_pil = Image.fromarray(frame)

    # Prepare image for the model
    inputs = image_processor(images=frame_pil, return_tensors="pt").to(device)  # Move input to GPU

    # Enable cuDNN (optional)
    # torch.backends.cudnn.benchmark = True  # Uncomment if applicable

    # Run inference using PyTorch
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original frame size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=frame.shape[:2][::-1],  # Get height, width from frame shape
        mode="bicubic",
        align_corners=False,
    )

    # Process and visualize the depth map
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")

    # Choose one of the following lines to convert and display the depth map:
    # Option 1: Convert PIL Image to NumPy array before cv2.cvtColor
    depth_map = cv2.cvtColor(np.array(formatted), cv2.COLOR_RGB2BGR)

    # Option 2: Convert PIL Image directly to BGR using OpenCV
    # depth_map = cv2.cvtColor(Image.fromarray(formatted), cv2.COLOR_PRGB2BGR)

    # Display results
    cv2.imshow('Frame', frame)
    cv2.imshow('Depth Map', depth_map)

    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
