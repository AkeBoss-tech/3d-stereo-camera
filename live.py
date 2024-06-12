import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image

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

image_processor, depth_model = load_depth_model()

camera = cv2.VideoCapture(int(input("Enter camera index: ")))

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    print("Image Captured")

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
        outputs = depth_model(**inputs)
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

    # stretch the formatted image size to the original image size
    formatted = cv2.resize(formatted, (frame.shape[1], frame.shape[0]))

    # Display the output with opencv in grayscale
    cv2.imshow("output", formatted)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()