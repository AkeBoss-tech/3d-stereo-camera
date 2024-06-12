from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("yolov8m.pt")

results = model.predict(source="1", show=True) # Display preds. Accepts all YOLO predict arguments
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments
