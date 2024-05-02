from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("yolov8m.pt")

results = model.predict(source="0", show=True) # Display preds. Accepts all YOLO predict arguments
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# Open camera stream with opencv
""" cap = cv2.VideoCapture(0)
while True:
    # Read frame
    ret, frame = cap.read()
    # Convert to PIL image
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Run inference
    results = model(frame)
    
    # Get bounding boxes
    boxes = results[0].boxes.xyxy.numpy()
    labels = results[0].names.numpy()

    # Display the resulting frame
    cv2.imshow("YOLOv8 Inference", frame)
    # Press q to quit
    if cv2.waitKey(1) == ord("q"):
        break """