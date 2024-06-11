# create a python script to take pictures from camera index 1
# and save them to a folder named pictures

import cv2

# Initialize camera
cap = cv2.VideoCapture(1)

# Initialize counter
count = 50

# Run loop to capture images
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow("Camera", frame)

    # Press 's' to save image
    if cv2.waitKey(1) == ord("s"):
        cv2.imwrite(f"pictures/image_{count}.png", frame)
        print(f"Image saved as image_{count}.png")
        count += 1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord("q"):
        break