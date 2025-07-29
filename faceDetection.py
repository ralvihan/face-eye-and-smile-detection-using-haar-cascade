import numpy as np
import cv2

# Load the pre-trained face cascade classifier
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

while True:
    # Capture frame-by-frame
    ret, img = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Video', img)

    # Break the loop when 'ESC' is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        print("Exiting...")
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
