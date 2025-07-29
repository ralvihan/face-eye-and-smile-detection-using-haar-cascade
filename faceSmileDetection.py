import numpy as np
import cv2

# Load the face and smile cascade classifiers
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('Cascades/haarcascade_smile.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

while True:
    # Capture frame from the camera
    ret, img = cap.read()
    if not ret:
        print("Error: Unable to capture video")
        break

    # Convert the image to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Define the region of interest (ROI) for smile detection
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect smile within the face ROI
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25, 25)
        )

        # Loop through detected smiles and draw rectangles
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', img)

    # Break the loop when 'ESC' is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # ESC key
        print("Exiting...")
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()