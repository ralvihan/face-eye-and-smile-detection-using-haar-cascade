import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# Load the Haar cascades
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('Cascades/haarcascade_smile.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Initialize lists to store detection counts
face_counts = []
eye_counts = []
smile_counts = []
frames = []

start_time = time.time()
frame_count = 0

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to capture image. Exiting...")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    face_count = len(faces)
    eye_count = 0
    smile_count = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=5,
            minSize=(5, 5)
        )

        eye_count += len(eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Detect smiles within the face region
        smiles = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25, 25)
        )

        smile_count += len(smiles)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

    frame_count += 1
    frames.append(frame_count)
    face_counts.append(face_count)
    eye_counts.append(eye_count)
    smile_counts.append(smile_count)

    # Display the resulting frame
    cv2.imshow('Video', img)

    # Stop detection after 15 seconds
    if time.time() - start_time > 15:
        break

    # Exit when 'ESC' is pressed
    k = cv2.waitKey(30) & 0xFF
    if k == 27:  # ESC key
        break

# Plot the detection results
plt.figure()
plt.plot(frames, face_counts, label='Faces', color='blue')
plt.plot(frames, eye_counts, label='Eyes', color='green')
plt.plot(frames, smile_counts, label='Smiles', color='red')
plt.xlabel('Frames')
plt.ylabel('Count')
plt.title('Detection Counts Over 15 Seconds')
plt.legend()
plt.grid(True)
plt.show()

# Resume video feed after plotting
time.sleep(2)  # Optional pause to view the graph before resuming
cap = cv2.VideoCapture(0)  # Reinitialize video capture
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to capture image. Exiting...")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', img)

    # Exit when 'ESC' is pressed
    k = cv2.waitKey(30) & 0xFF
    if k == 27:  # ESC key
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
