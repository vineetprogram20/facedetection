import cv2
import numpy as np

# Load pre-trained Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Function to detect eye blinking using Eye Aspect Ratio (EAR)
def detect_blinking(eye_region):
    # Implement EAR calculation logic here
    return False

# Function to detect smiling
def detect_smile(roi_gray):
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
    return len(smiles) > 0

# Function to check for motion in the frame
def detect_motion(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(prev_gray, curr_gray)
    _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    motion_pixels = np.count_nonzero(motion_mask)
    return motion_pixels > 1000
# Function to detect if eyes are open
def detect_eyes_open(eyes):
    return len(eyes) > 0

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize previous frame
ret, prev_frame = cap.read()

# Initialize motion frames counter
motion_frames = 0

# Main loop for liveness detection
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eye blinking
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if detect_blinking(roi_gray) or not detect_eyes_open(eyes):
             # Either eyes are closed or not detected, take appropriate action (considered as spoofing)
            cv2.putText(frame, "Spoofing Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            continue

        # Detect smiling
        if detect_smile(roi_gray):
            # Smile detected, take appropriate action
            pass

        # Draw green rectangle around the face (live person)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Update the previous frame
    prev_frame = frame.copy()

    # Display the frame
    cv2.imshow('Liveness Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()