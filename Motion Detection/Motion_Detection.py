import cv2
import numpy as np
import time

# Initialize video capture (0 is the default camera)
cap = cv2.VideoCapture(0)

# Create Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Initialize a variable to store the status of the room (Clear/Detected)
room_status = "Clear"
last_motion_time = time.time()  # Track the time of the last motion detected
no_motion_timeout = 5  # Time in seconds after which status changes to "Clear" (if no motion)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Find contours of the moving areas (people moving through the frame)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False  # Flag to check if motion is detected

    # Count large contours
    large_contours = 0
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue  # Ignore small contours that are not objects of interest

        # Get the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Draw a rectangle around the detected moving object (person)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        large_contours += 1  # Count large contours

    # If at least one large contour, motion is detected
    if large_contours > 0:
        room_status = "Detected"
        last_motion_time = time.time()  # Reset the timer

    # If no motion detected for the timeout period, change status to "Clear"
    elif time.time() - last_motion_time > no_motion_timeout:
        room_status = "Clear"

    # Add the room status text to the frame
    cv2.putText(frame, f"{room_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with the tracked object and room status
    cv2.imshow('Security Feed', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
