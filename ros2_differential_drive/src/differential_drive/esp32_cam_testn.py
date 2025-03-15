import numpy as np
import cv2 as cv

# URL for the stream
URL = "http://192.168.142.67" #"http://192.168.142.67" for left camera and 235 for right camera

# Open the video capture stream
cap = cv.VideoCapture(URL + ":81/stream")

# Check if the capture is opened correctly
if not cap.isOpened():
    print("Can't open camera")
    exit()

# Main loop for reading and displaying frames
while True:
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Can't receive frame, exiting...")
        break

    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Display the grayscale frame
    cv.imshow('frame', gray)

    # Exit the loop when 'q' key is pressed
    key = cv.waitKey(1) & 0xFF
    if cv.waitKey(1) == ord('q'):
        break
    elif key == ord('s'):
        cv.imwrite('/home/neeraj/our_setp_cmra.jpg',frame)
        break
# Release the capture object and close all windows
cap.release()
cv.destroyAllWindows()
# above code works and is for testing camera of esp32 cam.
