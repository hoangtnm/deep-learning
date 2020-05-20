import cv2

"""
To capture a video, you need to create a VideoCapture object
Its argument can be either the device index or the name of a video file.
After that, you can capture frame-by-frame.
But at the end, don’t forget to release the capture
"""

cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Capture frame-by-frame.
    ret, frame = cap.read()

    # Converts BGR to RGB.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame.
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
