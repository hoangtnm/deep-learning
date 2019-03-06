from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2


# Construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', required=True,
                    help='path to input video file')
args = vars(parser.parse_args())


# Open a pointer to the video stream and start the FPS timer
stream = cv2.VideoCapture(args['video'])
fps = FPS().start()


while True:
    # Capture frame-by-frame
    (grabbed, frame), frame = stream.read()

    # If the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # Resize the frame and convert it to grayscale (while still
    # retaining 3 channels)
    frame = imutils.resize(frame, width=500)

    # Display a piece of text to the frame (so we can benchmark
    # fairly against the fast method)
    cv2.putText(frame, 'Slow Method', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame and update the FPS counter
    cv2.imshow('Frame', frame)
    cv2.waitKey(1)
    fps.update()


# Stop the timer and display FPS information
fps.stop()
print(f'[INFO] elasped time: {fps.elapsed():.2f}')
print(f'[INFO] approx. FPS: {fps.fps():.2f}')

# When everything done, release the capture
stream.release()
cv2.destroyAllWindows()
