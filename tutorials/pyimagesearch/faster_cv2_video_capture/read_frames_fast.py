from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', required=True,
    help='path to input video file')
args = vars(parser.parse_args())


print('[INFO] starting video file thread...')
fvs = FileVideoStream(args['video']).start()
time.sleep(1.0)


# start the FPS timer
fps = FPS().start()


while fvs.more():
    # grab the frame from the threaded video file stream, resize it
    frame = fvs.read()
    frame = imutils.resize(frame, width=500)

    # display the size of the queue on the frame
    cv2.putText(frame, 'Queue Size: {}'.format(fvs.Q.qsize()),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	

    # show the frame and update the FPS counter
    cv2.imshow('Frame', frame)
    cv2.waitKey(1)
    fps.update()


# Stop the timer and display FPS information
fps.stop()
print(f'[INFO] elasped time: {fps.elapsed():.2f}')
print(f'[INFO] approx. FPS: {fps.fps():.2f}')


# When everything done, release the capture
cv2.destroyAllWindows()
fvs.stop()