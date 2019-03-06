# Faster video file FPS with cv2.VideoCapture and OpenCV

When working with video files and OpenCV you are likely using the `cv2.VideoCapture` function.

First, you instantiate your `cv2.VideoCapture`  object by passing in the path to your input video file.

Then you start a loop, calling the `.read`  method of `cv2.VideoCapture`  to poll the next frame from the video file so you can process it in your pipeline.

The *problem* (and the reason why this method can feel slow and sluggish) is that you’re **both reading and decoding the frame in your main processing thread!**

As I’ve mentioned in [previous posts](https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/), the `.read`  method is a [blocking operation](https://en.wikipedia.org/wiki/Blocking_(computing)) — the main thread of Python + OpenCV application is entirely blocked (i.e., stalled) until the frame is read from the video file, decoded, and returned to the calling function.

By moving these blocking I/O operations to a separate thread and maintaining a queue of decoded frames we can actually **improve our FPS processing rate by over 52%!**

This increase in frame processing rate (and therefore our overall video processing pipeline) comes from *dramatically reducing latency* — we don’t have to wait for the `.read`  method to finish reading and decoding a frame; instead, there is ***always*** a pre-decoded frame ready for us to process.

To accomplish this latency decrease our goal will be to move the reading and decoding of video file frames to an entirely separate thread of the program, freeing up our main thread to handle the actual image processing.

## Using threading to buffer frames with OpenCV

To improve the FPS processing rate of frames read from video files with OpenCV we are going to utilize ***threading*** and the [queue data structure](https://en.wikipedia.org/wiki/Queue_(abstract_data_type)):

<p align=center><img src='https://www.pyimagesearch.com/wp-content/uploads/2017/01/file_video_stream_queue.png'></p>

Since the `.read`  method of `cv2.VideoCapture`  is a blocking I/O operation we can obtain a significant speedup simply by creating a ***separate thread*** from our main Python script that is solely responsible for reading frames from the video file and maintaining a queue.

We can use `FileVideoStream` class in `imutils` but we’re going to review the code so you can understand what’s going on under the hood:

```python
from threading import Thread
from queue import Queue
import sys
import cv2
```

The `Thread` class is used to create and start threads in the Python.

We can now define the constructor to `FileVideoStream`:

```python
class FileVideoStream:
    def __init__(self, path, transform=None, queue_size=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)

        # intialize thread and
        # start a thread separate from the main thread.
        # This thread will call the .update  method
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
```

Our constructor takes a single required argument followed by an optional one:

- path : The path to our input video file.
- queue_size : The maximum number of frames to store in the queue. This value defaults to 128 frames, but you depending on the frame dimensions of your video and the amount of memory you can spare.

We then initialize a boolean to indicate if the threading process should be stopped along with our actual `Queue` data structure.

To kick off the thread, we’ll next define the `start` method:

```python
    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self
```

The `update`  method is responsible for **reading** and **decoding** frames from the video file, along with maintaining the actual queue data structure:

```python
    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set,
            # stop the thread
            if self.stopped:
                return

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return

                # add the frame to the queue
                self.Q.put(frame)
```

If our queue is ***not full***, we read the next frame from the video stream, check to see if we have reached the end of the video file, and then update the queue

```python
    def read(self):
        # return next frame in the queue
        return self.Q.get()
```

We’ll create a convenience function named `more`  that will return `True`  if there are still more frames in the queue (and `False`  otherwise):

```python
    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0
```

And finally, the `stop` method will be called if we want to stop the thread prematurely (i.e., before we have reached the end of the video file):

```python
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released
        # (producer thread might be still grabbing frame)
        self.thread.join()
```

## Full code

```python
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
    cv2.putText(frame, f'Queue Size: {fvs.Q.qsize()}',
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
```

Reference: [Faster video file FPS with cv2.VideoCapture and OpenCV](https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/)
