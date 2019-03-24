import os
import sys
import tarfile
import imutils
import time
import cv2
import numpy as np
import tensorflow as tf
import argparse

from utils import visualization_utils as vis_util
from utils import label_map_util
from object_detection.utils import ops as utils_ops

from io import StringIO
from matplotlib import pyplot as plt
from imutils.video import FileVideoStream
from imutils.video import FPS
from PIL import Image


# # Model preparation
MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = os.path.join('models', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')


# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Create a VideoCapture object.
# Its argument can be either the camera index or the name of a video file.
print('[INFO] starting video file thread...')
fvs = FileVideoStream(0).start()
fps = FPS().start()


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while fvs.more():
            image_np = fvs.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)            
            cv2.imshow('Object detection', cv2.resize(image_np, (800,600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


# When everything done, release the capture
cv2.destroyAllWindows()
fvs.stop()

# Stop the timer and display FPS information
fps.stop()
print(f'[INFO] elasped time: {fps.elapsed():.2f}')
print(f'[INFO] approx. FPS: {fps.fps():.2f}')
