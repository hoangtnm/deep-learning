import os
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import label_map_util
from pascal_voc_writer import Writer


PATH_TO_ANNOTATION_DIR = os.path.join('inference_data', 'Annotations', 'xmls')
PATH_TO_RAW_IMAGES_DIR = os.path.join('inference_data', 'JPEGImages')

MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'
PATH_TO_FROZEN_GRAPH = os.path.join(
    '..', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('training_data', 'label_map.pbtxt')

THRESHOLD = 0.5
NUM_CLASSES = 1


# Loading a (frozen) Tensorflow model into memory
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


# Check valid images in the PATH_TO_RAW_IMAGES_DIR
FILENAMES = [f for f in os.listdir(PATH_TO_RAW_IMAGES_DIR) if (
    os.path.isfile(PATH_TO_RAW_IMAGES_DIR + '/' + f) and f.endswith('.jpg'))]
FILENAMES.sort()


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name(
            'detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for image_path in FILENAMES:
            image = Image.open(
                os.path.join(PATH_TO_RAW_IMAGES_DIR, image_path)
            )
            image_width, image_height = image.size
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores,
                    detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes)
            scores = np.squeeze(scores)

            writer = Writer(
                os.path.join(PATH_TO_RAW_IMAGES_DIR, image),
                image_width,
                image_height
            )

            for index, score in enumerate(scores):
                if score < THRESHOLD:
                    continue

                label = category_index[classes[index]]['name']
                ymin, xmin, ymax, xmax = boxes[index]

                writer.addObject(label, int(xmin * image_width), int(ymin * image_height),
                                 int(xmax * image_width), int(ymax * image_height))

            xml_filename = os.path.splitext(image_path)[0] + '.xml'
            annotation_path = os.path.join(
                PATH_TO_ANNOTATION_DIR, xml_filename)
            writer.save(annotation_path)
