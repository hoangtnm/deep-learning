import os
import sys
import glob
import argparse
import tensorflow as tf
import numpy as np
import cv2
from random import shuffle

FLAGS = None

def get_init_dataset(SHUFFLE=True):
    """Read addresses and labels from `train` folder
    The `glob` module return a list of paths matching a pathname pattern

    Args:
        DATA_PATH = Path to the `input data`

    Returns:
        addrs  (list): address of each picture in the train folder
        labels (list): 0 = Cat, 1 = Dog
    """
    DATA_PATH = "%s/*.jpg" % (FLAGS.input_data)
    addrs = glob.glob(DATA_PATH)
    labels = [0 if 'cat' in addr else 1 for addr in addrs]

    # Shuffle the data
    if SHUFFLE:
        # Return a list of tuples, each one includes (addrs, label)
        data = list(zip(addrs, labels))
        shuffle(data)
        addrs, labels = zip(*data)

    # Divide the hata into 60% train, 20% validation, and 20% test
    init_dataset = {
        'train_addrs' : addrs[0:int(0.6*len(addrs))],
        'train_labels' : labels[0:int(0.6*len(labels))],
        'val_addrs' : addrs[int(0.6*len(addrs)):int(0.8*len(addrs))],
        'val_labels' : labels[int(0.6*len(addrs)):int(0.8*len(addrs))],
        'test_addrs' : addrs[int(0.8*len(addrs)):],
        'test_labels' : labels[int(0.8*len(labels)):]
    }
    
    return init_dataset

"""Create TFRecord
01. Load the image and convert it to float32
02. Stuff the images in a protocol buffer called Example including Feature
03. Serialize the protocol buffer to a string and write it to a TFRecord file
"""
def load_img(img_path):
    IMG_SIZE = FLAGS.img_size

    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    
    return img

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(dataset, name):
    """Converts a dataset to tfrecord."""
    images = dataset[name + '_addrs']
    labels = dataset[name + '_labels']

    filename = os.path.join(FLAGS.directory, name + '.tfrecord')
    print('Writing', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(len(images)):
            image_raw = load_img(images[index]).tostring()
            label = labels[index]
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": _int64_feature(int(label)),
                        "image_raw": _bytes_feature(tf.compat.as_bytes(image_raw))
                    }))
            writer.write(example.SerializeToString())

def main(unused_argv):
    # Convert to Examples and write the result to TFRecords.
    convert_to(dataset=get_init_dataset(), name='train')
    #convert_to(dataset.val, 'val')
    #convert_to(dataset.test, 'test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_data',
                        default='./train',
                        required=False,
                        help="path to input data")
    parser.add_argument('-o', '--output_dataset',
                        default='train',
                        required=False, 
                        help="type of the output dataset -- train/val/test")
    parser.add_argument('-s', '--image_size',
                        default=(224, 224),
                        required=False,
                        help="output image size")
    parser.add_argument('-d', '--directory',
                        type=str,
                        default='/tmp/data',
                        help='Directory to write the converted result')
    parser.add_argument('--validation_size',
                        type=int,
                        default=5000,
                        help="Valiation size")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)