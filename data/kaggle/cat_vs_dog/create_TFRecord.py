from random import shuffle
import glob
import tensorflow as tf
import numpy as np
import cv2

SHUFFLE_DATA = True
DATA_TRAIN_PATH = "./train/*.jpg"
IMG_SIZE = (224, 224)
TRAIN_DATA = "train.tfrecord"

"""Read addresses and labels from `train` folder
The `glob` module return a list of paths matching a pathname pattern

Args:
    DATA_TRAIN_PATH = Path to the `train`
Return:
    addrs: address of each picture in the train folder
    labels: 0 = Cat, 1 = Dog
"""
addrs = glob.glob(DATA_TRAIN_PATH)
labels = [0 if 'cat' in addr else 1 for addr in addrs]

# Shuffle the data
if SHUFFLE_DATA:
    # Return a list of tuples, each one includes (addrs, label)
    data = list(zip(addrs, labels))
    shuffle(data)
    addrs, labels = zip(*data)

# Divide the hata into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]
val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

"""Create TFRecord
01. Load the image and convert it to float32
02. Stuff the images in a protocol buffer called Example including Feature
03. Serialize the protocol buffer to a string and write it to a TFRecords file
"""
def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    return img

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

writer = tf.python_io.TFRecordWriter(TRAIN_DATA)

for index in range(len(train_addrs)):
    img = load_img(train_addrs[index])
    label = train_labels[index]

    feature = {"train/image": _bytes_feature(tf.compat.as_bytes(img.tostring())),
               "train/label": _int64_feature(label)}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(example.SerializeToString())

writer.close()
