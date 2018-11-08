# Guide to feeding data in TensorFlow


### This guide is conducted to give instruction on how to:

- Feed the in-memory data to your model.

- Feed the TFRecords format to your model.

- Feed the Raw Images on disk to your model.


### Using TFRecord

`Dataset used ` —  MNIST

`Model Architecture ` —  Two hidden layers of 100 neurons each and 1 output layer of 10 neurons

```python
import tensorflow as tf
import numpy as np
import os, sys
import time

# Define the type of features columns to be used on model.
feature_column = [tf.feature_column.numeric_column(key='image', shape=(784,))]

# Define the model
model = tf.estimator.DNNClassifier([100,100],n_classes=10,feature_columns=feature_column)

"""Convert the TFRecords format back to tensor
dtype: tf.float32 in case of image and to tf.int32 in case of label
TFRecords --> Serialized Example --> Example --> Tensor
"""
def _parse_(serialized_example):
    feature = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    example = tf.parse_single_example(serialized_example, feature)
    image = tf.decode_raw(example['image_raw'], tf.int64)
    label = tf.cast(example['label'], tf.int32)
    return (dict({'image':image}), label)

# Use this _parse_ function and return the batches of (features, label) to be feed to the model
mnist_tfrecord_path = os.path.abspath('./mnist_train.tfrecords')
def tfrecord_train_input_fn(batch_size=32):
    tfrecord_dataset = tf.data.TFRecordDataset(mnist_tfrecord_path)
    tfrecord_dataset = tfrecord_dataset.map(lambda   x:_parse_(x)).shuffle(True).batch(batch_size)
    tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()
    
    return tfrecord_iterator.get_next()
    
# Start to train the model
model.train(lambda:tfrecord_train_input_fn(32),steps=200)
```

# Read RAW images

```python
path = os.path.abspath('./digit-recognizer/train/')
def _ondisk_parse_(filename):
    filename = tf.cast([filename],tf.string)
    
    label = tf.string_split([tf.string_split(filename,'_').values[1]],'.').values[0]
    label = tf.string_to_number([label],tf.int32)
    
    path = os.path.abspath('./digit-recognizer/train//')
    path = tf.cast([path],tf.string)
    
    final_path = tf.string_join((path,tf.cast(['/'],tf.string),filename))
    
    image_string = tf.read_file(final_path[0])
    image = tf.image.decode_jpeg(image_string)
    image = tf.cast(image,tf.int8)
    image = tf.cast(image,tf.float32)
    image_reshaped = tf.reshape(image,(784,))
    return (dict({'image':image}),label)
	
# Final train input function
def ondisk_train_input_fn(filenames,batch_size=32):
    dataset  = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(lambda x:_ondisk_parse_(x)).shuffle(True).batch(batch_size)
    ondisk_iterator = dataset.make_one_shot_iterator()
    
    return ondisk_iterator.get_next()
	
# List of the images in train folder
f = !ls ./digit-recognizer/train/ 
# Train the model
model.train(lambda:ondisk_train_input_fn(f,32),steps=200)
```

References:

- [Beginner’s guide to feeding data in Tensorflow — Part1](https://medium.com/coinmonks/beginners-guide-to-feeding-data-in-tensorflow-faf21a745e4c)

- [Beginner’s guide to feeding data in Tensorflow — Part2](https://medium.com/coinmonks/beginners-guide-to-feeding-data-in-tensorflow-part2-5e2506d75429)
