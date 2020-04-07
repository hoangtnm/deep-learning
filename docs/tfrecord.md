# tf.data and TFRecord: Build TensorFlow input pipelines <!-- omit in toc -->

To read data efficiently it can be helpful to serialize your data and store it in a set of files (100-200MB each) that can each be read linearly. This is especially true if the data is being streamed over a network. This can also be useful for caching any data-preprocessing.

The TFRecord format is a simple format for storing a sequence of binary records. It is optimized for use with Tensorflow in multiple ways. Especially for datasets that are too large to be stored fully in memory this is an advantage as only the `data that is required at the time (e.g. a batch) is loaded from disk and then processed`.

<!-- `tf.Example` message (or protobuf) is a flexible message type that represents a `{"string": value}` mapping. It is designed for use with TensorFlow and is used throughout the higher-level APIs such as [TFX](https://www.tensorflow.org/tfx/). -->

> **\*Note:** While useful, TFRecord is optional. There is no need to convert existing code to use TFRecords, unless you are using [tf.data](https://www.tensorflow.org/guide/datasets) and reading data is still the bottleneck to training. See [Data Input Pipeline Performance](https://www.tensorflow.org/guide/performance/datasets) for dataset performance tips.

## Contents <!-- omit in toc -->

- [Resources](#resources)
- [Structuring TFRecord](#structuring-tfrecord)
- [tf.Example](#tfexample)
  - [Data types for **tf.Example**](#data-types-for-tfexample)
  - [Creating a tf.Example message](#creating-a-tfexample-message)
- [TFRecords format details](#tfrecords-format-details)

## Resources

- [TFRecord and tf.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord)
- [Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data)
- [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) API
- [Tensorflow VarLenFeature vs FixedLenFeature](https://stackoverflow.com/questions/41921746/tensorflow-varlenfeature-vs-fixedlenfeature)
- [Tensorflow Records? What they are and how to use them](https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564)

## Structuring TFRecord

A TFRecord file stores your data as a sequence of binary strings. This means you need to specify the structure of your data before you write it to the file. Tensorflow provides two components for this purpose: [tf.train.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example) and [tf.train.SequenceExample](https://www.tensorflow.org/api_docs/python/tf/train/SequenceExample). You have to store each example of your data in one of these structures, then serialize it and use a [tf.io.TFRecordWriter](https://www.tensorflow.org/api_docs/python/tf/io/TFRecordWriter) to write it to disk. _(This document only covers tf.train.Example)_

> tf.train.Example isn’t a normal Python class, but a protocol buffer.

`tf.Example` message (or protobuf) is a flexible message type that represents a `{"string": value}` mapping. It is designed for use with TensorFlow and is used throughout the higher-level APIs such as [TFX](https://www.tensorflow.org/tfx/).

## tf.Example

### Data types for **tf.Example**

Fundamentally, a `tf.Example` is a `{"string": tf.train.Feature}` mapping.

The `tf.train.Feature` message type can accept one of the following three types (See the [`.proto` file](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto) for reference). Most other generic types can be coerced into one of these:

1. [tf.train.BytesList](https://www.tensorflow.org/api_docs/python/tf/train/BytesList) (the following types can be coerced)

- `string`
- `byte`

2. [tf.train.FloatList](https://www.tensorflow.org/api_docs/python/tf/train/FloatList) (the following types can be coerced)

- `float` (`float32`)
- `double` (`float64`)

3. [tf.train.Int64List](https://www.tensorflow.org/api_docs/python/tf/train/Int64List) (the following types can be coerced)

- `bool`
- `enum`
- `int32`
- `uint32`
- `int64`
- `uint64`

In order to convert a standard TensorFlow type to a `tf.Example`-compatible [tf.train.Feature](https://www.tensorflow.org/api_docs/python/tf/train/Feature), you can use the shortcut functions below. Note that each function takes a scalar input value and returns a [tf.train.Feature](https://www.tensorflow.org/api_docs/python/tf/train/Feature) containing one of the three list types above:

```python
# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
```

> **\*Note:** To stay simple, this example only uses scalar inputs. The simplest way to handle non-scalar features is to use `tf.io.serialize_tensor` to convert tensors to binary-strings. Strings are scalars in tensorflow. Use `tf.io.parse_tensor` to convert the binary-string back to a tensor.

```python
# The following functions can be used to convert a `list of values` to a type compatible
# with tf.Example.

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
```

### Creating a tf.Example message

Suppose you want to create a `tf.Example` message from existing data. In practice, the dataset may come from anywhere, but the procedure of creating the `tf.Example` message from a single observation will be the same:

1. Within each observation, each value needs to be converted to a [tf.train.Feature](https://www.tensorflow.org/api_docs/python/tf/train/Feature) containing one of the 3 compatible types, using one of the functions above.

2. You create a map (dictionary) from the feature name string to the encoded feature value produced in #1.

3. The map produced in step 2 is converted to a [`Features` message](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto#L85).

Each of these features can be coerced into a `tf.Example`-compatible type using one of `_bytes_feature`, `_float_feature`, `_int64_feature`. You can then create a `tf.Example` message from these encoded features:

```python
def serialize_example(feature0, feature1, feature2, feature3):
    """Creates a tf.Example message ready to be written to a file."""

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }

    # Creates a Features message using tf.train.Example
    # and serializes to a binary-string

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# >>> serialized_example = serialize_example(False, 4, b'goat', 0.9876)
# b'\nR\n\x14\n\x08feature2\x12\x08\n\x06\n\x04goat\n\x14\n\x08feature3...'
```

To decode the message use the `tf.train.Example.FromString` method.

```python
example_proto = tf.train.Example.FromString(serialized_example)
```

## TFRecords format details

A TFRecord file contains a sequence of records. The file can only be read sequentially. _(Each record contains a byte-string, for the data-payload, plus the data-length, and CRC32C)_

<!-- ## TFRecord files using tf.data -->
