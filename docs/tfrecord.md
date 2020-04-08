# tf.data and TFRecord: Build TensorFlow input pipelines <!-- omit in toc -->

To read data efficiently it can be helpful to serialize your data and store it in a set of files (100-200MB each) that can each be read linearly. This is especially true if the data is being streamed over a network. This can also be useful for caching any data-preprocessing.

The TFRecord format is a simple format for storing a sequence of binary records. It is optimized for use with Tensorflow in multiple ways. Especially for datasets that are too large to be stored fully in memory this is an advantage as only the `data that is required at the time (e.g. a batch) is loaded from disk and then processed`.

<!-- `tf.Example` message (or protobuf) is a flexible message type that represents a `{"string": value}` mapping. It is designed for use with TensorFlow and is used throughout the higher-level APIs such as [TFX](https://www.tensorflow.org/tfx/). -->

> **\*Note:** While useful, TFRecord is optional. There is no need to convert existing code to use TFRecords, unless you are using [tf.data](https://www.tensorflow.org/guide/datasets) and reading data is still the bottleneck to training. See [Data Input Pipeline Performance](https://www.tensorflow.org/guide/performance/datasets) for dataset performance tips.

## Contents <!-- omit in toc -->

- [Resources](#resources)
- [Structuring TFRecord](#structuring-tfrecord)
- [tf.Example](#tfexample)
  - [Data types for tf.Example](#data-types-for-tfexample)
  - [Creating a tf.Example message](#creating-a-tfexample-message)
- [TFRecords format details](#tfrecords-format-details)
- [TFRecord files using tf.data](#tfrecord-files-using-tfdata)
  - [Writing a TFRecord file](#writing-a-tfrecord-file)
  - [Reading a TFRecord file](#reading-a-tfrecord-file)
- [Better performance with the tf.data API](#better-performance-with-the-tfdata-api)
  - [The naive approach](#the-naive-approach)
  - [Prefetching](#prefetching)
  - [Parallelize Data Extraction](#parallelize-data-extraction)
  - [Parallelize Data Transformation](#parallelize-data-transformation)
  - [Caching](#caching)
- [Best practice summary](#best-practice-summary)

## Resources

- [TFRecord and tf.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord)
- [Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data)
- [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) API
- [Tensorflow VarLenFeature vs FixedLenFeature](https://stackoverflow.com/questions/41921746/tensorflow-varlenfeature-vs-fixedlenfeature)
- [Tensorflow Records? What they are and how to use them](https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564)
- [Working with TFRecords and tf.train.Example](https://towardsdatascience.com/working-with-tfrecords-and-tf-train-example-36d111b3ff4d)

## Structuring TFRecord

A TFRecord file stores your data as a sequence of binary strings. This means you need to specify the structure of your data before you write it to the file. Tensorflow provides two components for this purpose: [tf.train.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example) and [tf.train.SequenceExample](https://www.tensorflow.org/api_docs/python/tf/train/SequenceExample). You have to store each example of your data in one of these structures, then serialize it and use a [tf.io.TFRecordWriter](https://www.tensorflow.org/api_docs/python/tf/io/TFRecordWriter) to write it to disk. _(This document only covers tf.train.Example)_

> tf.train.Example isn’t a normal Python class, but a protocol buffer.

`tf.Example` message (or protobuf) is a flexible message type that represents a `{"string": value}` mapping. It is designed for use with TensorFlow and is used throughout the higher-level APIs such as [TFX](https://www.tensorflow.org/tfx/).

## tf.Example

### Data types for tf.Example

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

## TFRecord files using tf.data

The [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) module also provides tools for reading and writing data in TensorFlow.

### Writing a TFRecord file

### Reading a TFRecord file

```python
def parse_fn(serialized_example):
    """Parse TFExample records and perform simple data augmentation."""

    features = {
        'image': tf.FixedLengthFeature((), tf.string),
        'label': tf.FixedLengthFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(serialized_example, features)
    image = tf.image.decode_image(parsed['image'])
    return image, parsed['label']

def get_dataset():
    files = tf.data.Dataset.list_files('/path/to/dataset/*.tfrecord')
    dataset = files.interleave(tf.data.TFRecordDataset)
    dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    dataset = dataset.map(parse_fn)
    dataset = dataset.batch(batch_size)
    return dataset
```

## Better performance with the tf.data API

GPUs and TPUs can radically reduce the time required to execute a single training step. Achieving peak performance requires an efficient input pipeline that delivers data for the next step before the current step has finished. The [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) API helps to build flexible and efficient input pipelines. This document demonstrates how to use the [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) API to build highly performant TensorFlow input pipelines.

### The naive approach

Start with a naive pipeline using no tricks, iterating over the dataset as-is. Under the hood, this is how your execution time was spent:

![](images/naive_pipeline.svg)

You can see that performing a training step involves:

- opening a file if it hasn't been opened yet
- fetching a data entry from the file
- using the data for training

However, in a naive synchronous implementation like here, while your pipeline is fetching the data, your model is sitting idle. Conversely, while your model is training, the input pipeline is sitting idle. The training step time is thus the sum of all, opening, reading and training time.

### Prefetching

_Prefetching_ overlaps the preprocessing and model execution of a training step. While the model is executing training step `s`, the input pipeline is reading the data for step `s+1`. Doing so reduces the step time to the maximum (as opposed to the sum) of the training and the time it takes to extract the data.

The tf.data API provides the tf.data.Dataset.prefetch transformation. It can be used to decouple the time when data is produced from the time when data is consumed. In particular, the transformation uses a background thread and an internal buffer to prefetch elements from the input dataset ahead of the time they are requested. The number of elements to prefetch should be equal to (or possibly greater than) the number of batches consumed by a single training step. You could either manually tune this value, or set it to `tf.data.experimental.AUTOTUNE` which will prompt the tf.data runtime to tune the value dynamically at runtime.

```python
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
```

![](images/prefetched.svg)

This time you can see that while the training step is running for sample 0, the input pipeline is reading the data for the sample 1, and so on.

### Parallelize Data Extraction

In a real-world setting, the input data may be stored remotely (for example, GCS or HDFS). A dataset pipeline that works well when reading data locally might become bottlenecked on I/O when reading data remotely because of the following differences between local and remote storage:

- **Time-to-first-byte:** Reading the first byte of a file from remote storage can take orders of magnitude longer than from local storage.
- **Read throughput:** While remote storage typically offers large aggregate bandwidth, reading a single file might only be able to utilize a small fraction of this bandwidth.

To mitigate the impact of the various data extraction overheads, the [tf.data.Dataset.interleave](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave) transformation can be used to parallelize the data loading step, interleaving the contents of other datasets (such as data file readers). The number of datasets to overlap can be specified by the `cycle_length` argument, while the level of parallelism can be specified by the `num_parallel_calls` argument. Similar to the prefetch transformation, the `interleave` transformation supports [tf.data.experimental.AUTOTUNE](https://www.tensorflow.org/api_docs/python/tf/data/experimental#AUTOTUNE) which will delegate the decision about what level of parallelism to use to the [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) runtime.

The following diagram illustrates the effect of supplying `cycle_length=2` to the `parallel_interleave` transformation:

![Parallel_interleave](images/datasets_parallel_io.png)

```python
dataset = dataset.interleave(TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
```

This time, the reading of the two datasets is parallelized, reducing the global data processing time.

### Parallelize Data Transformation

When preparing data, input elements may need to be pre-processed. To this end, the [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) API offers the [tf.data.Dataset.map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map) transformation, which applies a user-defined function to each element of the input dataset. Because input elements are independent of one another, the pre-processing can be parallelized across multiple CPU cores. To make this possible, similarly to the `prefetch` and `interleave` transformations, the map transformation provides the `num_parallel_calls` argument to specify the level of parallelism.

![Parallelize_Transformation](images/datasets_parallel_map.png)

**Note**:

- `num_parallel_calls` depends on hardware, characteristics of training data (such as size and shape), the cost of the map function, and what other processing is happening on the CPU at the same time.
- a simple heuristic is to use the number of available CPU cores

Furthermore, if batch size is in the hundreds or thousands, the pipeline will likely additionally benefit from parallelizing the batch creation.

```python
dataset = dataset.map(parse_fn, num_parasllel_calls=tf.data.experimental.AUTOTUNE)
```

### Caching

The [tf.data.Dataset.cache](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache) transformation can cache a dataset, either in memory or on local storage. This will save some operations (like file opening and data reading) from being executed during each epoch.

![](images/cached_dataset.svg)

```python
# Apply time consuming operations before cache
dataset = dataset.map(parse_fn, num_parasllel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.cache()
```

When you cache a dataset, the transformations before the cache one (like the file opening and data reading) are executed only during the first epoch. The next epochs will reuse the data cached by thecache transformation.

If the user-defined function passed into the `map` transformation is expensive, apply the `cache` transformation after the `map` transformation as long as the resulting dataset can still fit into memory or local storage. If the user-defined function increases the space required to store the dataset beyond the cache capacity, either apply it after the `cache` transformation or consider pre-processing your data before your training job to reduce resource usage.

## Best practice summary

Here is a summary of the best practices for designing performant TensorFlow input pipelines:

- [Use the `prefetch` transformation](#prefetching) to overlap the work of a producer and consumer.
- [Parallelize the data reading transformation](#parallelize-data-extraction) using the `interleave` transformation.
- [Parallelize the `map` transformation](#parallelize-data-transformation) by setting the `num_parallel_calls` argument.
- [Use the `cache` transformation](#caching) to cache data in memory during the first epoch
- [Vectorize user-defined functions](https://www.tensorflow.org/guide/data_performance#Map_and_batch) passed in to the `map` transformation
- [Reduce memory usage](https://www.tensorflow.org/guide/data_performance#Reducing_memory_footprint) when applying the `interleave`, `prefetch`, and `shuffle` transformations.
<!-- - Add `prefetch(n)` (n is the number of elements / batches consumed by a training step) to the end of the input pipeline
- If you are working with data stored remotely and / or requiring deserialization, we recommend using the `parallel_interleave` transformation to overlap the reading (and deserialization) of data from different files
- If the data can fit into memory, use the `cache` transformation to cache it in memory during the first epoch, so that subsequent epochs can avoid the overhead associated with reading, parsing, and transforming it
- I recommend applying the `shuffle` transformation before the `repeat` transformation, ideally using the fused `shuffle_and_repeat` transformation -->

Example code:

```python
def get_dataset(data_dir, batch_size):
    filenames = tf.data.Dataset.list_files(data_dir)

    # dataset = filenames.apply(tf.contrib.data.parallel_interleave(
    #     tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers, sloppy=True))

    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    # In some cases, `head -n20 /path/to/tfrecords` bash commmand is used to
    # find out the feature names of a TFRecord
    def parser(record):
        features = {
            "image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
            "image/class/label": tf.FixedLenFeature((), tf.int64,
                                                    default_value=tf.zeros([], dtype=tf.int64)),
        }
        parsed = tf.parse_single_example(record, features)

        # Perform additional preprocessing on the parsed data.
        image = tf.decode_raw(parsed["image/encoded"], tf.float32)
        image = tf.reshape(image, [224, 224, 3])
        label = tf.cast(parsed["image/class/label"], tf.int32)

        return {"image": image}, label

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.shuffle(buffer_size=10000)

    # Vectorize your mapped function
    dataset = dataset.batch(batch_size)

    # Parallelize map transformation
    dataset = dataset.map(parser, num_parasllel_calls=tf.data.experimental.AUTOTUNE)

    # Cache data
    dataset = dataset.cache()

    # Reduce memory usage
    dataset = dataset.map(memory_consumming_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Overlap producer and consumer works
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Each element of `dataset` is tuple containing a dictionary of features
    # (in which each value is a batch of values for that feature), and a batch of labels.
    return dataset
```
