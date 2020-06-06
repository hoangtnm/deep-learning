import tensorflow as tf
from functools import partial


def parse_fn(serialized_example, processors=None):
    """Parse TFExample records and perform simple data augmentation.

    Args:
        serialized_example: Serialized `tf.Example` message.
        processors: A list of functions to preprocess video.
    """
    features = {
        'video': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'seq_len': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channels': tf.io.FixedLenFeature([], tf.int64)
    }

    # Parse the input tf.Example proto using the dictionary above.
    parsed = tf.io.parse_single_example(serialized_example, features)

    # Decodes and reshapes video.
    seq_len = tf.cast(parsed['seq_len'], tf.uint32)
    height = tf.cast(parsed['height'], tf.uint32)
    width = tf.cast(parsed['width'], tf.uint32)
    channels = tf.cast(parsed['channels'], tf.uint32)
    video = tf.io.decode_raw(parsed['video'], tf.uint8)
    video = tf.reshape(video, shape=[seq_len, height, width, channels])

    # Preprocess video frames, labels.
    label = tf.cast(parsed['label'], tf.float32)
    video = tf.cast(video, tf.float32)

    if processors is not None:
        for processor in processors:
            video = processor(video)
    else:
        video = video / 255

    return video, label


def load_data(pattern: str, batch_size: int,
              processors=None) -> tf.data.Dataset:
    """Loads [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php).

    Args:
        pattern: A string containing path specification.
        batch_size: Number of samples per gradient update.
        processors: A list of functions to preprocess video.

    Returns:
        A `tf.data.Dataset` instance.

    Examples:

    >>> dataset = research.datasets.ucf101.load_data(
    ... pattern='./*.tfrecord', batch_size=8)
    """
    dataset = tf.data.Dataset.list_files(pattern)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # dataset = dataset.shuffle(buffer_size)
    # dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        partial(parse_fn, processors=processors),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
