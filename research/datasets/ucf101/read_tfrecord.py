#!/usr/bin/env python3

import tensorflow as tf


def parse_fn(serialized_example):
    """Parse TFExample records and perform simple data augmentation."""

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

    # Decodes and reshapes video
    seq_len = tf.cast(parsed['seq_len'], tf.uint32)
    height = tf.cast(parsed['height'], tf.uint32)
    width = tf.cast(parsed['width'], tf.uint32)
    channels = tf.cast(parsed['channels'], tf.uint32)
    video = tf.io.decode_raw(parsed['video'], tf.uint8)
    video = tf.reshape(video, shape=[seq_len, height, width, channels])

    # Normalizes video frames, label
    video = tf.cast(video, tf.float32) / 255
    label = tf.cast(parsed['label'], tf.float32)
    return video, label


def get_dataset(pattern, batch_size):
    dataset = tf.data.Dataset.list_files(pattern)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # dataset = dataset.shuffle(buffer_size)
    # dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    pass
