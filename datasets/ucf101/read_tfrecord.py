#!/usr/bin/env python3

# from functools import partial

import tensorflow as tf


def parse_fn(serialized_example):
    """Parse TFExample records and perform simple data augmentation."""

    features = {
        'video': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    # Parse the input tf.Example proto using the dictionary above.
    parsed = tf.io.parse_single_example(serialized_example, features)
    return parsed


# def extract_frames(example, frame_sampling: int):
def extract_frames(example):
    """Extract frames from video.

    Args:
        example: A dict of video and label.
        frame_sampling: number of frames to sample.

    Returns:
        video, label: preprocessed batch.
    """

    video = tf.io.parse_tensor(example['video'], tf.uint8)
    # num_frames = video.shape[0]
    # frame_max = num_frames - frame_sampling
    # frame_start = tf.random.uniform(
    #     shape=[], maxval=frame_max, dtype=tf.int32
    # )

    # # Slices N frames starting from frame_start
    # video = video[frame_start:frame_start+frame_sampling, :, :, :]
    example['video'] = video
    return example


def normalize_img(example):
    example['video'] = tf.cast(example['video'], tf.float32) / 255
    example['label'] = tf.cast(example['label'], tf.float32)
    return example


def split_properties(example):
    return example['video'], example['label']


def get_dataset(pattern, batch_size):
    dataset = tf.data.Dataset.list_files(pattern)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # dataset = dataset.shuffle(tf.data.experimental.AUTOTUNE)
    # dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        extract_frames, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(
        split_properties, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    pass
