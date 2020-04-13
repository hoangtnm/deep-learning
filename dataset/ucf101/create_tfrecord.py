#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random
from threading import Thread
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from research.utils import dataset_util

logger = logging.getLogger(__name__)


def video_to_tensor(path: str) -> tf.Tensor:
    """Creates a tensor of frames from a video.

    Args:
        path (str): Path to video.

    Returns:
        video_tensor (tf.Tensor): Tensor of frames.

    Shape:
        - Output: (N, H_{out}, W_{out}, C_{out})
    """

    frame_list = []
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        # Captures frame-by-frame
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame)
        else:
            break

    # Releases the capture
    cap.release()

    video_tensor = tf.convert_to_tensor(frame_list, dtype=tf.uint8)
    return video_tensor


def serialize_example(video_tensor: tf.Tensor, label: int):
    """Creates a tf.Example message ready to be written to a file."""

    # frames = video_tensor.shape[0]
    # height = video_tensor.shape[1]
    # width = video_tensor.shape[2]
    channels = video_tensor.shape[3]
    assert channels == 3, 'Only RGB video format is supported'

    # Transforms a Tensor into a serialized TensorProto proto.
    video_bytes = tf.io.serialize_tensor(video_tensor)

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type
    feature = {
        'video': dataset_util.bytes_feature(video_bytes),
        'label': dataset_util.int64_feature(label),
        # 'frames': dataset_util.int64_feature(frames),
        # 'height': dataset_util.int64_feature(height),
        # 'width': dataset_util.int64_feature(width),
        # 'channels': dataset_util.int64_feature(channels)
    }

    # Creates a Features message using tf.train.Example
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def get_label_map_dict(label_list: List[str]) -> Dict[str, int]:
    """Returns a dict mapping labels and the corresponding ids.

    Args:
        label_list: List of label names.

    Returns:
        label_map_dict: A dict mapping labels to the corresponding ids.
    """

    label_map_dict = {label: label_id
                      for label_id, label in enumerate(label_list)}
    return label_map_dict


def create_label_file(label_list: List[str], label_file: str) -> None:
    """Creates a JSON dict of labels.

    Args:
        label_list: List of label names.
        label_file: Path to write JSON file.
    """

    label_to_id = get_label_map_dict(label_list)
    with open(label_file, 'w') as f:
        json.dump(label_to_id, f, indent=4)


def get_dataset(data_dir: str, label_list: List[str]) -> List[Tuple[str, int]]:
    """Returns list of examples, which is a tuple of path and label.

    Args:
        data_dir: Path to dataset directory.
        label_list: List of label names.

    Returns:
        List of video info, which is a tuple of
            video path, and the corresponding label.
    """

    labels = []
    video_paths = []

    label_to_id = get_label_map_dict(label_list)

    for label in label_list:
        video_pattern = f'{data_dir}/{label}/*'
        matching_files = tf.io.gfile.glob(video_pattern)

        label_id = label_to_id[label]
        labels.extend([label_id] * len(matching_files))
        video_paths.extend(matching_files)

    # Shuffle the ordering of all video files in order to guarantee
    # random ordering of the video with respect to label in the
    # saved TFRecord files.
    shuffled_indices = list(range(len(video_paths)))
    random.shuffle(shuffled_indices)

    labels = [labels[idx] for idx in shuffled_indices]
    video_paths = [video_paths[idx] for idx in shuffled_indices]
    return list(zip(video_paths, labels))


def split_dataset(dataset: List[Tuple[str, int]], num_splits: int) -> List[List[Tuple[str, int]]]:
    """Splits dataset into shards.

    Args:
        dataset: List of examples. Each example is a tuple of path and label.
        num_splits: How many subsets that the dataset will be splitted into.

    Returns:
        shards: A list of shards. Each shard is a subset of the dataset.
    """

    shards = []

    spaced_indices = np.linspace(
        0, len(dataset), num_splits+1).astype(np.uint)
    shard_ranges = [[spaced_indices[idx], spaced_indices[idx+1]]
                    for idx in range(len(spaced_indices) - 1)]

    for shard_idx, (start, end) in enumerate(shard_ranges):
        shard = dataset[start:end]
        shards.append(shard)

    return shards


def write_tfrecord(thread_shard: List[Tuple[str, int]], filenames: List[str]):
    """Writes shards into TFRecords.

    Args:
        thread_shard: List of examples.
            Each example is a tuple of `path` and `label`.
        filenames: List of TFRecord filenames.
    """

    # Splits the dataset into shards.
    shards = split_dataset(thread_shard, num_splits=len(filenames))

    for idx, path in enumerate(filenames):
        writer = tf.io.TFRecordWriter(path)

        for video_path, label in shards[idx]:
            video_tensor = video_to_tensor(video_path)
            writer.write(serialize_example(video_tensor, label))

        writer.close()


def process_dataset(dataset: List[Tuple[str, int]], num_shards: int,
                    num_threads: int, phase: str, output_dir: str):

    assert num_shards % num_threads == 0

    # Splits dataset N shards for each thread.
    # At this stage, N will be equal to num_threads.
    dataset = split_dataset(dataset, num_splits=num_threads)

    # Coordinates the termination of a set of threads.
    coord = tf.train.Coordinator()
    threads = []

    shards_per_thread = num_shards // num_threads

    for idx, thread_idx in enumerate(range(num_threads)):
        filenames = [
            os.path.join(
                output_dir,
                f'{phase}-{thread_idx*shards_per_thread+shard_idx}-of-{num_shards}'
            ) for shard_idx in range(shards_per_thread)
        ]

        sub_dataset = dataset[idx]
        t = Thread(target=write_tfrecord, args=(sub_dataset, filenames))
        t.start()
        threads.append(t)

    # Waits for all the threads to terminate.
    coord.join(threads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        help='The UCF101 dataset directory',
        required=True)
    parser.add_argument(
        '--output_dir',
        help='The output directory where the TFRecord files will be written',
        required=True)
    # parser.add_argument(
    #     '--label_map',
    #     default='label_map.txt',
    #     help='Path to JSON-based label map',
    #     required=True)
    parser.add_argument(
        '--num_train_shards',
        default=32,
        help='Number of shards for training',
        required=False)
    parser.add_argument(
        '--num_val_shards',
        default=32,
        help='Number of shards for validation',
        required=False)
    parser.add_argument(
        '--num_threads',
        default=4,
        help='Number of threads to process in parallel',
        required=False
    )

    args = parser.parse_args()

    assert os.path.exists(args.input_dir)

    label_list = sorted(os.listdir(args.input_dir))
    label_map_file = os.path.join(args.output_dir, 'label_map.txt')
    create_label_file(label_list, label_map_file)

    logging.basicConfig(level=logging.INFO)

    logger.info('***** Creating TFRecords *****')
    logger.info(f'  Num labels = {len(label_list)}')
    logger.info(f'  Num threads = {args.num_threads}')

    # # Non-threaded method
    # writer = tf.io.TFRecordWriter(
    #     os.path.join(args.output_dir, 'train.tfrecord'))

    # for label in label_list:
    #     video_folder = os.path.join(args.input_dir, label)
    #     video_list = os.listdir(video_folder)

    #     # Transforms each video into a serialized TensorProto proto
    #     # and writes to TFRecords
    #     for video in video_list:
    #         video_path = os.path.join(video_folder, video)
    #         video_tensor = video_to_tensor(video_path)
    #         label_int = label_to_id[label]
    #         writer.write(serialize_example(video_tensor, label_int))

    # writer.close()
    # # Non-threaded method

    # Threaded method
    dataset = get_dataset(args.input_dir, label_list)
    trainset, valset = train_test_split(dataset, test_size=0.2)
    process_dataset(trainset, args.num_train_shards, args.num_threads,
                    'train', args.output_dir)
    process_dataset(valset, args.num_val_shards, args.num_threads,
                    'val', args.output_dir)
    # Threaded method


if __name__ == "__main__":
    main()
