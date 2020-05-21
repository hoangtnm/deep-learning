#!/usr/bin/env python3

import argparse
import functools
import json
import logging
import multiprocessing
import os
import random
from multiprocessing import Pool
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from research.datasets.utils import bytes_feature, int64_feature

logger = logging.getLogger(__name__)


def extract_frames(path: str, shape: Tuple[int, int],
                   seq_len: int) -> np.ndarray:
    """Extract frames from a video.

    Args:
        path (str): Path to video.
        shape: Shape of extracted frames without channel, e.g. (120, 120)
        seq_len: Number of extracted frames.

    Returns:
        frame_list (np.ndarray): An array of extracted frames.

    Shape:
        - Output: (N_{out}, H_{out}, W_{out}, C_{out})
    """

    frame_list = []
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        # Captures frame-by-frame
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, shape,
                               interpolation=cv2.INTER_LINEAR)
            frame_list.append(frame)
        else:
            break

    # Releases the capture
    cap.release()

    # Video temporal sampling: selects random `seq_len` sequential frames
    frame_list = np.asarray(frame_list)
    # frame_indices = np.linspace(
    #     0, frame_list.shape[0] - 1, seq_len).astype(np.uint)
    frame_indices = np.arange(frame_list.shape[0])
    random.shuffle(frame_indices)
    frame_indices = sorted(frame_indices[:seq_len])
    frame_list = frame_list[frame_indices, :, :, :]
    return frame_list


def serialize_example(video: np.ndarray, label: int):
    """Creates a tf.Example message ready to be written to a file."""

    seq_len = video.shape[0]
    height = video.shape[1]
    width = video.shape[2]
    channels = video.shape[3]
    assert channels == 3, 'Only RGB video format is supported'

    # Transforms a Tensor into a serialized TensorProto proto.
    # TODO: Replaces to.string method from numpy with tf.io.serialize_tensor
    # Note: tf.io.serialize_tensor causes issues when using Keras's model.fit
    #   ValueError: Cannot take the length of shape with unknown rank.
    # video_bytes = tf.io.serialize_tensor(video)
    video_bytes = video.tostring()

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type
    feature = {
        'video': bytes_feature(video_bytes),
        'label': int64_feature(label),
        'seq_len': int64_feature(seq_len),
        'height': int64_feature(height),
        'width': int64_feature(width),
        'channels': int64_feature(channels)
    }

    # Creates a Features message using tf.train.Example
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def get_label_map_dict(label_list: List[str]) -> Dict[str, int]:
    """Returns a dict mapping labels and the corresponding id.

    Args:
        label_list: List of label names.

    Returns:
        A dict mapping labels to the corresponding id.
    """
    return {label: label_id for label_id, label in enumerate(label_list)}


def write_label_file(label_list: List[str], filename: str) -> None:
    """Writes a JSON dict of labels to disk.

    Args:
        label_list: List of label names.
        filename: Path to JSON file.
    """

    label_to_id = get_label_map_dict(label_list)

    with open(filename, 'w') as f:
        json.dump(label_to_id, f, indent=4)


def get_dataset(data_dir: str, label_list: List[str]) -> List[Tuple[str, int]]:
    """Returns a list of examples, each of which is a tuple of path and label.

    Args:
        data_dir: Path to dataset directory.
        label_list: List of label names.

    Returns:
        List of examples. Each example is a tuple of
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


def split_dataset(dataset: List[Tuple[str, int]], num_splits: int) -> List[
    List[Tuple[str, int]]]:
    """Splits dataset to shards.

    Args:
        dataset: List of examples. Each example is a tuple of
            video path, and the corresponding label.
        num_splits: How many shards that the dataset will be splitted to.

    Returns:
        shards: A list of shards. Each shard is a subset of the dataset.
    """

    shards = []

    spaced_indices = np.linspace(
        0, len(dataset), num_splits + 1).astype(np.uint)
    shard_ranges = [[spaced_indices[idx], spaced_indices[idx + 1]]
                    for idx in range(len(spaced_indices) - 1)]

    for shard_idx, (start, end) in enumerate(shard_ranges):
        shard = dataset[start:end]
        shards.append(shard)

    return shards


# def write_tfrecord(shard: List[Tuple[str, int]], filename: str):
def write_tfrecord(work: Tuple[List[Tuple[str, int]], str],
                   shape: Tuple[int, int], seq_len: int):
    """Writes a shard to a TFRecord file.

    Args:
        work: A tuple of dataset shard, filename, shape and frames.
            shard: List of examples.
                Each example is a tuple of `path` and `label`.
            filename: Path to the TFRecord file.
        shape: Shape of extracted frames without channel, e.g. (120, 120)
        seq_len: Number of extracted frames.
    """

    shard, filename = work

    with tf.io.TFRecordWriter(filename) as writer:
        for video_path, label in shard:
            video = extract_frames(video_path, shape, seq_len)
            writer.write(serialize_example(video, label))


# def process_dataset(dataset: List[Tuple[str, int]], num_shards: int,
#                     num_threads: int, phase: str, output_dir: str):
#     assert num_shards % num_threads == 0

#     # Splits dataset N shards for each thread.
#     # At this stage, N will be equal to num_threads.
#     dataset = split_dataset(dataset, num_splits=num_threads)

#     # Coordinates the termination of a set of threads.
#     coord = tf.train.Coordinator()
#     threads = []

#     shards_per_thread = num_shards // num_threads

#     for idx, thread_idx in enumerate(range(num_threads)):
#         filenames = [
#             os.path.join(
#                 output_dir,
#                 f'{phase}-{thread_idx * shards_per_thread + shard_idx}-of-{num_shards}'
#             ) for shard_idx in range(shards_per_thread)
#         ]

#         sub_dataset = dataset[idx]
#         t = Thread(target=write_tfrecord, args=(sub_dataset, filenames))
#         t.start()
#         threads.append(t)

#     # Waits for all the threads to terminate.
#     coord.join(threads)


def get_filenames(output_dir: str, phase: str, num_shards: int) -> List[str]:
    """Returns a list of TFRecord filenames.

    Args:
        output_dir: Path to the directory where TFRecord files will be written.
        phase: type of output file, e.g. `train` or `val`
        num_shards: Number of shards.

    Returns:
        filenames: A list of TFRecord filenames.
    """

    filenames = []

    for idx in range(num_shards):
        filename = f'{phase}.tfrecord-{idx:05d}-of-{num_shards:05d}'
        filename = os.path.join(output_dir, filename)
        filenames.append(filename)

    return filenames


def process_dataset_v2(shards: List[List[Tuple[str, int]]],
                       filenames: List[str],
                       shape: Tuple[int, int],
                       seq_len: int,
                       num_processes: int):
    """Writes dataset shards to TFRecord files.

    Args:
        shards: A list of shards, each of which is a list of examples.
            Each example is a tuple of video path and the corresponding label.
        filenames: A list of TFRecord filenames.
        shape: Shape of extracted frames without channel, e.g. (120, 120)
        seq_len: Number of extracted frames.
        num_processes: Number of CPU cores to process in parallel.
    """

    work = tuple(zip(shards, filenames))

    with Pool(num_processes) as pool:
        pool.map(
            functools.partial(write_tfrecord, shape=shape, seq_len=seq_len),
            work)


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
    parser.add_argument(
        '--frame_shape',
        default=[112, 112],
        nargs='+',
        type=int,
        help='Shape of extracted frames from each video',
        required=False
    )
    parser.add_argument(
        '--seq_len',
        default=16,
        type=int,
        help='Number of extracted frames from each video',
        required=False
    )
    # parser.add_argument(
    #     '--label_map',
    #     default='label_map.txt',
    #     help='Path to JSON-based label map',
    #     required=True)
    parser.add_argument(
        '--num_train_shards',
        default=64,
        type=int,
        help='Number of shards for training',
        required=False)
    parser.add_argument(
        '--num_val_shards',
        default=64,
        type=int,
        help='Number of shards for validation',
        required=False)
    parser.add_argument(
        '--num_cpu',
        default=2,
        type=int,
        help='Number of CPUs to process in parallel',
        required=False
    )

    args = parser.parse_args()
    args.frame_shape = tuple(args.frame_shape)

    assert os.path.exists(args.input_dir)
    assert args.num_train_shards >= args.num_cpu
    assert args.num_val_shards >= args.num_cpu

    if args.num_cpu != 1:
        assert args.num_cpu <= multiprocessing.cpu_count()
        assert args.num_cpu % 2 == 0, 'Number of CPU shoule be divisible by 2'

    label_list = sorted(os.listdir(args.input_dir))

    # Creates output_dir if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Writes JSON-based label_map file
    label_map_file = os.path.join(args.output_dir, 'label_map.txt')
    write_label_file(label_list, label_map_file)

    logging.basicConfig(level=logging.INFO)

    logger.info('***** Creating TFRecords *****')
    logger.info(f'  Num labels = {len(label_list)}')
    logger.info(f'  Num CPUs = {args.num_cpu}')

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

    dataset = get_dataset(args.input_dir, label_list)
    trainset, valset = train_test_split(dataset, test_size=0.2)
    dataset_dict = {'train': trainset, 'val': valset}

    for phase in ['train', 'val']:
        num_shards = args.num_train_shards if phase == 'train' \
            else args.num_val_shards

        # Split to shards.
        dataset_shards = split_dataset(dataset_dict[phase], num_shards)

        # Corresponding filenames of each shard.
        filenames = get_filenames(args.output_dir, phase, num_shards)

        # Writes TFRecords files.
        process_dataset_v2(dataset_shards, filenames,
                           args.frame_shape, args.seq_len, args.num_cpu)


if __name__ == "__main__":
    main()
