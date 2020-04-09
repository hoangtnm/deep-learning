#!/usr/bin/env python3

import argparse
import json
import logging
import os
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple

import cv2
import tensorflow as tf

from research.utils import dataset_util

logger = logging.getLogger(__name__)


def video_to_tensor(path: str):
    """Creates a tensor of frames from a video.

    Args:
        path (str): path to video.

    Returns:
        video_tensor (tf.Tensor): tensor of frames.

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


def serialize_example(video_tensor, label: int):
    """Creates a tf.Example message ready to be written to a file."""

    frames = video_tensor.shape[0]
    height = video_tensor.shape[1]
    width = video_tensor.shape[2]
    channels = video_tensor.shape[3]

    # Transforms a Tensor into a serialized TensorProto proto.
    video_bytes = tf.io.serialize_tensor(video_tensor)

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type
    feature = {
        'video': dataset_util.bytes_feature(video_bytes),
        'label': dataset_util.int64_feature(label),
        'frames': dataset_util.int64_feature(frames),
        'height': dataset_util.int64_feature(height),
        'width': dataset_util.int64_feature(width),
        'channels': dataset_util.int64_feature(channels)
    }

    # Creates a Features message using tf.train.Example
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def get_label_map_dict(class_list: List[str]) -> Dict[str, int]:
    """Returns a dict mapping classes and the corresponding ids.

    Args:
        class_list: List of classes.

    Returns:
        label_map_dict: A dict mapping classes to the corresponding ids.
    """

    label_map_dict = {class_str: class_id
                      for class_id, class_str in enumerate(class_list)}
    return label_map_dict


def create_label_file(class_list: List[str], label_file: str):
    """Creates a JSON dict of classes.

    Args:
        class_list: list of classes.
        label_file: path to write JSON file.
    """

    class_to_id = get_label_map_dict(class_list)
    with open(label_file, 'w') as f:
        json.dump(class_to_id, f, indent=4)


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
        '--label_map',
        default='label_map.txt',
        help='Path to JSON-based label map',
        required=True)
    parser.add_argument(
        '--num_threads',
        default=1,
        help='Number of threads to process in parallel',
        required=False
    )

    args = parser.parse_args()

    writer = tf.io.TFRecordWriter(
        os.path.join(args.output_dir, 'train.tfrecord'))
    class_list = sorted(os.listdir(args.input_dir))
    class_to_id = get_label_map_dict(class_list)
    create_label_file(class_list, args.label_map)

    logger.info('***** Creating TFRecords *****')
    logger.info(f'  Num classes = {len(class_list)}')

    for class_label in class_list:
        video_folder = os.path.join(args.input_dir, class_label)
        video_list = os.listdir(video_folder)

        # Transforms each video into a serialized TensorProto proto
        # and writes to TFRecords
        for video in video_list:
            video_path = os.path.join(video_folder, video)
            video_tensor = video_to_tensor(video_path)
            label_int = class_to_id[class_label]
            writer.write(serialize_example(video_tensor, label_int))

    writer.close()


if __name__ == "__main__":
    main()
