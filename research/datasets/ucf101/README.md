# UCF101 Dataset <!-- omit in toc -->

## Contents <!-- omit in toc -->

- [Overview](#overview)
- [Download the dataset](#download-the-dataset)
- [Creating TFRecord files](#creating-tfrecord-files)
- [Reading TFRecord files](#reading-tfrecord-files)

## Overview

UCF101 is an action recognition data set of realistic action videos, collected from YouTube, having 101 action categories. This data set is an extension of UCF50 data set which has 50 action categories.

The videos in 101 action categories are grouped into 25 groups, where each group can consist of 4-7 videos of an action. The videos from the same group may share some common features, such as similar background, similar viewpoint, etc.

![](https://www.crcv.ucf.edu/data/UCF101/UCF101.jpg)

**Citation**

```
@article{DBLP:journals/corr/abs-1212-0402,
  author    = {Khurram Soomro and
               Amir Roshan Zamir and
               Mubarak Shah},
  title     = { {UCF101:} {A} Dataset of 101 Human Actions Classes From Videos in
               The Wild},
  journal   = {CoRR},
  volume    = {abs/1212.0402},
  year      = {2012},
  url       = {http://arxiv.org/abs/1212.0402},
  archivePrefix = {arXiv},
  eprint    = {1212.0402},
  timestamp = {Mon, 13 Aug 2018 16:47:45 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1212-0402},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Download the dataset

```bash
bash download_dataset.sh
```

## Creating TFRecord files

To use UCF-101 dataset in TensorFlow, it is required to convert it into [TFRecord file format](../../../docs/tfrecord.md).

The following command will create a label map with the dataset. This label map defines a mapping from string class names to integer class id. Moreover, the dataset will be also shared into multiple files for performance purposes:

- [tf.data](https://www.tensorflow.org/guide/data) API can read input examples in parallel improving throughput.
- [tf.data](https://www.tensorflow.org/guide/data) API can shuffle the examples better with sharded files which improves performance of the model slightly.

```sh
DATASET_DIR={path to UCF-101 directory}
OUTPUT_DIR={path to output directory}
SEQ_LEN=16
NUM_TRAIN_SHARDS=64
NUM_VAL_SHARDS=64
NUM_CPU=8

python creating_tfrecord.py \
    --input_dir $DATASET_DIR \
    --output_dir $OUTPUT_DIR \
    --frame_shape 112 112 \
    --seq_len $SEQ_LEN \
    --num_train_shards $NUM_TRAIN_SHARDS \
    --num_val_shards $NUM_VAL_SHARDS \
    --num_cpu $NUM_CPU
```

## Reading TFRecord files

This section demonstrates how to read created TFRecord files and use them for training a Keras model.

> You can read `read_tfrecord.py` file for more underlying techniques to process TFRecord files such as parsing, mapping, batching, etc.

```python
from research.datasets.ucf101.read_tfrecord import get_dataset


EPOCHS = 1
BATCH_SIZE = 8

train_dataset = get_dataset('train.tfrecord*', BATCH_SIZE)
val_dataset = get_dataset('val.tfrecord*', BATCH_SIZE)

...
model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)
```
