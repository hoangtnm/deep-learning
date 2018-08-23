#!/usr/bin/python

import tensorflow as tf

def input_fn(pipeline):
    files = tf.data.Dataset.list_files(FLAGS.data.dir)
    """
    TFRecordDataset uses tf.contrib.data.parallel_interleave under the hood
    """
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=32)
    dataset = dataset.shuffle(10000)                        # Sliding windows of 2048 records
    dataset = dataset.repeat(NUM_EPOCHS)
    dataset = dataset.map(parser_fn, num_parallel_calls=64) #num_parallel_calls=`nproc`
    dataset = dataset.batch(batch_size)
    """
    Prefetch ensures everything above is pipelined with the GPU for training
    Larger prefetch buffers also smooth over latency variability
    """
    dataset = dataset.prefetch(2)

    return dataset

def input_fn(batch_size):
    files = tf.data.Dataset.list_files(FLAGS.data_dir)

    def tfrecord_dataset(filename):
        buffer_size = 8 * 1024 * 1024 # 8MB/file
        return tf.data.TFRecordDataset(filename, buffer_size=buffer_size) 

    dataset = files.apply(tf.contrib.data.parallel_interleave(
        tfrecord_dataset, cycle_length=32, sloppy=True))
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000, NUM_EPOCHS))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(parser_fn,
                            batch_size, num_parallel_batches=4))
    dataset = dataset.prefetch(4)

    return dataset