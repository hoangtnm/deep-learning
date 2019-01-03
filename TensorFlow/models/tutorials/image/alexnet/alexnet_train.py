import os

from datetime import datetime
import time

import tensorflow as tf
import alexnet

# Uncomment to use CPU instead of GPU(s)
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '../../../../../data/kaggle/cat_vs_dog/train.tfrecord',
                           """Directory to TFRecord files""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/alexnet_train',
                           """Directory where to write event logs"""
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('cycle_length', 4,
                            """Number of datasets to overlap for parallel I/O.""")
tf.app.flags.DEFINE_integer('num_parallel_calls', 4,
                            """Number of CPU cores for the pre-processing.""")
tf.app.flags.DEFINE_integer('max_step', 100000,
                            """Number of steps to run.""")
tf.app.flags.DEFINE_integer('num_epochs', 100000,
                            """Number of steps to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def input_fn(batch_size):
    # Read records from a list of files
    filenames = tf.data.Dataset.list_files(FLAGS.data_dir)

    dataset = filenames.apply(
        tf.data.experimental.parallel_interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length=FLAGS.cycle_length, sloppy=True))

    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    # In some cases, `head -n20 /path/to/tfrecords` bash commmand is used to
    # find out the feature names of a TFRecord
    def parser_fn(record):
        features = {
            "image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
            "image/class/label": tf.FixedLenFeature((), tf.int64,
                                                    default_value=tf.zeros([], dtype=tf.int64)),
        }
        parsed = tf.parse_single_example(record, features)

        # Perform additional preprocessing on the parsed data.
        image_decoded = tf.image.decode_jpeg(parsed["image/encoded"])
        image_resized = tf.image.resize_image_with_pad(
            image_decoded, target_height=224, target_width=224)
        image_resized = tf.cast(image_resized, tf.float32)
        label = tf.cast(parsed["image/class/label"], tf.int32)

        return {'image_data': image_resized}, label

    # Shuffles and repeats a Dataset returning a new permutation for each epoch
    dataset = dataset.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=10000, count=FLAGS.num_epochs)
    )

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    # Parse string values into tensors
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(map_func=parser_fn, batch_size=batch_size, drop_remainder=True)
    )

    # Use prefetch() to overlap the producer and consumer
    dataset = dataset.prefetch(2)

    # Each element of `dataset` is tuple containing a dictionary of features
    # (in which each value is a batch of values for that feature), and a batch of
    # labels.
    return dataset


def train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        with tf.device('/cpu:0'):
            dataset = input_fn(FLAGS.batch_size)
            iterator = dataset.make_one_shot_iterator()
            next_examples, next_labels = iterator.get_next()

        # Build a Graph computing logits prediction from the
        # inference model
        logits = alexnet.inference(next_examples['image_data'])

        # Calculate loss
        loss = alexnet.loss(logits, next_labels)

        # Build a Graph training the model with one batch of examples and
        # updating the model parameters
        train_op = alexnet.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime"""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    example_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    print(
                        f'{datetime.now()}: step {self._step}, loss = {loss_value:.2f} '
                        f'({example_per_sec:.1f} examples/sec; {sec_per_batch:.3f} sec/batch)'
                    )
                    
        # Automatically initializes and/or restores variables before returning
        # MonitoredSession.run() automatically recovers from PS failure,
        # and can run additional code in hooks
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_step),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(log_device_placement=False,
                                  gpu_options=tf.GPUOptions(allow_growth=True))) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(unused_argv):
    train()


if __name__ == '__main__':
    tf.app.run()
