import os

from datetime import datetime
import time

import tensorflow as tf
import alexnet_bench as alexnet

# Uncomment to use GPU(s) instead of CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '../../../../../data/kaggle/cat_vs_dog/train.tfrecord',
                           """Directory to TFRecord files""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs"""
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('NUM_EPOCHS', 100000,
                            """Number of epochs to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def input_fn(batch_size=32):
    # Read records from a list of files
    filenames = tf.data.Dataset.list_files(FLAGS.data_dir)

    dataset = filenames.apply(
        tf.contrib.data.parallel_interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length=4, sloppy=True))

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
        image_decoded = tf.decode_raw(parsed["image/encoded"], tf.float32)
        image_resized = tf.reshape(image_decoded, [224, 224, 3])
        label = tf.cast(parsed["image/class/label"], tf.int32)

        #return {"image_data": image}, label
        return image_resized, label

    # Randomly shuffle using a buffer of 10000 examples
    dataset = dataset.shuffle(buffer_size=10000)
    
    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    # Parse string values into tensors
    dataset = dataset.map(parser, num_parallel_calls=4)
    
    # Repeat for FLAGS.NUM_EPOCHS epochs
    dataset = dataset.repeat(FLAGS.NUM_EPOCHS)

    # Combine batch_size consecutive elements into a batch
    dataset = dataset.batch(batch_size, drop_remainder=True)

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
            dataset = input_fn()
            iterator = dataset.make_one_shot_iterator()
            next_examples, next_labels = iterator.get_next()
            # next_example, next_label = next_element['image_resized'], next_element['label']

        # Build a Graph computing logits prediction from the
        # inference model
        logits = alexnet.inference(next_examples)

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

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')

                    print(format_str % (datetime.now(), self._step,
                                        loss_value, example_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.NUM_EPOCHS),
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
