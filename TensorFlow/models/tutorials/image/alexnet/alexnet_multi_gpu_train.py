from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import alexnet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/alexnet_multi_gputrain',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', '../../../../../data/kaggle/cat_vs_dog/train-00000-of-00002',
                           """Directory to TFRecord files""")
tf.app.flags.DEFINE_integer('cycle_length', 32,
                            """Number of datasets to overlap for parallel I/O.""")
tf.app.flags.DEFINE_integer('num_parallel_calls', 32,
                            """Number of CPU cores for the pre-processing.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_epochs', 100000,
                            """Number of steps to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def input_fn(batch_size):
    # Read records from a list of files
    filenames = tf.data.Dataset.list_files(FLAGS.data_dir)

    dataset = filenames.apply(
        tf.contrib.data.parallel_interleave(
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


def tower_loss(scope, images, labels):
    """Calculate the total loss on a single tower running the AlexNet model.

    Args:
      scope: unique prefix string identifying the AlexNet tower, e.g. 'tower_0'
      images: Images. 4D tensor of shape [batch_size, height, width, 3].
      labels: Labels. 1D tensor of shape [batch_size].

    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """

    # Build inference Graph.
    logits = alexnet.inference(images)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = alexnet.loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % alexnet.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    """Train AlexNet for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.train.get_or_create_global_step()

        with tf.device('/cpu:0'):
            dataset = input_fn(FLAGS.batch_size)
            iterator = dataset.make_one_shot_iterator()
            next_examples, next_labels = iterator.get_next()

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (alexnet.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                 FLAGS.batch_size / FLAGS.num_gpus)
        decay_steps = int(num_batches_per_epoch * alexnet.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(alexnet.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        alexnet.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.RMSPropOptimizer(lr)

        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device(f'/gpu:{i}'):
                    with tf.name_scope(f'{alexnet.TOWER_NAME}_{i}') as scope:
                        # Dequeues one batch for the GPU
                        # image_batch, label_batch = batch_queue.dequeue()
                        # Calculate the loss for one tower of the AlexNet model. This function
                        # constructs the entire AlexNet model but shares the variables across
                        # all towers.
                        loss = tower_loss(
                            scope, next_examples['image_data'], next_labels)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(
                            tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this AlexNet tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(
                    var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            alexnet.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                # format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                #               'sec/batch)')
                # print(format_str % (datetime.now(), step, loss_value,
                #                     examples_per_sec, sec_per_batch))
                print(
                    f'{datetime.now()}: step {step}, loss = {loss_value:.2f} '
                    f'({examples_per_sec:.1f} examples/sec; {sec_per_batch:.3f} sec/batch)'
                )

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    tf.app.run()
