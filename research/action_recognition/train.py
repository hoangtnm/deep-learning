#!/usr/bin/env python3

import argparse
import logging
import os
from datetime import datetime
from typing import Dict, Union

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        TensorBoard)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from research.datasets.ucf101.read_tfrecord import get_dataset
from research.action_recognition.models import TFC3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_eagerly(model: Model, train_dataset, val_dataset,
                  optimizer, epochs: int, log_dir: str):
    """Trains the model for a fixed number of epochs (iterations on a dataset).

    Args:
        model: A Keras model instance.
        train_dataset: A `tf.data` dataset. Should return a tuple
            of `(inputs, labels)`
        val_dataset: A `tf.data` dataset on which to evaluate
            the loss and any metrics at the end of each epoch.
            Should return a tuple of `(inputs, labels)`
        optimizer: A Keras optimizer instance.
        epochs: Number of epochs to train the model.
        log_dir: Path to the directory where TensorBoard logs will be written.

    Returns:
        model: A Keras trained model.
    """

    criterion = SparseCategoricalCrossentropy()

    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, 'train'))
    val_summary_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, 'validation'))

    # Defines metrics for logging to TensorBoard.
    train_loss = Mean('train_loss', dtype=tf.float32)
    train_accuracy = SparseCategoricalAccuracy('train_accuracy')
    val_loss = Mean('val_loss', dtype=tf.float32)
    val_accuracy = SparseCategoricalAccuracy('val_accuracy')

    for epoch in range(epochs):
        # For human-readability purposes,
        # epoch logging starts from 1 rather than 0.
        print(f'Epoch: {epoch + 1}/{epochs + 1}')

        # Training
        for batch, (inputs, labels) in enumerate(train_dataset):
            train_step(model, optimizer, criterion, inputs, labels,
                       train_loss, train_accuracy)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        # Validation
        for batch, (inputs, labels) in enumerate(val_dataset):
            val_step(model, criterion, inputs, labels, val_loss, val_accuracy)
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)

        template = 'loss: {:.4f} - accuracy: {:.4f} - ' \
                   'val_loss: {:.4f} - val_accuracy: {:.4f}'
        print(template.format(train_loss.result(),
                              train_accuracy.result() * 100,
                              val_loss.result(),
                              val_accuracy.result() * 100))

        # Reset metrics every epoch
        train_loss.reset_states()
        val_loss.reset_states()
        train_accuracy.reset_states()
        val_accuracy.reset_states()

    return model


@tf.function
def train_step(model: Model, optimizer, criterion, inputs, labels,
               train_loss, train_accuracy):
    """Performs one training step."""

    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = criterion(labels, predictions)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def val_step(model: Model, criterion, inputs, labels,
             test_loss, test_accuracy):
    """Performs one validation step."""

    predictions = model(inputs)
    loss = criterion(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)


def train_keras(train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset,
                dropout: float, learning_rate: float, optimizer: str,
                epochs: int, trial_dir: str,
                hparams: Dict[hp.HParam, Union[str, float]]):
    """Train model using Keras style.

    Args:
        train_dataset: A `tf.data` dataset. Should return a tuple
            of `(inputs, labels)`.
        val_dataset: A `tf.data` dataset on which to evaluate
            the loss and any metrics at the end of each epoch.
            Should return a tuple of `(inputs, labels)`.
        dropout: Float between 0 and 1. Fraction of the input units to drop.
        learning_rate: The learning rate.
        optimizer: Name of optimizer.
        epochs: Number of epochs to train the model.
        trial_dir: The directory where logs will be written.
        hparams: A Dict of `hp.HParam` key and the corresponding value.

    Returns:
        accuracy: The best accuracy on the validation_dataset.
    """

    model = TFC3D(num_classes=101, dropout=dropout)

    if optimizer == 'sgd':
        optimizer = SGD(learning_rate)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3),
        TensorBoard(trial_dir),
        # hp.KerasCallback(trial_dir, hparams)
    ]

    model.fit(train_dataset, validation_data=val_dataset,
              epochs=epochs, callbacks=callbacks)
    _, accuracy = model.evaluate(val_dataset)
    return accuracy


def write_trial_log(metric: str, accuracy: float,
                    hparams: Dict[hp.HParam, Union[str, float]],
                    trial_dir: str):
    """Writes trial logs to directory.
    Used for hyperparameter tuning.

    Args:
        metric: Name of desired metric.
        accuracy: Value of the metric.
        hparams: A Dict of `hp.HParam` key and the corresponding value.
        trial_dir: Path to the directory where logs will be written.
    """

    with tf.summary.create_file_writer(trial_dir).as_default():
        # record the values used in this trial.
        hp.hparams(hparams)
        tf.summary.scalar(metric, accuracy, step=1)


def hparam_tuning(train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset,
                  epochs: int, log_dir: str):
    """Performs hyperparameter tuning.

    Args:
        train_dataset: A `tf.data` dataset. Should return a tuple
            of `(inputs, labels)`.
        val_dataset: A `tf.data` dataset on which to evaluate
            the loss and any metrics at the end of each epoch.
            Should return a tuple of `(inputs, labels)`.
        epochs: Number of epochs to train the model.
        log_dir: The directory where logs will be written.
    """

    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1, 0.2, 0.5]))
    HP_LEARNING_RATE = hp.HParam('learning_rate',
                                 hp.Discrete([1e-3, 1e-4, 1e-5]))
    HP_OPTIMIZER = hp.HParam('optimizer',
                             hp.Discrete(['sgd', 'rmsprop', 'adam']))

    METRIC_ACCURACY = 'accuracy'

    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams_config(
            hparams=[HP_DROPOUT, HP_LEARNING_RATE, HP_OPTIMIZER],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')]
        )

    trial_step = 0

    for dropout in HP_DROPOUT.domain.values:
        for learning_rate in HP_LEARNING_RATE.domain.values:
            for optimizer in HP_OPTIMIZER.domain.values:
                hparams = {
                    HP_DROPOUT: dropout,
                    HP_LEARNING_RATE: learning_rate,
                    HP_OPTIMIZER: optimizer,
                }

                trial_id = f'run-{trial_step}'
                trial_dir = os.path.join(log_dir, trial_id)
                logging.info(f'--- Starting trial: {trial_id}')
                logging.info({h.name: hparams[h] for h in hparams})

                accuracy = train_keras(train_dataset, val_dataset, dropout,
                                       learning_rate, optimizer, epochs,
                                       trial_dir, hparams)
                write_trial_log(METRIC_ACCURACY, accuracy, hparams, trial_dir)

                trial_step += 1


def get_callbacks(log_dir, checkpoint_dir):
    """Returns a list of callbacks.

    Args:
        log_dir: Path to the directory
            where TensorBoard logs will be written.
        checkpoint_dir: Path to the directory
            where model checkpoints will be written.

    Returns:
        A list of callbacks.
    """

    return [
        ModelCheckpoint(os.path.join(checkpoint_dir, 'checkpoint'),
                        monitor='val_loss', save_best_only=True,
                        save_weights_only=True),
        EarlyStopping(monitor='val_accuracy', patience=3),
        TensorBoard(log_dir)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        help='Path to UCF101 directory, which includes TFRecords files',
        required=True)
    parser.add_argument(
        '--batch_size',
        default=8,
        type=int,
        help='Number of samples per gradient update',
        required=True)

    args = parser.parse_args()

    # Dataset preparation.
    train_dataset = get_dataset(os.path.join(args.input_dir, 'train*'),
                                args.batch_size)
    val_dataset = get_dataset(os.path.join(args.input_dir, 'val*'),
                              args.batch_size)

    # Hyperparameter tuning
    tuning_dir = os.path.join('logs', 'hparam_tuning',
                              datetime.now().strftime("%Y%m%d-%H%M"))
    hparam_tuning(train_dataset, val_dataset, epochs=30, log_dir=tuning_dir)

    # TF2 Custom training
    # train_eagerly(model, train_dataset, val_dataset,
    #               optimizer, epochs=30, log_dir='logs')


if __name__ == "__main__":
    main()
