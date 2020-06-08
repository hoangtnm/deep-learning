# Hyperparameter Tuning with the HParams Dashboard <!-- omit in toc -->

## Contents <!-- omit in toc -->

- [Experiment setup and the HParams experiment summary](#experiment-setup-and-the-hparams-experiment-summary)
- [Adapt TensorFlow runs to log hyperparameters and metrics](#adapt-tensorflow-runs-to-log-hyperparameters-and-metrics)
- [Start runs and log them all under one parent directory](#start-runs-and-log-them-all-under-one-parent-directory)
- [References](#references)

When building machine learning models, you need to choose various [hyperparameters](<https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)>), such as the dropout rate in a layer or the learning rate. These decisions impact model metrics, such as accuracy. Therefore, an important step in the machine learning workflow is to identify the best hyperparameters for your problem, which often involves experimentation. This process is known as "Hyperparameter Optimization" or "Hyperparameter Tuning".

The HParams dashboard in TensorBoard provides several tools to help with this process of identifying the best experiment or most promising sets of hyperparameters.

## Experiment setup and the HParams experiment summary

Experiment with three hyperparameters in the model:

- Number of units in the first dense layer
- Dropout rate in the dropout layer
- Optimizer

List the values to try, and log an experiment configuration to TensorBoard. This step is optional: you can provide domain information to enable more precise filtering of hyperparameters in the UI, and you can specify which metrics should be displayed.

```python
from tensorboard.plugins.hparams import api as hp


HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

HPARAMS = [HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER]

METRIC_ACCURACY = 'accuracy'

METRICS = [hp.Metric(METRIC_ACCURACY, display_name='Accuracy')]

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(hparams=HPARAMS, metrics=METRICS)
```

If you choose to skip this step, you can use a string literal wherever you would otherwise use an HParam value: e.g., hparams['dropout'] instead of hparams[HP_DROPOUT].

## Adapt TensorFlow runs to log hyperparameters and metrics

The model will be quite simple: two dense layers with a dropout layer between them. The training code will look familiar, although the hyperparameters are no longer hardcoded. Instead, the hyperparameters are provided in an hparams dictionary and used throughout the training function:

```python
def train_test_model(hparams):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ])

    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(x_train, y_train, epochs=1)
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy
```

For each run, log an hparams summary with the hyperparameters and final accuracy:

```python
def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # Record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
```

When training Keras models, you can use callbacks instead of writing these directly:

```python
METRIC_ACCURACY = 'epoch_accuracy'

METRICS = [hp.Metric(tag=METRIC_ACCURACY, group='validation', display_name='Accuracy')]


with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(hparams=HPARAMS, metrics=METRICS,)


def train_test_model(hparams):
    ...
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        callbacks=[
            tf.keras.callbacks.TensorBoard(logdir),  # log metrics
            hp.KerasCallback(logdir, hparams),  # log hparams
        ]
    )
```

### Notes <!-- omit in toc -->

As you can see the configurations for `hp.Metric` when using callbacks are slightly different from the first one in terms of `tag` and `group`. The reseason is that [TensorBoard callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard) writes specific summary tags for metrics such as `epoch_accuracy` and `epoch_lossnot` w.r.t custom tag like `accuracy`, which is inconsistent with `hp.hparams_config(..., metrics=METRICS)` and causes TensorBoard's HParams plugin cannot find the summary tag with the name `accuracy` in log files.
Therefore, the value for hp.Metric's tag must be `epoch_accuracy`, which is compatible with TensorBoard callback.

- tag: The tag name of the scalar summary that corresponds to this metric.
- group: An optional string listing the subdirectory under the session's log directory containing summaries for this metric. For instance, if summaries for training runs are written to events files in `ROOT_LOGDIR/SESSION_ID/train`, then `group` should be `"train"`. Defaults to the empty string: i.e., summaries are expected to be written to the session logdir.

## Start runs and log them all under one parent directory

```python
session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            hparams = {
                HP_NUM_UNITS: num_units,
                HP_DROPOUT: dropout_rate,
                HP_OPTIMIZER: optimizer,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run('logs/hparam_tuning/' + run_name, hparams)
            session_num += 1
```

## References

[1] TensorFlow. Hyperparameter Tuning with the HParams Dashboard, [Online]. Available: https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams. (accessed: 06.08.2020).

[2] TensorFlow. A metric in an experiment, [Online]. Available: https://github.com/tensorflow/tensorboard/blob/2e01eebce680fa50a783afdd3a4a9aec36ff3492/tensorboard/plugins/hparams/summary_v2.py#L545. (accessed: 06.08.2020).
