import os

from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import (Callback, EarlyStopping,
                                        ModelCheckpoint, TensorBoard)


class F1Score(Callback):
    def __init__(self, average='macro', **kwargs):
        """Callback to compute F1 score, also known as balanced F-score.

        The F1 score can be interpreted as a weighted average of the precision
        and recall, where an F1 score reaches its best value at 1 and
        worst score at 0. The relative contribution of precision and recall to
        the F1 score are equal. The formula for the F1 score is:

            F1 = 2 * (precision * recall) / (precision + recall)

        In the multi-class and multi-label case, this is the average of
        the F1 score of each class with weighting depending on the ``average``
        parameter.

        Args:
            average: One of [None, ‘binary’, ‘micro’, ‘macro’, ‘samples’, ‘weighted’]

                This parameter is required for multi-class/multi-label targets.
                If ``None``, the scores for each class are returned. Otherwise,
                this determines the type of averaging performed on the data.

        See details at
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        """
        super().__init__(**kwargs)
        self.average = average

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass


def get_common_callbacks(log_dir, checkpoint_dir, monitor='val_accuracy',
                         **kwargs):
    """Returns a list of common TensorFlow callbacks.

    Args:
        log_dir: Path to the directory
            where TensorBoard logs will be written.
        checkpoint_dir: Path to the directory
            where checkpoints will be written.
        monitor: Quantity to monitor.
            Recommended values are `val_accuracy` or `val_loss`.

    Returns:
        A list of callbacks.
    """
    return [
        TensorBoard(log_dir, **kwargs),
        EarlyStopping(monitor=monitor, patience=3,
                      restore_best_weights=True, **kwargs),
        ModelCheckpoint(os.path.join(checkpoint_dir, 'checkpoint'),
                        monitor=monitor, save_best_only=True,
                        save_weights_only=True, **kwargs),
    ]
