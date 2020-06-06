import os

from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        TensorBoard)


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
