import os

from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        TensorBoard)


def get_default_callbacks(log_dir, checkpoint_dir, **kwargs):
    """Returns a list of common callbacks.

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
                        save_weights_only=True, **kwargs),
        EarlyStopping(monitor='val_accuracy', patience=3, **kwargs),
        TensorBoard(log_dir, **kwargs)
    ]
