from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import (GRU, Dense, GlobalAveragePooling2D,
                                     TimeDistributed)


def LRMobileNetV2(seq_length, frame_shape, classes, weights):
    """Long-term Recurrent MobileNetV2 architecture.

    Reference paper:
    - [Long-term Recurrent Convolutional Networks for Visual Recognition and Description]
    (https://arxiv.org/abs/1411.4389)

    Args:
        seq_length: Number of frames in each sample.
        frame_shape: Shape of each frame in the sample.
        classes: Number of classes to classify samples.
        weights: One of `None` (random initialization),
            or the path to the weights file to be loaded.

    Returns:
        model: A `keras.Model` instance.
    """

    model = keras.Sequential()
    model.add(TimeDistributed(
        MobileNetV2(frame_shape, include_top=False, weights='imagenet'),
        input_shape=((seq_length,) + frame_shape)))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(GRU(512, return_sequences=True))
    model.add(GRU(256, return_sequences=True))
    model.add(GRU(128))
    model.add(Dense(classes))

    # Load weights
    if weights is not None:
        model.load_weights(weights)

    return model
