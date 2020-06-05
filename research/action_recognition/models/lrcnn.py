from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, TimeDistributed


def LRMobileNetV2(seq_length, frame_shape, classes, weights, dropout=0.5):
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
        MobileNetV2(frame_shape, include_top=False,
                    pooling='avg', weights='imagenet'),
        input_shape=((seq_length,) + frame_shape)))
    model.add(LSTM(1024, return_sequences=True))
    model.add(Dropout(dropout=dropout))
    model.add(GRU(512, return_sequences=True))
    model.add(Dropout(dropout=dropout))
    model.add(GRU(256, return_sequences=True))
    model.add(Dropout(dropout=dropout))
    model.add(GRU(128))
    model.add(Dropout(dropout=dropout))
    model.add(Dense(classes, activation='softmax', name='predictions'))

    # Load weights
    if weights is not None:
        model.load_weights(weights)

    return model
