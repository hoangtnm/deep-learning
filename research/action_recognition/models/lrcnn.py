from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import (
    GRU, Dense, GlobalAveragePooling2D, TimeDistributed)
from tensorflow.keras.models import Sequential


def get_LRMobileNetV2(frames: int, classes: int, input_shape=(224, 224, 3)):
    """Long-term Recurrent MobileNetV2.

    Args:
        frames: number of frames in each sample.
        classes: number of classes to classify frame sequences into.
    """

    # MobileNetV2 Feature Extractor
    mobilenet = Sequential([
        MobileNetV2(
            input_shape, include_top=False, weights='imagenet'
        ),
        GlobalAveragePooling2D(),
        Dense(1024)
    ])

    model = Sequential()
    model.add(TimeDistributed(mobilenet,
                              input_shape=((frames,) + input_shape)))
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32))
    model.add(Dense(classes))
    return model
