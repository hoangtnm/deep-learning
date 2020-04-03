import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import (GRU, LSTM, Conv3D, Dense, Dropout,
                                     Flatten, GlobalAveragePooling2D,
                                     MaxPool3D, TimeDistributed)
from tensorflow.keras.models import Model, Sequential


class TFC3D(Model):

    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes

        # 1st group
        self.conv1a = Conv3D(64, (3, 3, 3), (1, 1, 1),
                             padding='same', activation='relu',
                             input_shape=(16, 112, 112, 3))
        self.pool1 = MaxPool3D((1, 2, 2), (1, 2, 2))

        # 2nd group
        self.conv2a = Conv3D(128, (3, 3, 3), (1, 1, 1),
                             padding='same', activation='relu')
        self.pool2 = MaxPool3D((2, 2, 2), (2, 2, 2))

        # 3rd group
        self.conv3a = Conv3D(256, (3, 3, 3), (1, 1, 1),
                             padding='same', activation='relu')
        self.conv3b = Conv3D(256, (3, 3, 3), (1, 1, 1),
                             padding='same', activation='relu')
        self.pool3 = MaxPool3D((2, 2, 2), (2, 2, 2))

        # 4th group
        self.conv4a = Conv3D(512, (3, 3, 3), (1, 1, 1),
                             padding='same', activation='relu')
        self.conv4b = Conv3D(512, (3, 3, 3), (1, 1, 1),
                             padding='same', activation='relu')
        self.pool4 = MaxPool3D((2, 2, 2), (2, 2, 2))

        # 5th group
        self.conv5a = Conv3D(512, (3, 3, 3), (1, 1, 1),
                             padding='same', activation='relu')
        self.conv5b = Conv3D(512, (3, 3, 3), (1, 1, 1),
                             padding='same', activation='relu')
        self.pool5 = MaxPool3D((2, 2, 2), (2, 2, 2))

        # fc group
        self.flatten = Flatten()
        self.fc6 = Dense(4096, activation='relu')
        self.dropout6 = Dropout(dropout)
        self.fc7 = Dense(4096, activation='relu')
        self.dropout7 = Dropout(dropout)
        self.fc8 = Dense(num_classes, activation='softmax')

    @tf.function
    def call(self, x, training=False):
        x = self.conv1a(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)

        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.pool5(x)

        x = self.flatten(x)
        x = self.dropout6(self.fc6(x), training=training)
        x = self.dropout7(self.fc7(x), training=training)
        x = self.fc8(x)
        return x


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
    model.add(TimeDistributed(mobilenet, input_shape=(frames, 224, 224, 3)))
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32))
    model.add(Dense(classes))

    return model
