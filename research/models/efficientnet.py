# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""EfficientNet models for PyTorch.

Reference paper:
  - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks]
    (https://arxiv.org/abs/1905.11946) (ICML 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_BLOCKS_ARGS = [{
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 32,
    'filters_out': 16,
    'expand_ratio': 1,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 2,
    'filters_in': 16,
    'filters_out': 24,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 2,
    'filters_in': 24,
    'filters_out': 40,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 3,
    'filters_in': 40,
    'filters_out': 80,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 3,
    'filters_in': 80,
    'filters_out': 112,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 4,
    'filters_in': 112,
    'filters_out': 192,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 192,
    'filters_out': 320,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


class EfficientNet(nn.Module):
    """Instantiates the EfficientNet architecture using given scaling coefficients.

    Reference:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
        https://arxiv.org/abs/1905.11946
    ) (ICML 2019)

    Args:
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_size: default input image size.
        dropout_rate: dropout rate before final classifier layer.
        drop_connect_rate: dropout rate at skip connections.
        depth_divisor: a unit of network width.
    """

    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 default_size: int,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 depth_divisor: int = 8,
                 #  activation='swish',
                 #  blocks_args='default',
                 #  model_name='efficientnet',
                 #  include_top=True,
                 weights='imagenet',
                 #  input_tensor=None,
                 #  input_shape=None,
                 #  pooling=None,
                 num_classes=1000):
        super().__init__()

    def forward(self, x):
        return x
