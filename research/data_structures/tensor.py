import torch
from torch import Tensor
from typing import Optional


class NestedTensor:
    """NestedTensor for DETR.

    Reference:
    - [End-to-end Object Detection with Transformers](
        https://github.com/facebookresearch/detr/blob/master/util/misc.py
    )
    """

    def __init__(self, tensors: Tensor, mask: Optional[Tensor] = None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            # assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
