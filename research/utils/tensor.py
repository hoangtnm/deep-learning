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

    def __init__(self, tensor: Tensor, mask: Optional[Tensor] = None):
        self.tensor = tensor
        self.mask = mask

    def to(self, device: torch.device):
        tensor = self.tensor.to(device)
        mask = self.mask
        mask = mask.to(device) if mask is not None else None
        return NestedTensor(tensor, mask)

    def decompose(self):
        return self.tensor, self.mask

    def __repr__(self):
        return str(self.tensor)
