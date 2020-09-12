import torch
from torch import Tensor
# from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    """Performs box-coordinates conversion, from COCO to Pascal VOC.

    Args:
        x: Boxes are expected to be in (x, y, w, h) format.

    Returns:
        Boxes in (x1, y1, x2, y2) format.

    Shape:
        - input: (N, 4)
        - output: (N, 4)
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    """Performs box-coordinates, from Pascal VOC to COCO.

    Args:
        x: boxes are expected to be in (x1, y1, x2, y2) format.

    Returns:
        Boxes in (x, y, w, h) format.

    Shape:
        - input: (N, 4)
        - output: (N, 4)
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)
