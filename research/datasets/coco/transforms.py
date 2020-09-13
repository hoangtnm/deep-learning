import random
from typing import Callable, Dict, List, Optional, Union, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from torch import Tensor

from research.utils import box_xyxy_to_cxcywh


def crop(img: Image.Image, target: Dict[str, Tensor], region: List[int]):
    """Crop the given PIL Image.

    Args:
        img: PIL Image to be cropped.
        target: A dict mapping keys to the corresponding values.
        region: A list of [top, left, height, width]
            or [y_min, x_min, height, width].

    Returns:
        img: Cropped image.
        target: A dict mapping keys to the corresponding values.
    """
    img = F.crop(img, *region)
    fields = ['labels', 'area', 'iscrowd']
    target = target.copy()

    if 'boxes' in target:
        boxes = target['boxes']
        top, left, h, w = region
        max_size = torch.tensor([w, h], dtype=torch.float32)
        boxes = boxes - torch.tensor([left, top, left, top])
        boxes = torch.min(boxes.view(-1, 2, 2), max_size)
        # Boxes having zero area will be filtered later.
        boxes = boxes.clamp(min=0)
        # area = (x2 - x1) * (y2 - y1)
        area = (boxes[:, 1, :] - boxes[:, 0, :]).prod(dim=1)
        assert area.dim() == 1
        target['boxes'] = boxes.view(-1, 4)
        target['area'] = area
        fields.append('boxes')

    # Remove elements for which the boxes having zero area
    if 'boxes' in target:
        # boxes = boxes.view(-1, 2, 2)
        # keep = torch.all(boxes[:, 1, :] > boxes[:, 0, :], dim=1)
        keep = area > 0
        for field in fields:
            target[field] = target[field][keep]

    return img, target


def get_size_maintaining_aspect_ratio(img: Image.Image, size: int,
                                      max_size: Optional[int] = None):
    w, h = img.size
    if max_size is not None:
        min_original_size = float(min(w, h))
        max_original_size = float(max(w, h))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return h, w

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return oh, ow


def resize(img: Image.Image,
           target: Dict[str, Tensor],
           size: Union[int, List, Tuple],
           max_size: Optional[int] = None):
    """Resize the input PIL Image to the given size.

    Args:
        img: PIL Image to be resized.
        target: A dict mapping keys to the corresponding values.
        size: Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        max_size: Max size.
    """

    if isinstance(size, (list, tuple)):
        h, w = size
    else:
        h, w = get_size_maintaining_aspect_ratio(img, size, max_size)

    size = h, w
    scaled_img = F.resize(img, size)

    if target is None:
        return scaled_img, None

    scaled_ratios = [float(s) / float(s_orig)
                     for s, s_orig in zip(scaled_img.size, img.size)]
    ratio_width, ratio_height = scaled_ratios

    target = target.copy()
    if 'boxes' in target:
        boxes = target['boxes']
        scaled_boxes = boxes * torch.tensor([ratio_width, ratio_height,
                                             ratio_width, ratio_height])
        target['boxes'] = scaled_boxes

    if 'area' in target:
        area = target['area']
        resized_area = area * (ratio_width * ratio_height)
        target['area'] = resized_area

    target['size'] = torch.tensor([h, w])

    return scaled_img, target


class Compose:
    """Composes several transforms together.

    Args:
        transforms: list of transforms to compose.
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img, target: Dict[str, Tensor]):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomChoice:
    """Apply single transformation randomly picked from a list."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img, target: Dict[str, Tensor]):
        t = random.choice(self.transforms)
        return t(img, target)


class RandomHorizontalFlip:
    """Horizontally flip the given image randomly with a given probability."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target: Dict[str, Tensor]):
        """
        Args:
            img: PIL Image to be flipped.
            target: target: A dict mapping keys to the corresponding values.

        Returns:
            img: Randomly flipped image.
            target: The original target containing flipped box coordinates.
        """
        if torch.rand(1) < self.p:
            img = F.hflip(img)
            target = target.copy()
            if 'boxes' in target:
                boxes = target['boxes']
                w, h = img.size

                # Box coordinates are expected in (x1, y1, x2, y2) format
                # x_new = w - x
                # y_new = y
                boxes = (boxes[:, [2, 1, 0, 3]]
                         * torch.tensor([-1, 1, -1, 1])
                         + torch.tensor([w, 0, w, 0]))
                target['boxes'] = boxes
        return img, target


class RandomResizedCrop:
    """Crop the given PIL Image to random size."""

    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: Image.Image, target: Dict[str, Tensor]):
        w = random.randint(self.min_size, self.max_size)
        h = random.randint(self.min_size, self.max_size)
        region = T.RandomCrop.get_params(img, (h, w))
        return crop(img, target, region)


class RandomResize:
    def __init__(self, sizes: List[int], max_size: Optional[int] = None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target: Dict[str, Tensor] = None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class ToTensor:
    def __call__(self, img, target: Dict[str, Tensor]):
        return F.to_tensor(img), target


class Normalize:
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std

    def __call__(self,
                 tensor: Tensor,
                 target: Optional[Dict[str, Tensor]] = None):
        """
        Args:
            tensor: Tensor image of size (C, H, W) to be normalized.
            target: A dict mapping keys to the corresponding values.

        Returns:
            tensor: Normalized tensor image.
            target: The original target dict.
        """
        tensor = F.normalize(tensor, mean=self.mean, std=self.std)
        if target is None:
            return tensor, None
        target = target.copy()
        h, w = tensor.shape[-2:]
        if 'boxes' in target:
            boxes = target['boxes']
            boxes = box_xyxy_to_cxcywh(boxes)
            # Box coordinates are expected in (x1, y1, x2, y2) format
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target['boxes'] = boxes
        return tensor, target
