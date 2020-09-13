import torch
import torchvision
from PIL import Image
from torch import Tensor
from typing import Callable, Optional, Tuple, Dict, Union
import research.datasets.coco.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    """Coco detection dataset."""

    def __init__(self, root: str, annFile: str,
                 transforms: Optional[Callable] = None) -> None:
        super().__init__(root, annFile)
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple[Union[Image.Image, Tensor],
                                             Dict[str, Tensor]]:
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]

        w, h = img.size
        image_id = torch.tensor([image_id])
        anno = [obj for obj in target if
                'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # [x_min, y_min, w, h] -> [x_min, y_min, x_max, y_max]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.long)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {
            'boxes': boxes,
            'labels': classes,
            'image_id': image_id,
        }

        # For conversion to COCO API
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.tensor([int(h), int(w)])
        target["size"] = torch.tensor([int(h), int(w)])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def data_transforms(phase: str, max_size=1333):
    if phase == 'train':
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomChoice([
                T.RandomResize(scales, max_size),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomResizedCrop(384, 600),
                    T.RandomResize(scales, max_size)
                ])
            ]),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
