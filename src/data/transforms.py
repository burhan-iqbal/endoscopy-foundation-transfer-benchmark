"""Image transforms for CE-NBI classification."""

from __future__ import annotations

from typing import Sequence

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    split: str,
    image_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    rotation_degrees: float = 10.0,
    color_jitter: dict | None = None,
) -> transforms.Compose:
    """
    Conservative augmentations for training; resize+normalize only for val/test.

    Vascular patterns are preserved: no heavy elastic warps or aggressive colour shifts.
    """
    color_jitter = color_jitter or {
        "brightness": 0.1,
        "contrast": 0.1,
        "saturation": 0.05,
        "hue": 0.02,
    }
    normalize = transforms.Normalize(mean=list(mean), std=list(std))

    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=rotation_degrees),
                transforms.ColorJitter(**color_jitter),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
