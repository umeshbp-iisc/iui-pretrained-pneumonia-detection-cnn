"""Image/box augmentation transforms, kept separate from the Dataset class."""
from __future__ import annotations

import random
from typing import Any

import torch


class Compose:
    """Chains multiple transforms, each applied to (image, target)."""

    def __init__(self, transforms: list[Any]) -> None:
        self.transforms = transforms

    def __call__(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    """Randomly flips the image and boxes horizontally with probability p."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if random.random() < self.p:
            width = image.shape[-1]
            image = image.flip(-1)
            boxes = target["boxes"]
            if boxes.numel() > 0:
                x_min = boxes[:, 0].clone()
                x_max = boxes[:, 2].clone()
                boxes[:, 0] = width - x_max
                boxes[:, 2] = width - x_min
                target["boxes"] = boxes
        return image, target


class RandomBrightnessContrast:
    """Randomly jitters brightness and contrast; boxes are unaffected.

    Chosen over geometric jitter because X-ray opacity detection is more
    sensitive to intensity variation (windowing) than to scale/translation.
    """

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, p: float = 0.5) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if random.random() < self.p:
            brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = image.mean()
            image = (image - mean) * contrast_factor + mean * brightness_factor
            image = image.clamp(0.0, 1.0)
        return image, target


def get_train_transforms() -> Compose:
    """Default training-time augmentation pipeline."""
    return Compose([RandomHorizontalFlip(p=0.5), RandomBrightnessContrast(p=0.5)])


def get_eval_transforms() -> Compose:
    """No-op pipeline for validation/test (deterministic evaluation)."""
    return Compose([])
