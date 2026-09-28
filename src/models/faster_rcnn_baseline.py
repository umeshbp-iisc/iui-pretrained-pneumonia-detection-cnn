"""Pretrained Faster R-CNN baseline for 2-class pneumonia opacity detection."""
from __future__ import annotations

import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 2  # background, pneumonia-opacity


def build_baseline_model(
    backbone: str = "resnet50_fpn_v2",
    min_size: int = 512,
    max_size: int = 768,
    pretrained: bool = True,
) -> nn.Module:
    """Build a Faster R-CNN with the box head replaced for 2 classes.

    Resizing (shortest side to `min_size`, capped at `max_size`, aspect ratio
    preserved) is handled internally by torchvision's `GeneralizedRCNNTransform`,
    so the dataset can return full-resolution images unchanged.

    Args:
        backbone: "resnet50_fpn_v2" (default, per SPEC-1) or "mobilenet_v3" as
            an explicit, documented fallback for limited GPU memory. Never
            swapped silently — must be passed as a config flag.
        min_size: Target shortest side length.
        max_size: Maximum allowed longest side length.
        pretrained: If False, builds a randomly-initialized backbone (no
            ImageNet/COCO weights) — used only for the SPEC-2 ablation lower
            bound, never as a silent default.

    Returns:
        A torchvision detection model ready for fine-tuning.
    """
    weights = "DEFAULT" if pretrained else None
    if backbone == "resnet50_fpn_v2":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=weights, min_size=min_size, max_size=max_size
        )
    elif backbone == "mobilenet_v3":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights, min_size=min_size, max_size=max_size
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model
