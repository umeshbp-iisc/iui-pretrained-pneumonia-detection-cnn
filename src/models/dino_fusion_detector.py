"""SPEC-2 DINOv2 fusion detector: injects frozen DINOv2 global context after ROI Align.

Architecture (per SPEC.md SPEC-2):
    SPEC-1 Faster R-CNN up to ROI Align
        -> box_head (TwoMLPHead) produces per-proposal features
        -> broadcast + concatenate the projected DINOv2 global embedding
        -> fusion MLP -> class logits, box deltas
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.models.dino_encoder import PROJECTED_DIM, DinoV2Encoder
from src.models.faster_rcnn_baseline import build_baseline_model


class FusionPredictor(nn.Module):
    """Replaces FastRCNNPredictor: concatenates ROI features with a broadcast
    DINOv2 embedding, then predicts class logits and box deltas.

    `dino_embeddings` and `boxes_per_image` must be set externally (by
    `DinoFusionDetector`) before each forward call.
    """

    def __init__(
        self, in_channels: int, num_classes: int, dino_dim: int = PROJECTED_DIM, hidden_dim: int = 1024
    ) -> None:
        super().__init__()
        self.fusion_fc1 = nn.Linear(in_channels + dino_dim, hidden_dim)
        self.fusion_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.cls_score = nn.Linear(hidden_dim, num_classes)
        self.bbox_pred = nn.Linear(hidden_dim, num_classes * 4)

        self.dino_embeddings: torch.Tensor | None = None
        self.boxes_per_image: list[int] | None = None

    def forward(self, box_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if box_features.dim() == 4:
            box_features = box_features.flatten(start_dim=1)
        if self.dino_embeddings is None or self.boxes_per_image is None:
            raise RuntimeError(
                "FusionPredictor requires dino_embeddings/boxes_per_image to be set "
                "before forward (this should be done by DinoFusionDetector)."
            )

        counts = torch.as_tensor(self.boxes_per_image, device=box_features.device)
        expanded_dino = torch.repeat_interleave(self.dino_embeddings, counts, dim=0)
        fused = torch.cat([box_features, expanded_dino], dim=1)
        fused = F.relu(self.fusion_fc1(fused))
        fused = F.relu(self.fusion_fc2(fused))
        return self.cls_score(fused), self.bbox_pred(fused)


class DinoFusionDetector(nn.Module):
    """Wraps a Faster R-CNN detector plus a frozen DINOv2 encoder.

    Computes the DINOv2 global embedding per image, then uses a forward
    pre-hook on `box_roi_pool` to learn each image's proposal count so the
    embedding can be correctly broadcast onto per-proposal ROI features inside
    `FusionPredictor`.
    """

    def __init__(self, detector: nn.Module, dino_encoder: DinoV2Encoder) -> None:
        super().__init__()
        self.detector = detector
        self.dino_encoder = dino_encoder
        self.detector.roi_heads.box_roi_pool.register_forward_pre_hook(
            self._capture_boxes_per_image
        )

    def _capture_boxes_per_image(self, module: nn.Module, inputs: tuple[Any, ...]) -> None:
        boxes = inputs[1]
        self.detector.roi_heads.box_predictor.boxes_per_image = [len(b) for b in boxes]

    def forward(
        self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None
    ):
        dino_embeddings = self.dino_encoder(images)
        self.detector.roi_heads.box_predictor.dino_embeddings = dino_embeddings
        return self.detector(images, targets)


def build_dino_fusion_model(
    baseline_checkpoint: str,
    backbone: str = "resnet50_fpn_v2",
    min_size: int = 512,
    max_size: int = 768,
    unfreeze_box_predictor: bool = False,
) -> DinoFusionDetector:
    """Build the SPEC-2 fusion detector, initialized from the SPEC-1 baseline.

    Per SPEC-2, only the DINOv2 projection layer and fusion MLP are trainable
    in the first experiment; everything else (backbone/RPN/box_head, and the
    DINOv2 backbone itself) is frozen.

    Args:
        baseline_checkpoint: Path to the SPEC-1 `baseline_best.pt` weights.
        backbone: Must match the backbone used for `baseline_checkpoint`.
        min_size: Faster R-CNN input resize shortest side.
        max_size: Faster R-CNN input resize longest side cap.
        unfreeze_box_predictor: Reserved for a future experiment that also
            unfreezes upper backbone layers; must be a separate run per SPEC-2,
            not the default (currently unused; the fusion predictor is always
            trainable regardless of this flag).
    """
    model = build_baseline_model(backbone=backbone, min_size=min_size, max_size=max_size)

    state_dict = torch.load(baseline_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = model.roi_heads.box_predictor.cls_score.out_features
    model.roi_heads.box_predictor = FusionPredictor(in_features, num_classes)

    # Freeze everything except the new fusion predictor (SPEC-2 first-experiment default).
    for name, param in model.named_parameters():
        if not name.startswith("roi_heads.box_predictor"):
            param.requires_grad = False

    dino_encoder = DinoV2Encoder()

    return DinoFusionDetector(model, dino_encoder)
