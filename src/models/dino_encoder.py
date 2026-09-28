"""Frozen DINOv2 ViT-S/14 global-context encoder with a trainable projection head."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.transforms import functional as TF

DINO_INPUT_SIZE = 518  # multiple of the ViT-S/14 patch size (14 * 37)
DINO_MEAN = (0.485, 0.456, 0.406)  # ImageNet stats used by the official DINOv2 hub model
DINO_STD = (0.229, 0.224, 0.225)
DINO_EMBED_DIM = 384
PROJECTED_DIM = 256


class DinoV2Encoder(nn.Module):
    """Wraps a frozen DINOv2 ViT-S/14 backbone plus a trainable projection layer.

    The DINOv2 backbone is always frozen (`requires_grad=False`) and always run
    in `.eval()` mode, even while the outer detector is in `.train()` mode, per
    SPEC-2's non-negotiable constraint.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.projection = nn.Sequential(
            nn.Linear(DINO_EMBED_DIM, PROJECTED_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def preprocess(self, images: list[torch.Tensor]) -> torch.Tensor:
        """Resize/normalize raw [0,1] 3-channel images to DINOv2's expected input.

        This is independent of the Faster R-CNN input pipeline (different
        target size and normalization statistics) — do not reuse one for both.
        """
        resized = [
            TF.resize(img, [DINO_INPUT_SIZE, DINO_INPUT_SIZE], antialias=True)
            for img in images
        ]
        batch = torch.stack(resized, dim=0)
        return TF.normalize(batch, mean=DINO_MEAN, std=DINO_STD)

    def forward(self, images: list[torch.Tensor]) -> torch.Tensor:
        """Return a [batch, PROJECTED_DIM] global embedding per image.

        Args:
            images: List of FloatTensor[3, H, W] images in [0, 1], one per
                image in the batch (Faster R-CNN's native per-image format).
        """
        dino_input = self.preprocess(images).to(next(self.projection.parameters()).device)

        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(dino_input)  # [batch, DINO_EMBED_DIM]

        return self.projection(features)
