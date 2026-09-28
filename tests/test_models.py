"""Tests for SPEC-2 model code: DINOv2 encoder and fusion module shapes/freezing."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.dino_encoder import PROJECTED_DIM, DinoV2Encoder
from src.models.dino_fusion_detector import build_dino_fusion_model
from src.models.faster_rcnn_baseline import build_baseline_model


@pytest.fixture(scope="module")
def baseline_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A small, freshly-initialized baseline checkpoint for fusion tests."""
    model = build_baseline_model(backbone="resnet50_fpn_v2")
    path = tmp_path_factory.mktemp("checkpoints") / "baseline_best.pt"
    torch.save(model.state_dict(), path)
    return str(path)


def test_dino_encoder_output_shape() -> None:
    encoder = DinoV2Encoder()
    encoder.eval()
    images = [torch.rand(3, 512, 512), torch.rand(3, 600, 480)]
    with torch.no_grad():
        embeddings = encoder(images)
    assert embeddings.shape == (2, PROJECTED_DIM)


def test_dino_backbone_is_frozen() -> None:
    encoder = DinoV2Encoder()
    for param in encoder.backbone.parameters():
        assert param.requires_grad is False
    assert encoder.backbone.training is False


def test_dino_backbone_stays_eval_in_train_mode() -> None:
    encoder = DinoV2Encoder()
    encoder.train()  # simulate outer model .train() call
    images = [torch.rand(3, 400, 400)]
    with torch.no_grad():
        encoder(images)
    assert encoder.backbone.training is False


def test_dino_projection_is_trainable() -> None:
    encoder = DinoV2Encoder()
    assert all(p.requires_grad for p in encoder.projection.parameters())


def test_fusion_detector_output_shape_train_mode(baseline_checkpoint: str) -> None:
    model = build_dino_fusion_model(baseline_checkpoint=baseline_checkpoint)
    model.train()
    images = [torch.rand(3, 512, 512), torch.rand(3, 480, 480)]
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor([0]),
            "area": torch.tensor([1600.0]),
            "iscrowd": torch.tensor([0]),
        },
        {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([1]),
            "area": torch.zeros((0,)),
            "iscrowd": torch.zeros((0,), dtype=torch.int64),
        },
    ]
    loss_dict = model(images, targets)
    assert isinstance(loss_dict, dict)
    assert all(torch.isfinite(v) for v in loss_dict.values())


def test_fusion_detector_output_shape_eval_mode(baseline_checkpoint: str) -> None:
    model = build_dino_fusion_model(baseline_checkpoint=baseline_checkpoint)
    model.eval()
    images = [torch.rand(3, 512, 512)]
    with torch.no_grad():
        outputs = model(images)
    assert len(outputs) == 1
    assert "boxes" in outputs[0] and "scores" in outputs[0] and "labels" in outputs[0]
    assert outputs[0]["boxes"].shape[1] == 4


def test_fusion_detector_freezes_backbone(baseline_checkpoint: str) -> None:
    model = build_dino_fusion_model(baseline_checkpoint=baseline_checkpoint)
    for name, param in model.detector.named_parameters():
        if not name.startswith("roi_heads.box_predictor"):
            assert param.requires_grad is False, f"{name} should be frozen"
    for param in model.detector.roi_heads.box_predictor.parameters():
        assert param.requires_grad is True
    for param in model.dino_encoder.backbone.parameters():
        assert param.requires_grad is False
    assert all(p.requires_grad for p in model.dino_encoder.projection.parameters())
