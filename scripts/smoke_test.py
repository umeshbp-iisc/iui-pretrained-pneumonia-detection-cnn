"""Smoke test: overfit a small image subset to prove the pipeline is wired correctly.

Per SPEC-1 acceptance criteria, this must drive training loss below 0.1 within
~100 iterations before any full training run is attempted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import RSNAPneumoniaDataset, collate_fn
from src.models.dino_fusion_detector import build_dino_fusion_model
from src.models.faster_rcnn_baseline import build_baseline_model


def run_smoke_test(
    images_dir: str,
    labels_csv: str,
    model_name: str = "baseline",
    backbone: str = "resnet50_fpn_v2",
    subset_size: int = 50,
    iterations: int = 100,
    loss_threshold: float = 0.1,
    baseline_checkpoint: str = "outputs/checkpoints/baseline_best.pt",
    lr: float = 0.005,
    optimizer_name: str = "sgd",
    device: str | None = None,
) -> bool:
    """Overfit `subset_size` images for `iterations` steps; return pass/fail."""
    if model_name not in ("baseline", "dino_fusion"):
        raise ValueError(f"Unsupported --model for this script: {model_name}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = RSNAPneumoniaDataset(images_dir=images_dir, labels_csv=labels_csv)
    subset_indices = list(range(min(subset_size, len(full_dataset))))
    subset = Subset(full_dataset, subset_indices)
    loader = DataLoader(
        subset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0
    )

    if model_name == "dino_fusion":
        model = build_dino_fusion_model(baseline_checkpoint=baseline_checkpoint, backbone=backbone)
    else:
        model = build_baseline_model(backbone=backbone)
    model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
    else:
        optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9)

    last_loss = float("inf")
    step = 0
    while step < iterations:
        for images, targets in loader:
            if step >= iterations:
                break
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss_dict}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = loss.item()
            step += 1
            if step % 10 == 0 or step == 1:
                print(f"step {step}/{iterations} loss={last_loss:.4f}")

    passed = last_loss < loss_threshold
    print(f"Final loss: {last_loss:.4f} (threshold {loss_threshold}) -> {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="baseline", choices=["baseline", "dino_fusion"])
    parser.add_argument("--backbone", default="resnet50_fpn_v2", choices=["resnet50_fpn_v2", "mobilenet_v3"])
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--subset-size", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--loss-threshold", type=float, default=0.1)
    parser.add_argument("--baseline-checkpoint", default="outputs/checkpoints/baseline_best.pt")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    args = parser.parse_args()

    ok = run_smoke_test(
        images_dir=args.images_dir,
        labels_csv=args.labels_csv,
        model_name=args.model,
        backbone=args.backbone,
        baseline_checkpoint=args.baseline_checkpoint,
        subset_size=args.subset_size,
        iterations=args.iterations,
        loss_threshold=args.loss_threshold,
        lr=args.lr,
        optimizer_name=args.optimizer,
    )
    sys.exit(0 if ok else 1)
