"""Evaluate a trained detector on the held-out test split.

Reports mAP@0.50, mAP@0.50:0.95, precision, recall, average IoU (matched),
and false positives per image, per SPEC-1/SPEC-2 acceptance criteria.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import RSNAPneumoniaDataset, collate_fn
from src.metrics import (
    compute_average_iou,
    compute_average_precision,
    compute_false_positives_per_image,
    compute_map_50_95,
    compute_precision_recall,
)
from src.models.dino_fusion_detector import build_dino_fusion_model
from src.models.faster_rcnn_baseline import build_baseline_model
from src.transforms import get_eval_transforms
from src.visualization import draw_predictions

CHECKPOINT_PATHS = {
    "baseline": "outputs/checkpoints/baseline_best.pt",
    "dino_fusion": "outputs/checkpoints/dino_fusion_best.pt",
}


@torch.no_grad()
def run_inference(
    model: torch.nn.Module, loader: DataLoader, device: str
) -> tuple[list, list, list, list, list]:
    """Run inference over a loader; returns per-image images/gt/pred tensors."""
    model.eval()
    images_out, gt_boxes_out, pred_boxes_out, pred_scores_out, pred_labels_out = (
        [],
        [],
        [],
        [],
        [],
    )

    for images, targets in loader:
        device_images = [img.to(device) for img in images]
        outputs = model(device_images)
        for img, tgt, out in zip(images, targets, outputs):
            images_out.append(img)
            gt_boxes_out.append(tgt["boxes"])
            pred_boxes_out.append(out["boxes"].cpu())
            pred_scores_out.append(out["scores"].cpu())
            pred_labels_out.append(out["labels"].cpu())

    return images_out, gt_boxes_out, pred_boxes_out, pred_scores_out, pred_labels_out


def evaluate(
    model_name: str,
    images_dir: str,
    test_csv: str,
    backbone: str = "resnet50_fpn_v2",
    batch_size: int = 8,
    checkpoint_path: str | None = None,
    baseline_checkpoint: str = "outputs/checkpoints/baseline_best.pt",
    output_dir: str = "outputs/predictions",
    num_visualizations: int = 5,
    device: str | None = None,
) -> dict:
    if model_name not in CHECKPOINT_PATHS:
        raise ValueError(f"Unknown model: {model_name}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = checkpoint_path or CHECKPOINT_PATHS[model_name]

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train the model first."
        )

    dataset = RSNAPneumoniaDataset(
        images_dir=images_dir, labels_csv=test_csv, transforms=get_eval_transforms()
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    if model_name == "dino_fusion":
        model = build_dino_fusion_model(baseline_checkpoint=baseline_checkpoint, backbone=backbone)
    else:
        model = build_baseline_model(backbone=backbone)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    images, gt_boxes, pred_boxes, pred_scores, _pred_labels = run_inference(model, loader, device)

    map50 = compute_average_precision(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)
    map50_95 = compute_map_50_95(pred_boxes, pred_scores, gt_boxes)
    precision, recall = compute_precision_recall(pred_boxes, pred_scores, gt_boxes)
    avg_iou = compute_average_iou(pred_boxes, pred_scores, gt_boxes)
    fp_per_image = compute_false_positives_per_image(pred_boxes, pred_scores, gt_boxes)

    metrics = {
        "model": model_name,
        "num_test_images": len(images),
        "mAP@0.50": map50,
        "mAP@0.50:0.95": map50_95,
        "precision": precision,
        "recall": recall,
        "avg_iou_matched": avg_iou,
        "false_positives_per_image": fp_per_image,
    }

    if map50 in (0.0, 1.0):
        print(
            f"WARNING: mAP@0.50 is exactly {map50} — check for a box-format or "
            "split-leakage bug before treating this as a final result."
        )

    print(json.dumps(metrics, indent=2))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    num_viz = min(num_visualizations, len(images))
    for i in range(num_viz):
        draw_predictions(
            images[i],
            gt_boxes[i],
            pred_boxes[i],
            pred_scores[i],
            save_path=output_path / f"{model_name}_pred_{i:03d}.png",
        )
    print(f"saved {num_viz} qualitative prediction images to {output_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="baseline", choices=["baseline", "dino_fusion"])
    parser.add_argument("--backbone", default="resnet50_fpn_v2", choices=["resnet50_fpn_v2", "mobilenet_v3"])
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--baseline-checkpoint", default="outputs/checkpoints/baseline_best.pt")
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--num-visualizations", type=int, default=5)
    args = parser.parse_args()

    test_csv = str(Path(args.splits_dir) / f"{args.split}.csv")

    evaluate(
        model_name=args.model,
        images_dir=args.images_dir,
        test_csv=test_csv,
        backbone=args.backbone,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path,
        baseline_checkpoint=args.baseline_checkpoint,
        output_dir=args.output_dir,
        num_visualizations=args.num_visualizations,
    )
