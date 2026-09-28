"""Draw ground-truth vs. predicted boxes on X-ray images for qualitative review."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

GT_COLOR = (0, 200, 0)
PRED_COLOR = (255, 60, 60)


def draw_predictions(
    image: torch.Tensor,
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    save_path: str | Path,
    score_threshold: float = 0.5,
) -> None:
    """Overlay ground-truth (green) and predicted (red) boxes and save to disk.

    Args:
        image: FloatTensor[3, H, W] in [0, 1].
        gt_boxes: FloatTensor[N, 4] in (x_min, y_min, x_max, y_max).
        pred_boxes: FloatTensor[M, 4] in the same format.
        pred_scores: FloatTensor[M] confidence scores.
        save_path: Output file path (e.g. .png).
        score_threshold: Minimum score for a prediction to be drawn.
    """
    array = (image.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    pil_image = Image.fromarray(array).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for box in gt_boxes.tolist():
        draw.rectangle(box, outline=GT_COLOR, width=3)

    keep = pred_scores >= score_threshold
    for box, score in zip(pred_boxes[keep].tolist(), pred_scores[keep].tolist()):
        draw.rectangle(box, outline=PRED_COLOR, width=2)
        draw.text((box[0], max(box[1] - 12, 0)), f"{score:.2f}", fill=PRED_COLOR)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(save_path)
