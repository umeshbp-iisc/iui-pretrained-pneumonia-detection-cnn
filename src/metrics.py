"""Minimal detection metrics: mAP@0.50, mAP@0.50:0.95, precision, recall, IoU."""
from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import box_iou


def _match_predictions(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy match predictions (sorted by score) to ground-truth boxes.

    Returns:
        tp: boolean array, one entry per prediction (in score order).
        fp: boolean array, complementary to tp.
        num_gt: number of ground-truth boxes for this image.
    """
    num_preds = pred_boxes.shape[0]
    num_gt = gt_boxes.shape[0]
    tp = np.zeros(num_preds, dtype=bool)
    fp = np.zeros(num_preds, dtype=bool)

    if num_preds == 0:
        return tp, fp, num_gt

    order = torch.argsort(pred_scores, descending=True)
    matched_gt = set()

    if num_gt == 0:
        fp[:] = True
        return tp, fp, num_gt

    ious = box_iou(pred_boxes[order], gt_boxes)

    # Greedy matching in score order: each prediction claims its best-IoU
    # ground truth only if that box isn't already claimed by a higher-score prediction.
    for i, pred_idx in enumerate(order.tolist()):
        row = ious[i]
        best_iou, best_gt = torch.max(row, dim=0)
        best_gt = int(best_gt.item())
        if best_iou.item() >= iou_threshold and best_gt not in matched_gt:
            tp[i] = True
            matched_gt.add(best_gt)
        else:
            fp[i] = True

    # Undo the score-sort so tp/fp align with `order`'s original prediction indices.
    tp_out = np.zeros(num_preds, dtype=bool)
    fp_out = np.zeros(num_preds, dtype=bool)
    tp_out[order.numpy()] = tp
    fp_out[order.numpy()] = fp
    return tp_out, fp_out, num_gt


def compute_average_precision(
    all_pred_boxes: list[torch.Tensor],
    all_pred_scores: list[torch.Tensor],
    all_gt_boxes: list[torch.Tensor],
    iou_threshold: float = 0.5,
) -> float:
    """Compute AP at a single IoU threshold across a dataset (single class).

    Args:
        all_pred_boxes: Per-image predicted boxes [Ni, 4].
        all_pred_scores: Per-image predicted scores [Ni].
        all_gt_boxes: Per-image ground-truth boxes [Mi, 4].
        iou_threshold: IoU threshold for a match to count as a true positive.

    Returns:
        Average precision (area under the precision-recall curve).
    """
    scored_tp: list[tuple[float, bool]] = []
    total_gt = 0

    for pred_boxes, pred_scores, gt_boxes in zip(
        all_pred_boxes, all_pred_scores, all_gt_boxes
    ):
        tp, _fp, num_gt = _match_predictions(pred_boxes, pred_scores, gt_boxes, iou_threshold)
        total_gt += num_gt
        for score, is_tp in zip(pred_scores.tolist(), tp.tolist()):
            scored_tp.append((score, is_tp))

    if total_gt == 0 or len(scored_tp) == 0:
        return 0.0

    scored_tp.sort(key=lambda x: x[0], reverse=True)
    tp_cum = np.cumsum([1 if is_tp else 0 for _, is_tp in scored_tp])
    fp_cum = np.cumsum([0 if is_tp else 1 for _, is_tp in scored_tp])

    recalls = tp_cum / total_gt
    precisions = tp_cum / (tp_cum + fp_cum)

    # 101-point interpolation (COCO-style).
    recall_levels = np.linspace(0, 1, 101)
    interpolated_precisions = np.zeros_like(recall_levels)
    for i, r in enumerate(recall_levels):
        mask = recalls >= r
        interpolated_precisions[i] = precisions[mask].max() if mask.any() else 0.0

    return float(interpolated_precisions.mean())


def compute_precision_recall(
    all_pred_boxes: list[torch.Tensor],
    all_pred_scores: list[torch.Tensor],
    all_gt_boxes: list[torch.Tensor],
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> tuple[float, float]:
    """Compute precision and recall at a fixed score/IoU threshold."""
    total_tp = 0
    total_fp = 0
    total_gt = 0

    for pred_boxes, pred_scores, gt_boxes in zip(
        all_pred_boxes, all_pred_scores, all_gt_boxes
    ):
        keep = pred_scores >= score_threshold
        boxes = pred_boxes[keep]
        scores = pred_scores[keep]
        tp, fp, num_gt = _match_predictions(boxes, scores, gt_boxes, iou_threshold)
        total_tp += int(tp.sum())
        total_fp += int(fp.sum())
        total_gt += num_gt

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    return precision, recall


def compute_map_50_95(
    all_pred_boxes: list[torch.Tensor],
    all_pred_scores: list[torch.Tensor],
    all_gt_boxes: list[torch.Tensor],
) -> float:
    """Average AP over IoU thresholds 0.50:0.05:0.95 (COCO-style mAP)."""
    thresholds = np.arange(0.5, 1.0, 0.05)
    aps = [
        compute_average_precision(all_pred_boxes, all_pred_scores, all_gt_boxes, t)
        for t in thresholds
    ]
    return float(np.mean(aps))


def compute_average_iou(
    all_pred_boxes: list[torch.Tensor],
    all_pred_scores: list[torch.Tensor],
    all_gt_boxes: list[torch.Tensor],
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> float:
    """Mean IoU over matched (true-positive) detections only."""
    matched_ious: list[float] = []

    for pred_boxes, pred_scores, gt_boxes in zip(
        all_pred_boxes, all_pred_scores, all_gt_boxes
    ):
        keep = pred_scores >= score_threshold
        boxes = pred_boxes[keep]
        scores = pred_scores[keep]
        if boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
            continue

        order = torch.argsort(scores, descending=True)
        ious = box_iou(boxes[order], gt_boxes)
        matched_gt: set[int] = set()
        for row in ious:
            best_iou, best_gt = torch.max(row, dim=0)
            best_gt = int(best_gt.item())
            if best_iou.item() >= iou_threshold and best_gt not in matched_gt:
                matched_gt.add(best_gt)
                matched_ious.append(float(best_iou.item()))

    return float(np.mean(matched_ious)) if matched_ious else 0.0


def compute_false_positives_per_image(
    all_pred_boxes: list[torch.Tensor],
    all_pred_scores: list[torch.Tensor],
    all_gt_boxes: list[torch.Tensor],
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> float:
    """Mean count of unmatched (false-positive) predictions per image."""
    total_fp = 0
    num_images = len(all_pred_boxes)

    for pred_boxes, pred_scores, gt_boxes in zip(
        all_pred_boxes, all_pred_scores, all_gt_boxes
    ):
        keep = pred_scores >= score_threshold
        boxes = pred_boxes[keep]
        scores = pred_scores[keep]
        _tp, fp, _num_gt = _match_predictions(boxes, scores, gt_boxes, iou_threshold)
        total_fp += int(fp.sum())

    return total_fp / num_images if num_images > 0 else 0.0
