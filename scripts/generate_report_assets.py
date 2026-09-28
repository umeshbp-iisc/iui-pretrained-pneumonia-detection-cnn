"""Generate report-ready assets: comparison charts, metrics CSV, and an asset
manifest, assembled into report_package/ for import into a document generator
(e.g. Copilot 365 Word).

Run once after all three ablation runs (random-init, baseline, dino_fusion)
have been trained and evaluated.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "report_package"

RUNS = {
    "Random-init": {
        "metrics_json": ROOT / "outputs/predictions/random_init_run/baseline_metrics.json",
        "training_history": ROOT / "outputs/checkpoints/random_init_best_training/history.json",
        "training_curves_png": ROOT / "outputs/checkpoints/random_init_best_training/training_curves.png",
        "prediction_images": sorted(
            (ROOT / "outputs/predictions/random_init_run").glob("baseline_pred_*.png")
        ),
    },
    "Pretrained Baseline": {
        "metrics_json": ROOT / "outputs/predictions/12k_run/baseline_metrics.json",
        "training_history": ROOT / "outputs/checkpoints/baseline_best_12k_training/history.json",
        "training_curves_png": ROOT / "outputs/checkpoints/baseline_best_12k_training/training_curves.png",
        "prediction_images": sorted(
            (ROOT / "outputs/predictions/12k_run").glob("baseline_pred_*.png")
        ),
    },
    "DINOv2 Fusion": {
        "metrics_json": ROOT / "outputs/predictions/dino_fusion_run/dino_fusion_metrics.json",
        "training_history": ROOT / "outputs/checkpoints/dino_fusion_best_training/history.json",
        "training_curves_png": ROOT / "outputs/checkpoints/dino_fusion_best_training/training_curves.png",
        "prediction_images": sorted(
            (ROOT / "outputs/predictions/dino_fusion_run").glob("dino_fusion_pred_*.png")
        ),
    },
}

METRIC_KEYS = [
    "mAP@0.50",
    "mAP@0.50:0.95",
    "precision",
    "recall",
    "avg_iou_matched",
    "false_positives_per_image",
]
METRIC_LABELS = [
    "mAP@0.50",
    "mAP@0.50:0.95",
    "Precision",
    "Recall",
    "Avg IoU",
    "FP/image",
]


def load_all_metrics() -> dict[str, dict]:
    metrics = {}
    for name, paths in RUNS.items():
        with open(paths["metrics_json"]) as f:
            metrics[name] = json.load(f)
    return metrics


def plot_comparison_bar_chart(metrics: dict[str, dict], output_path: Path) -> None:
    """Grouped bar chart comparing all models across every metric."""
    model_names = list(metrics.keys())
    x = np.arange(len(METRIC_KEYS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model_name in enumerate(model_names):
        values = [metrics[model_name][key] for key in METRIC_KEYS]
        ax.bar(x + i * width, values, width, label=model_name)

    ax.set_xticks(x + width)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylabel("Score")
    ax.set_title("Three-Way Ablation: Random-init vs. Pretrained Baseline vs. DINOv2 Fusion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_map_only_bar_chart(metrics: dict[str, dict], output_path: Path) -> None:
    """Focused chart on the two primary detection metrics, for a simpler slide."""
    model_names = list(metrics.keys())
    keys = ["mAP@0.50", "mAP@0.50:0.95"]
    x = np.arange(len(keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model_name in enumerate(model_names):
        values = [metrics[model_name][key] for key in keys]
        bars = ax.bar(x + i * width, values, width, label=model_name)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(keys)
    ax.set_ylabel("mAP")
    ax.set_title("Primary Detection Metrics Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_validation_map_overlay(output_path: Path) -> None:
    """Overlay validation mAP@0.50 curves for all three runs on one chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, paths in RUNS.items():
        with open(paths["training_history"]) as f:
            history = json.load(f)
        ax.plot(history["epoch"], history["val_map50"], marker="o", label=name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mAP@0.50")
    ax.set_title("Validation mAP@0.50 Across Training Epochs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_metrics_csv(metrics: dict[str, dict], output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + METRIC_LABELS)
        for model_name, model_metrics in metrics.items():
            writer.writerow([model_name] + [model_metrics[key] for key in METRIC_KEYS])


def copy_training_and_prediction_assets() -> None:
    for name, paths in RUNS.items():
        safe_name = name.lower().replace(" ", "_")

        training_dst = PACKAGE_DIR / "training_graphs" / f"{safe_name}_training_curves.png"
        shutil.copy2(paths["training_curves_png"], training_dst)

        pred_dir = PACKAGE_DIR / "prediction_images" / safe_name
        pred_dir.mkdir(parents=True, exist_ok=True)
        for i, img_path in enumerate(paths["prediction_images"][:5]):
            shutil.copy2(img_path, pred_dir / f"prediction_{i:02d}.png")


def write_content_outline(metrics: dict[str, dict]) -> None:
    lines = [
        "# Report Content Outline (paste into Copilot 365 Word)",
        "",
        "## 1. Title",
        "Pneumonia Opacity Detection: Pretrained Faster R-CNN with DINOv2 Semantic Fusion",
        "",
        "## 2. Objective",
        "Detect pneumonia opacity regions in chest X-rays using a pretrained "
        "Faster R-CNN ResNet-50-FPN v2 detector, enhanced with a frozen DINOv2 "
        "ViT-S/14 global-context fusion module.",
        "",
        "## 3. Dataset",
        "- RSNA Pneumonia Detection Challenge, DICOM chest X-rays",
        "- 12,000-patient stratified subset: 8,400 train / 1,799 val / 1,801 test",
        "- Splits stratified by patientId and positive-box presence, seed 42",
        "",
        "## 4. Architecture Diagrams",
        "See docs/REPORT.md Section 3 for the SPEC-1 and SPEC-2 architecture "
        "diagrams (insert as text/diagram blocks).",
        "",
        "## 5. Training Graphs (insert images from training_graphs/)",
        "- random-init_training_curves.png",
        "- pretrained_baseline_training_curves.png",
        "- dinov2_fusion_training_curves.png",
        "- validation_map_overlay.png (all three runs on one chart)",
        "",
        "## 6. Comparison Graphs (insert images from comparison_graphs/)",
        "- three_way_metric_comparison.png (all 6 metrics, grouped bar chart)",
        "- map_primary_metrics_comparison.png (mAP@0.50 and mAP@0.50:0.95 only)",
        "",
        "## 7. Metrics Table (insert metrics_table.csv as a table)",
    ]
    lines.append("")
    lines.append("| Model | " + " | ".join(METRIC_LABELS) + " |")
    lines.append("|---|" + "---|" * len(METRIC_LABELS))
    for model_name, model_metrics in metrics.items():
        row = [f"{model_metrics[key]:.4f}" for key in METRIC_KEYS]
        lines.append(f"| {model_name} | " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## 8. Qualitative Predictions (insert images from prediction_images/)",
            "Each subfolder (random-init, pretrained_baseline, dinov2_fusion) has "
            "5 examples with green=ground truth, red=prediction+confidence.",
            "",
            "## 9. Conclusion",
            "1. Pretraining clearly improves detection: pretrained Faster R-CNN "
            "reaches 2.5x the mAP@0.50 of random-init (0.4673 vs 0.1847).",
            "2. DINOv2 fusion shows no measurable mAP@0.50 improvement over the "
            "pretrained baseline (0.4672 vs 0.4673), but slightly improves "
            "mAP@0.50:0.95, precision, recall, and produces markedly more "
            "stable training (no overfitting collapse).",
            "3. This is consistent with the domain-shift hypothesis: DINOv2 was "
            "pretrained on natural images, not radiographs.",
            "",
            "## 10. Reproducibility",
            "See docs/REPORT.md Section 11 for exact commands.",
        ]
    )
    (PACKAGE_DIR / "content_outline.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PACKAGE_DIR.mkdir(exist_ok=True)
    (PACKAGE_DIR / "training_graphs").mkdir(exist_ok=True)
    (PACKAGE_DIR / "comparison_graphs").mkdir(exist_ok=True)
    (PACKAGE_DIR / "prediction_images").mkdir(exist_ok=True)
    (PACKAGE_DIR / "metrics").mkdir(exist_ok=True)

    metrics = load_all_metrics()

    plot_comparison_bar_chart(metrics, PACKAGE_DIR / "comparison_graphs" / "three_way_metric_comparison.png")
    plot_map_only_bar_chart(metrics, PACKAGE_DIR / "comparison_graphs" / "map_primary_metrics_comparison.png")
    plot_validation_map_overlay(PACKAGE_DIR / "training_graphs" / "validation_map_overlay.png")
    write_metrics_csv(metrics, PACKAGE_DIR / "metrics" / "metrics_table.csv")
    copy_training_and_prediction_assets()
    write_content_outline(metrics)

    print(f"Report package assembled at: {PACKAGE_DIR}")


if __name__ == "__main__":
    main()
