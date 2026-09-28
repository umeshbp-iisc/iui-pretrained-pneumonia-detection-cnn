"""Train the SPEC-1 Faster R-CNN baseline (or DINOv2 fusion, once implemented)."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import RSNAPneumoniaDataset, collate_fn
from src.metrics import compute_average_precision, compute_map_50_95
from src.models.dino_fusion_detector import build_dino_fusion_model
from src.models.faster_rcnn_baseline import build_baseline_model
from src.transforms import get_eval_transforms, get_train_transforms

logger = logging.getLogger("train")


def setup_logging(log_file: str | Path) -> None:
    """Log to both console and an append-mode file so status survives session loss."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"logging to {log_path}")


def build_optimizer(
    model: torch.nn.Module, name: str, lr: float, weight_decay: float = 0.0005
) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, name: str, epochs: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unknown scheduler: {name}")


@torch.no_grad()
def evaluate_metrics(model: torch.nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    all_pred_boxes, all_pred_scores, all_gt_boxes = [], [], []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            all_pred_boxes.append(out["boxes"].cpu())
            all_pred_scores.append(out["scores"].cpu())
            all_gt_boxes.append(tgt["boxes"])
    map50 = compute_average_precision(all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold=0.5)
    map50_95 = compute_map_50_95(all_pred_boxes, all_pred_scores, all_gt_boxes)
    return {"mAP@0.50": map50, "mAP@0.50:0.95": map50_95}


def save_training_plots(history: dict[str, list[float]], output_dir: str | Path) -> None:
    """Save training history as JSON and a four-panel PNG summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "history.json", "w") as history_file:
        json.dump(history, history_file, indent=2)

    epochs = history["epoch"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, history["train_loss"], marker="o")
    axes[0, 0].set_title("Training loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].plot(epochs, history["val_map50"], marker="o", label="mAP@0.50")
    axes[0, 1].plot(epochs, history["val_map50_95"], marker="o", label="mAP@0.50:0.95")
    axes[0, 1].set_title("Validation detection metrics")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("mAP")
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, history["learning_rate"], marker="o", color="tab:orange")
    axes[1, 0].set_title("Learning rate")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("LR")
    axes[1, 1].plot(epochs, history["train_loss"], marker="o", label="loss")
    axes[1, 1].plot(epochs, history["val_map50"], marker="o", label="mAP@0.50")
    axes[1, 1].set_title("Loss / primary metric")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    figure.tight_layout()
    figure.savefig(output_path / "training_curves.png", dpi=160)
    plt.close(figure)


def backup_checkpoint(checkpoint_path: str | Path, epoch: int, map50: float) -> str:
    """Copy the just-saved best checkpoint into a timestamped backups folder."""
    checkpoint_path = Path(checkpoint_path)
    backups_dir = checkpoint_path.parent / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    backup_path = (
        backups_dir / f"{checkpoint_path.stem}_epoch{epoch:02d}_map{map50:.4f}{checkpoint_path.suffix}"
    )
    shutil.copy2(checkpoint_path, backup_path)
    return str(backup_path)


def train(
    images_dir: str,
    train_csv: str,
    val_csv: str,
    model_name: str = "baseline",
    backbone: str = "resnet50_fpn_v2",
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 0.005,
    weight_decay: float = 0.0005,
    optimizer_name: str = "sgd",
    scheduler_name: str = "step",
    warmup_iters: int = 500,
    grad_clip_norm: float = 5.0,
    min_size: int = 512,
    max_size: int = 768,
    pretrained: bool = True,
    init_from: str | None = None,
    checkpoint_path: str = "outputs/checkpoints/baseline_best.pt",
    plots_dir: str | None = None,
    log_file: str | None = None,
    device: str | None = None,
) -> None:
    if model_name not in ("baseline", "dino_fusion"):
        raise ValueError(f"Unsupported --model for this script: {model_name}")

    log_file = log_file or str(
        Path("outputs/logs") / f"{Path(checkpoint_path).stem}_train.log"
    )
    setup_logging(log_file)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device} epochs={epochs} batch_size={batch_size} lr={lr}")

    train_dataset = RSNAPneumoniaDataset(
        images_dir=images_dir, labels_csv=train_csv, transforms=get_train_transforms()
    )
    val_dataset = RSNAPneumoniaDataset(
        images_dir=images_dir, labels_csv=val_csv, transforms=get_eval_transforms()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    if model_name == "dino_fusion":
        baseline_checkpoint = init_from or "outputs/checkpoints/baseline_best.pt"
        model = build_dino_fusion_model(
            baseline_checkpoint=baseline_checkpoint, backbone=backbone, min_size=min_size, max_size=max_size
        )
        logger.info(f"built dino_fusion model initialized from baseline checkpoint {baseline_checkpoint}")
    else:
        model = build_baseline_model(
            backbone=backbone, min_size=min_size, max_size=max_size, pretrained=pretrained
        )
        logger.info(f"pretrained={pretrained}")
        if init_from is not None:
            init_path = Path(init_from)
            if not init_path.exists():
                raise FileNotFoundError(f"Initial checkpoint not found: {init_from}")
            model.load_state_dict(torch.load(init_path, map_location="cpu"))
            logger.info(f"initialized model weights from {init_from}")
    model.to(device)

    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, scheduler_name, epochs)
    scaler = torch.amp.GradScaler(device=device, enabled=(device == "cuda"))

    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = plots_dir or str(
        Path(checkpoint_path).parent / f"{Path(checkpoint_path).stem}_training"
    )

    global_step = 0
    best_map50 = -1.0
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_map50": [],
        "val_map50_95": [],
        "learning_rate": [],
    }
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        logger.info(f"epoch {epoch} started")
        for step, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Linear warmup only during the first `warmup_iters` steps overall.
            if global_step < warmup_iters:
                warmup_scale = (global_step + 1) / warmup_iters
                for group in optimizer.param_groups:
                    group["lr"] = lr * warmup_scale

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch} step {step}: {loss_dict}"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            if step % 20 == 0:
                logger.info(f"epoch {epoch} step {step} loss={loss.item():.4f}")

        if scheduler is not None and global_step >= warmup_iters:
            scheduler.step()

        val_metrics = evaluate_metrics(model, val_loader, device)
        val_map50 = val_metrics["mAP@0.50"]
        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(
            f"epoch {epoch} COMPLETE avg_train_loss={avg_train_loss:.4f} "
            f"val_mAP@0.50={val_map50:.4f} val_mAP@0.50:0.95={val_metrics['mAP@0.50:0.95']:.4f}"
        )
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_map50"].append(val_map50)
        history["val_map50_95"].append(val_metrics["mAP@0.50:0.95"])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        # Persist history/plots after every epoch so progress survives an interruption.
        save_training_plots(history, plots_dir)

        if val_map50 > best_map50:
            best_map50 = val_map50
            torch.save(model.state_dict(), checkpoint_path)
            backup_path = backup_checkpoint(checkpoint_path, epoch, val_map50)
            logger.info(
                f"saved new best checkpoint to {checkpoint_path} (mAP@0.50={best_map50:.4f}); "
                f"backup: {backup_path}"
            )

    save_training_plots(history, plots_dir)
    logger.info(f"training complete. best val mAP@0.50={best_map50:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="baseline", choices=["baseline", "dino_fusion"])
    parser.add_argument("--backbone", default="resnet50_fpn_v2", choices=["resnet50_fpn_v2", "mobilenet_v3"])
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--scheduler", default="step", choices=["none", "step", "cosine"])
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--max-size", type=int, default=768)
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Build a randomly-initialized backbone (SPEC-2 ablation lower bound)",
    )
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--checkpoint-path", default="outputs/checkpoints/baseline_best.pt")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    train(
        images_dir=args.images_dir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        model_name=args.model,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        warmup_iters=args.warmup_iters,
        grad_clip_norm=args.grad_clip_norm,
        min_size=args.min_size,
        max_size=args.max_size,
        pretrained=args.pretrained,
        init_from=args.init_from,
        checkpoint_path=args.checkpoint_path,
        plots_dir=args.plots_dir,
        log_file=args.log_file,
    )
