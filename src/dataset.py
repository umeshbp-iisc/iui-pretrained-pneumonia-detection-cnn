"""RSNAPneumoniaDataset: the only sanctioned entry point for DICOM data access."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dicom_utils import convert_box_format, load_dicom_image, replicate_channels


class RSNAPneumoniaDataset(Dataset):
    """Groups CSV rows by patientId into one multi-box sample per image.

    Args:
        images_dir: Directory containing `<patientId>.dcm` files.
        labels_csv: Path to a CSV with columns
            `patientId, x, y, width, height, Target`, already filtered to the
            desired split (see scripts/create_splits.py).
        transforms: Optional callable `(image, target) -> (image, target)`.
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_csv: str | Path,
        transforms: Callable[..., Any] | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transforms = transforms

        df = pd.read_csv(labels_csv)
        self.labels_df = df
        # One dataset item per unique patientId, not per CSV row, so multi-box
        # positives are grouped into a single sample instead of duplicated.
        self.patient_ids = df["patientId"].drop_duplicates().reset_index(drop=True).tolist()

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        patient_id = self.patient_ids[idx]
        rows = self.labels_df[self.labels_df["patientId"] == patient_id]

        dcm_path = self.images_dir / f"{patient_id}.dcm"
        image = load_dicom_image(str(dcm_path))
        image = replicate_channels(image)
        image_tensor = torch.from_numpy(image).float()

        positive_rows = rows[rows["Target"] == 1]

        if len(positive_rows) == 0:
            # Target == 0 patients have no box rows; Faster R-CNN requires an
            # explicit empty [0, 4]/[0] tensor pair, not a missing key.
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            box_list = [
                convert_box_format(r.x, r.y, r.width, r.height)
                for r in positive_rows.itertuples()
            ]
            boxes = torch.tensor(box_list, dtype=torch.float32)
            labels = torch.ones((len(box_list),), dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.zeros((len(box_list),), dtype=torch.int64)

        target: dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]
) -> tuple[tuple[torch.Tensor, ...], tuple[dict[str, torch.Tensor], ...]]:
    """Torchvision detection convention collate: tuple(zip(*batch))."""
    return tuple(zip(*batch))
