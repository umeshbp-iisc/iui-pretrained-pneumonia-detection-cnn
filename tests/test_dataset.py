"""Tests for src/dataset.py: shapes, negative images, multi-box grouping."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import src.dataset as dataset_module
from src.dataset import RSNAPneumoniaDataset, collate_fn


@pytest.fixture(autouse=True)
def _stub_dicom_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real DICOM I/O: return a deterministic fake image for any path."""

    def fake_load_dicom_image(path: str) -> np.ndarray:
        return np.zeros((128, 128), dtype=np.float32)

    monkeypatch.setattr(dataset_module, "load_dicom_image", fake_load_dicom_image)


def _write_labels_csv(tmp_path: Path, rows: list[dict]) -> Path:
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_negative_image_has_empty_boxes(tmp_path: Path) -> None:
    rows = [
        {"patientId": "patientA", "x": None, "y": None, "width": None, "height": None, "Target": 0}
    ]
    csv_path = _write_labels_csv(tmp_path, rows)
    ds = RSNAPneumoniaDataset(images_dir=tmp_path, labels_csv=csv_path)

    image, target = ds[0]
    assert image.shape == (3, 128, 128)
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)


def test_multi_box_grouping_by_patient_id(tmp_path: Path) -> None:
    rows = [
        {"patientId": "patientB", "x": 10, "y": 10, "width": 20, "height": 20, "Target": 1},
        {"patientId": "patientB", "x": 50, "y": 50, "width": 30, "height": 30, "Target": 1},
    ]
    csv_path = _write_labels_csv(tmp_path, rows)
    ds = RSNAPneumoniaDataset(images_dir=tmp_path, labels_csv=csv_path)

    assert len(ds) == 1
    image, target = ds[0]
    assert target["boxes"].shape == (2, 4)
    assert target["labels"].shape == (2,)
    assert torch.all(target["labels"] == 1)
    expected_boxes = torch.tensor(
        [[10.0, 10.0, 30.0, 30.0], [50.0, 50.0, 80.0, 80.0]]
    )
    assert torch.allclose(target["boxes"], expected_boxes)


def test_image_and_target_shapes(tmp_path: Path) -> None:
    rows = [
        {"patientId": "patientC", "x": 0, "y": 0, "width": 10, "height": 10, "Target": 1},
    ]
    csv_path = _write_labels_csv(tmp_path, rows)
    ds = RSNAPneumoniaDataset(images_dir=tmp_path, labels_csv=csv_path)

    image, target = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.float32
    assert target["image_id"].shape == (1,)
    assert target["area"].shape == (1,)
    assert target["iscrowd"].shape == (1,)


def test_collate_fn_preserves_variable_sizes(tmp_path: Path) -> None:
    rows = [
        {"patientId": "patientD", "x": None, "y": None, "width": None, "height": None, "Target": 0},
        {"patientId": "patientE", "x": 1, "y": 2, "width": 3, "height": 4, "Target": 1},
    ]
    csv_path = _write_labels_csv(tmp_path, rows)
    ds = RSNAPneumoniaDataset(images_dir=tmp_path, labels_csv=csv_path)

    batch = [ds[0], ds[1]]
    images, targets = collate_fn(batch)

    assert len(images) == 2
    assert len(targets) == 2
    assert targets[0]["boxes"].shape == (0, 4)
    assert targets[1]["boxes"].shape == (1, 4)
