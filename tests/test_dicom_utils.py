"""Tests for src/dicom_utils.py: box conversion and normalization."""
from __future__ import annotations

import numpy as np

from src.dicom_utils import convert_box_format, normalize_pixels, replicate_channels


def test_convert_box_format_basic() -> None:
    x_min, y_min, x_max, y_max = convert_box_format(10.0, 20.0, 30.0, 40.0)
    assert (x_min, y_min, x_max, y_max) == (10.0, 20.0, 40.0, 60.0)


def test_convert_box_format_zero_size() -> None:
    x_min, y_min, x_max, y_max = convert_box_format(5.0, 5.0, 0.0, 0.0)
    assert x_max == x_min
    assert y_max == y_min


def test_normalize_pixels_output_range() -> None:
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 4096, size=(64, 64)).astype(np.float32)
    normalized = normalize_pixels(pixels)
    assert normalized.dtype == np.float32
    assert normalized.shape == pixels.shape
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_normalize_pixels_constant_image() -> None:
    pixels = np.full((16, 16), 500.0, dtype=np.float32)
    normalized = normalize_pixels(pixels)
    assert np.all(normalized == 0.0)


def test_replicate_channels_shape() -> None:
    image = np.random.rand(32, 32).astype(np.float32)
    replicated = replicate_channels(image)
    assert replicated.shape == (3, 32, 32)
    assert np.array_equal(replicated[0], replicated[1])
    assert np.array_equal(replicated[1], replicated[2])
