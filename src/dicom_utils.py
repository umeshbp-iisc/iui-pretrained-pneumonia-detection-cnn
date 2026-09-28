"""DICOM reading, normalization, and box-format conversion utilities.

Per CLAUDE.md / SPEC.md, this is the ONLY place raw CSV box format
(x, y, width, height) is converted to the internal Torchvision convention
(x_min, y_min, x_max, y_max).
"""
from __future__ import annotations

import numpy as np
import pydicom


def load_dicom_image(path: str) -> np.ndarray:
    """Read a DICOM file and return a normalized single-channel float image.

    Applies MONOCHROME1 inversion handling and percentile-based normalization
    to the range [0, 1].

    Args:
        path: Path to the .dcm file.

    Returns:
        A float32 array of shape (H, W) with values in [0, 1].
    """
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array.astype(np.float32)

    # MONOCHROME1 means higher values are darker; invert so higher = brighter,
    # matching MONOCHROME2 convention expected by downstream normalization.
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        pixels = pixels.max() - pixels

    return normalize_pixels(pixels)


def normalize_pixels(pixels: np.ndarray) -> np.ndarray:
    """Percentile-based normalization of a raw pixel array to [0, 1].

    Uses the 0.5th and 99.5th percentiles as clipping bounds to reduce the
    influence of outlier pixel intensities, then min-max scales to [0, 1].

    Args:
        pixels: Raw pixel array of any numeric dtype.

    Returns:
        A float32 array of the same shape, with values clipped to [0, 1].
    """
    pixels = pixels.astype(np.float32)
    lo, hi = np.percentile(pixels, [0.5, 99.5])
    if hi <= lo:
        # Degenerate (near-constant) image; avoid division by zero.
        return np.zeros_like(pixels, dtype=np.float32)
    normalized = (pixels - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def replicate_channels(image: np.ndarray) -> np.ndarray:
    """Replicate a single-channel (H, W) image into 3 channels (3, H, W).

    Args:
        image: Single-channel float array of shape (H, W).

    Returns:
        Float32 array of shape (3, H, W).
    """
    return np.repeat(image[np.newaxis, :, :], 3, axis=0).astype(np.float32)


def convert_box_format(
    x: float, y: float, w: float, h: float
) -> tuple[float, float, float, float]:
    """Convert a raw CSV box (x, y, width, height) to (x_min, y_min, x_max, y_max).

    Args:
        x: Top-left x coordinate.
        y: Top-left y coordinate.
        w: Box width.
        h: Box height.

    Returns:
        Tuple of (x_min, y_min, x_max, y_max).
    """
    return (float(x), float(y), float(x + w), float(y + h))
