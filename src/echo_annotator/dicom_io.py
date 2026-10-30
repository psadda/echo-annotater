"""DICOM loading helpers: decode a file into displayable frames and list sibling files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.misc import is_dicom
from pydicom.multival import MultiValue

DEFAULT_FPS = 30.0
MIN_FPS, MAX_FPS = 1.0, 240.0


class DicomLoadError(Exception):
    """Raised when a file cannot be read as a DICOM with displayable pixel data."""


@dataclass
class Clip:
    path: Path
    frames: np.ndarray  # (N, H, W) grayscale or (N, H, W, 3) RGB, uint8
    fps: float

    @property
    def n_frames(self) -> int:
        return self.frames.shape[0]

    def gray(self, index: int) -> np.ndarray:
        """Frame `index` as a 2-D uint8 image (luma if the clip is RGB), for image-processing tools."""
        frame = self.frames[index]
        if frame.ndim == 2:
            return frame
        return np.dot(frame[..., :3], [0.299, 0.587, 0.114]).round().astype(np.uint8)


def load_clip(path: Path) -> Clip:
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array  # YBR photometrics are converted to RGB by pydicom
    except Exception as exc:
        raise DicomLoadError(f"{path.name}: {exc}") from exc

    n_frames = int(ds.get("NumberOfFrames", 1) or 1)
    if n_frames == 1:
        arr = arr[np.newaxis]
    if arr.ndim not in (3, 4) or (arr.ndim == 4 and arr.shape[-1] != 3):
        raise DicomLoadError(f"{path.name}: unsupported pixel array shape {arr.shape}")

    frames = _to_uint8(arr, ds)
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        frames = 255 - frames
    return Clip(path=path, frames=np.ascontiguousarray(frames), fps=_frame_rate(ds))


def _to_uint8(arr: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    data = arr.astype(np.float32)
    center, width = _first(ds.get("WindowCenter")), _first(ds.get("WindowWidth"))
    if center is not None and width:
        lo, hi = center - width / 2, center + width / 2
    else:
        lo, hi = float(data.min()), float(data.max())
    scale = 255.0 / (hi - lo) if hi > lo else 1.0
    return np.clip((data - lo) * scale, 0, 255).astype(np.uint8)


def _first(value) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, MultiValue):
        value = value[0]
    return float(value)


def _frame_rate(ds: pydicom.Dataset) -> float:
    try:
        frame_time_ms = float(ds.get("FrameTime", 0) or 0)  # (0018,1063)
        if frame_time_ms > 0:
            return min(max(1000.0 / frame_time_ms, MIN_FPS), MAX_FPS)
        for keyword in ("CineRate", "RecommendedDisplayFrameRate"):
            rate = float(ds.get(keyword, 0) or 0)
            if rate > 0:
                return min(max(rate, MIN_FPS), MAX_FPS)
    except (TypeError, ValueError):
        pass
    return DEFAULT_FPS


def _natural_key(name: str) -> list:
    # re.split with a capture group alternates str, int, str, ... so keys always compare like-for-like.
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def sibling_dicoms(path: Path) -> list[Path]:
    """DICOM files in `path`'s folder in ascending (natural) name order, always including `path`."""
    files = [p for p in path.parent.iterdir() if p.is_file() and (p == path or is_dicom(p))]
    files.sort(key=lambda p: _natural_key(p.name))
    return files
