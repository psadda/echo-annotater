"""Clip loading helpers: decode a DICOM or AVI file into displayable frames and list sibling files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pydicom
from pydicom.misc import is_dicom
from pydicom.multival import MultiValue

DEFAULT_FPS = 30.0
MIN_FPS, MAX_FPS = 1.0, 240.0
VIDEO_SUFFIXES = {".avi"}
GRAY_TOLERANCE = 3  # max difference between colour channels (of 255) for a video to count as grayscale


class ClipLoadError(Exception):
    """Raised when a file cannot be read as a DICOM or video with displayable frames."""


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


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def load_clip(path: Path) -> Clip:
    return _load_video(path) if is_video(path) else _load_dicom(path)


def _load_video(path: Path) -> Clip:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ClipLoadError(f"{path.name}: could not open the video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        # The container's frame count is unreliable for AVI, so read until the stream ends.
        bgr = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            bgr.append(frame)
    finally:
        capture.release()
    if not bgr:
        raise ClipLoadError(f"{path.name}: no frames could be decoded")

    frames = np.stack(bgr)[..., ::-1]  # BGR -> RGB
    if np.ptp(frames, axis=-1).max() <= GRAY_TOLERANCE:
        # Echo videos are usually grayscale stored as 3 channels (lossy codecs leave small channel
        # differences), so keep one channel; this also cuts memory and speeds up the image tools.
        frames = frames[..., 1]
    return Clip(path=path, frames=np.ascontiguousarray(frames), fps=_clamp_fps(fps))


def _load_dicom(path: Path) -> Clip:
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array  # YBR photometrics are converted to RGB by pydicom
    except Exception as exc:
        raise ClipLoadError(f"{path.name}: {exc}") from exc

    n_frames = int(ds.get("NumberOfFrames", 1) or 1)
    if n_frames == 1:
        arr = arr[np.newaxis]
    if arr.ndim not in (3, 4) or (arr.ndim == 4 and arr.shape[-1] != 3):
        raise ClipLoadError(f"{path.name}: unsupported pixel array shape {arr.shape}")

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


def _clamp_fps(fps: float) -> float:
    """`fps` limited to a sane range; a missing or invalid rate (<= 0 or NaN) becomes the default."""
    if not fps > 0:
        return DEFAULT_FPS
    return min(max(fps, MIN_FPS), MAX_FPS)


def _frame_rate(ds: pydicom.Dataset) -> float:
    try:
        frame_time_ms = float(ds.get("FrameTime", 0) or 0)  # (0018,1063)
        if frame_time_ms > 0:
            return _clamp_fps(1000.0 / frame_time_ms)
        for keyword in ("CineRate", "RecommendedDisplayFrameRate"):
            rate = float(ds.get(keyword, 0) or 0)
            if rate > 0:
                return _clamp_fps(rate)
    except (TypeError, ValueError):
        pass
    return DEFAULT_FPS


def _natural_key(name: str) -> list:
    # re.split with a capture group alternates str, int, str, ... so keys always compare like-for-like.
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def sibling_clips(path: Path) -> list[Path]:
    """DICOM and AVI files in `path`'s folder in ascending (natural) name order, always including `path`."""
    files = [p for p in path.parent.iterdir() if p.is_file() and (p == path or is_video(p) or is_dicom(p))]
    files.sort(key=lambda p: _natural_key(p.name))
    return files
