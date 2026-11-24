"""Clip loading helpers: decode an AVI video into displayable frames and list sibling files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_FPS = 30.0
MIN_FPS, MAX_FPS = 1.0, 240.0
VIDEO_SUFFIXES = {".avi"}
GRAY_TOLERANCE = 3  # max difference between colour channels (of 255) for a video to count as grayscale


class ClipLoadError(Exception):
    """Raised when a file cannot be read as a video with displayable frames."""


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
    if not is_video(path):
        raise ClipLoadError(f"{path.name}: only .avi videos are supported")
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


def _clamp_fps(fps: float) -> float:
    """`fps` limited to a sane range; a missing or invalid rate (<= 0 or NaN) becomes the default."""
    if not fps > 0:
        return DEFAULT_FPS
    return min(max(fps, MIN_FPS), MAX_FPS)


def _natural_key(name: str) -> list:
    # re.split with a capture group alternates str, int, str, ... so keys always compare like-for-like.
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def sibling_clips(path: Path) -> list[Path]:
    """AVI files in `path`'s folder in ascending (natural) name order, always including `path`."""
    files = [p for p in path.parent.iterdir() if p.is_file() and (p == path or is_video(p))]
    files.sort(key=lambda p: _natural_key(p.name))
    return files
