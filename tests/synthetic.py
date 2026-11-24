"""Synthetic echo-like data: speckled textures, moving scenes, and an AVI writer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

H, W = 120, 160


def speckle_texture(h: int = H, w: int = W, seed: int = 0, sigma: float = 1.5, lo: int = 40, hi: int = 220) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), sigma)
    return ((tex - tex.min()) / (tex.max() - tex.min()) * (hi - lo) + lo).astype(np.uint8)


def drifting_frames(n: int, h: int = H, w: int = W, seed: int = 0) -> list[np.ndarray]:
    """`n` uint8 frames of one texture drifting by (1.0, 0.5) px per frame."""
    tex = speckle_texture(h, w, seed)
    return [
        cv2.warpAffine(tex, np.float32([[1, 0, 1.0 * t], [0, 1, 0.5 * t]]), (w, h), borderMode=cv2.BORDER_REFLECT)
        for t in range(n)
    ]


def numbered_frames(n: int, h: int = H, w: int = W) -> list[np.ndarray]:
    """Frame i is uniformly (i + 1) * 10, so frame identity can be read from any pixel."""
    return [np.full((h, w), (i + 1) * 10, np.uint8) for i in range(n)]


def write_avi(path: Path, frames: list[np.ndarray], fps: float = 25.0, *, color: bool = False) -> Path:
    """Write an MJPG AVI. Grayscale frames are stored as three equal channels, like real echo exports."""
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h), True)
    assert writer.isOpened(), "OpenCV cannot write MJPG AVI here"
    for f in frames:
        if color:
            writer.write(np.dstack([f, (f * 0.5).astype(np.uint8), np.full_like(f, 200)]))  # clearly not gray
        else:
            writer.write(np.dstack([f, f, f]))
    writer.release()
    return path


# ---- a scene for the image-processing tools: a dark stripe between two bright bands, drifting and rotating ----

SCENE_H, SCENE_W = 240, 320
STRIPE_ROWS = (100, 122)  # dark band in frame 0


def scene_motion(t: int) -> np.ndarray:
    """3x3 matrix taking frame 0 to frame t: slow drift plus slight rotation and scale."""
    m = cv2.getRotationMatrix2D((160, 110), 0.3 * t, 1 + 0.003 * t)
    m[:, 2] += (0.9 * t, -0.5 * t + 3 * np.sin(t / 4))
    return np.vstack([m, [0, 0, 1]])


def scene_frames(n: int, seed: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    scene = np.full((SCENE_H, SCENE_W), 110.0)
    scene[90:100, 20:300] = 210
    scene[STRIPE_ROWS[0] : STRIPE_ROWS[1], 20:300] = 45
    scene[122:132, 20:300] = 210
    speckle = cv2.GaussianBlur(rng.gamma(6.0, 1 / 6.0, (SCENE_H, SCENE_W)).astype(np.float32), (0, 0), 1.0)
    scene = np.clip(scene * speckle, 0, 255).astype(np.float32)
    return [
        cv2.warpAffine(scene, scene_motion(t)[:2], (SCENE_W, SCENE_H), borderMode=cv2.BORDER_REFLECT).astype(np.uint8)
        for t in range(n)
    ]


def stripe_mask(t: int) -> np.ndarray:
    """Where the dark stripe is in frame t (1 inside)."""
    mask = np.zeros((SCENE_H, SCENE_W), np.uint8)
    mask[STRIPE_ROWS[0] : STRIPE_ROWS[1], 20:300] = 1
    return cv2.warpAffine(mask, scene_motion(t)[:2], (SCENE_W, SCENE_H), flags=cv2.INTER_NEAREST)


def curve_distance(a, b, count: int = 200) -> float:
    """Mean symmetric distance between two closed curves (point lists), via arc-length resampling."""
    from echo_annotator.geometry import resample_polygon

    a, b = np.asarray(resample_polygon(a, count)), np.asarray(resample_polygon(b, count))
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    return 0.5 * (d.min(1).mean() + d.min(0).mean())
