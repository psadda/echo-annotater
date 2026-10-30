"""Propagate a polygon through time with sparse Lucas-Kanade optical flow."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from .annotations import Polygon
from .geometry import polygon_area

WINDOW = (31, 31)  # LK window; large enough to lock onto echo speckle
PYRAMID_LEVELS = 3
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
FORWARD_BACKWARD_TOLERANCE = 1.5  # px; a track must return to its start when run backwards
RANSAC_THRESHOLD = 3.0  # px
MIN_AREA = 4.0  # px^2; below this the tracked polygon has collapsed and tracking stops


def propagate_polygon(
    gray_at: Callable[[int], np.ndarray], n_frames: int, source: int, polygon: Polygon
) -> dict[int, Polygon]:
    """Track `polygon` from frame `source` forward to the last frame and backward to the first.

    `gray_at(i)` returns frame i as a 2-D uint8 image. Each direction stops early if tracking is lost,
    so the result may cover fewer than all other frames. Frame `source` itself is not included.
    """
    result: dict[int, Polygon] = {}
    for step in (1, -1):
        current, frame = polygon, source
        prev_image = gray_at(frame)
        while 0 <= frame + step < n_frames:
            next_image = gray_at(frame + step)
            current = _track_step(prev_image, next_image, current)
            if current is None:
                break
            frame += step
            result[frame] = current
            prev_image = next_image
    return result


def _track_step(prev: np.ndarray, nxt: np.ndarray, polygon: Polygon) -> Polygon | None:
    """Move `polygon`'s vertices from image `prev` to `nxt`; None if tracking failed."""
    height, width = prev.shape
    # OpenCV puts pixel centres on integer coordinates; ours are offset by half a pixel.
    start = (np.asarray(polygon, dtype=np.float32) - 0.5).reshape(-1, 1, 2)
    moved, ok_forward, _ = cv2.calcOpticalFlowPyrLK(prev, nxt, start, None, winSize=WINDOW, maxLevel=PYRAMID_LEVELS, criteria=CRITERIA)
    back, ok_back, _ = cv2.calcOpticalFlowPyrLK(nxt, prev, moved, None, winSize=WINDOW, maxLevel=PYRAMID_LEVELS, criteria=CRITERIA)
    start, moved, back = start.reshape(-1, 2), moved.reshape(-1, 2), back.reshape(-1, 2)

    good = (
        (ok_forward.ravel() == 1)
        & (ok_back.ravel() == 1)
        & (np.linalg.norm(back - start, axis=1) < FORWARD_BACKWARD_TOLERANCE)
    )
    if not good.any():
        return None

    # Vertices whose own track failed follow the motion of the ones that worked.
    result = moved.copy()
    if not good.all():
        result[~good] = _fallback_motion(start, moved, good)[~good]

    result = result + 0.5
    result[:, 0] = np.clip(result[:, 0], 0, width)
    result[:, 1] = np.clip(result[:, 1], 0, height)
    tracked = [(float(x), float(y)) for x, y in result]
    return tracked if polygon_area(tracked) >= MIN_AREA else None


def _fallback_motion(start: np.ndarray, moved: np.ndarray, good: np.ndarray) -> np.ndarray:
    """Predicted positions of every vertex from the good tracks: a similarity fit, or a shift if too few."""
    if good.sum() >= 3:
        matrix, _ = cv2.estimateAffinePartial2D(
            start[good], moved[good], method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD
        )
        if matrix is not None:
            return start @ matrix[:, :2].T + matrix[:, 2]
    return start + np.median(moved[good] - start[good], axis=0)
