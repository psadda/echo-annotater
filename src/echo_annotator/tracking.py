"""Propagate a polygon through time with sparse Lucas-Kanade optical flow, kept smooth by regularization.

Tracking each vertex on its own is noisy: speckle gives every track about a pixel of independent error, which
on a densely sampled contour shows up as jagged outpouchings. So each step solves for the new vertex positions
x of the closed contour, given the start positions p and the measured flow d (target y = p + d):

    E(x) = sum_i w_i |x_i - y_i|^2                      data: follow the flow, trusting confident tracks
         + |D1 u|^2 * a + |D2 u|^2 * b,   u = x - p     neighbouring vertices must move alike (elastic + bending)
         + lam * |L x - s R L r|^2                      optional: keep the reference shape's local curvature

D1 and D2 are the first and second differences around the contour and L is the umbrella Laplacian. The smoothness
terms act on the displacement u, not on the shape, so a pure translation is free and the contour cannot shrink.
w_i is zero where a track failed, so those vertices are interpolated from their neighbours. Vertices are
re-spaced evenly by arc length after every step, which also removes sliding along the contour (only the normal
component of the motion of a boundary is measurable anyway).
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from .annotations import Polygon
from .geometry import polygon_area, polygon_perimeter, resample_polygon, target_vertex_count

WINDOW = (31, 31)  # LK window; large enough to lock onto echo speckle
PYRAMID_LEVELS = 3
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
FORWARD_BACKWARD_TOLERANCE = 1.5  # px; a track must return to its start when run backwards, or it gets no weight
MIN_AREA = 4.0  # px^2; below this the tracked polygon has collapsed and tracking stops

SMOOTHING_LENGTH = 15.0  # px along the contour over which displacements are made to agree
CURVATURE_WEIGHT = 1.0  # lam, relative to a confident data point (weight 1); used only if preserve_curvature
ROBUST_SCALE = 1.5  # px; flow vectors further than this from the fit are down-weighted (Cauchy)
ROBUST_ITERATIONS = 3


def propagate_polygon(
    gray_at: Callable[[int], np.ndarray],
    n_frames: int,
    source: int,
    polygon: Polygon,
    preserve_curvature: bool = False,
) -> dict[int, Polygon]:
    """Track `polygon` from frame `source` forward to the last frame and backward to the first.

    `gray_at(i)` returns frame i as a 2-D uint8 image. The tracked polygons have evenly spaced vertices (16-32 of
    them), not the source polygon's own. Each direction stops early if tracking is lost, so the result may cover
    fewer than all other frames. Frame `source` itself is not included.

    With `preserve_curvature`, each step also pulls the contour toward the source polygon's local curvature
    (rotated and scaled by the structure's overall motion), which suppresses outpouchings without flattening
    real bends, but resists genuine changes of shape.
    """
    reference = resample_polygon(polygon, target_vertex_count(polygon))
    result: dict[int, Polygon] = {}
    for step in (1, -1):
        current, frame = reference, source
        prev_image = gray_at(frame)
        while 0 <= frame + step < n_frames:
            next_image = gray_at(frame + step)
            current = _track_step(prev_image, next_image, current, reference if preserve_curvature else None)
            if current is None:
                break
            frame += step
            result[frame] = current
            prev_image = next_image
    return result


def _track_step(
    prev: np.ndarray, nxt: np.ndarray, polygon: Polygon, reference: Polygon | None
) -> Polygon | None:
    """Move `polygon` from image `prev` to `nxt`; None if tracking failed."""
    height, width = prev.shape
    # OpenCV puts pixel centres on integer coordinates; ours are offset by half a pixel.
    start = np.asarray(polygon, dtype=np.float32) - 0.5
    moved, ok_forward, _ = cv2.calcOpticalFlowPyrLK(prev, nxt, start.reshape(-1, 1, 2), None, winSize=WINDOW, maxLevel=PYRAMID_LEVELS, criteria=CRITERIA)
    back, ok_back, _ = cv2.calcOpticalFlowPyrLK(nxt, prev, moved, None, winSize=WINDOW, maxLevel=PYRAMID_LEVELS, criteria=CRITERIA)
    moved, back = moved.reshape(-1, 2), back.reshape(-1, 2)

    error = np.linalg.norm(back - start, axis=1)
    confidence = np.where(
        (ok_forward.ravel() == 1) & (ok_back.ravel() == 1),
        np.clip(1 - error / FORWARD_BACKWARD_TOLERANCE, 0.0, 1.0),
        0.0,
    )
    if not (confidence > 0).any():
        return None

    reference_points = None if reference is None else np.asarray(reference, dtype=np.float64) - 0.5
    solved = _regularized_positions(start.astype(np.float64), moved.astype(np.float64), confidence, reference_points)
    solved = np.asarray(resample_polygon([tuple(p) for p in solved], len(solved)))  # re-space; vertex 0 stays put
    solved = solved + 0.5
    solved[:, 0] = np.clip(solved[:, 0], 0, width)
    solved[:, 1] = np.clip(solved[:, 1], 0, height)
    tracked = [(float(x), float(y)) for x, y in solved]
    return tracked if polygon_area(tracked) >= MIN_AREA else None


def _regularized_positions(
    start: np.ndarray, target: np.ndarray, confidence: np.ndarray, reference: np.ndarray | None
) -> np.ndarray:
    """New vertex positions minimizing the energy in the module docstring (closed contour, all (n, 2))."""
    n = len(start)
    identity = np.eye(n)
    after, before = np.roll(identity, -1, axis=0), np.roll(identity, 1, axis=0)
    d1, d2 = after - identity, after - 2 * identity + before
    laplacian = identity - 0.5 * (after + before)

    spacing = polygon_perimeter([tuple(p) for p in start]) / n
    a = (SMOOTHING_LENGTH / max(spacing, 1e-6)) ** 2  # smoothing scales with arc length, not vertex count
    b = a * a / 4  # with a, gives a flat low-pass cut-off at about SMOOTHING_LENGTH
    smooth = a * d1.T @ d1 + b * d2.T @ d2

    lam, shape_target = 0.0, np.zeros_like(start)
    if reference is not None:
        lam = CURVATURE_WEIGHT
        shape_target = _reference_laplacian(reference, target, confidence, laplacian)

    robust = np.ones(n)
    positions = target
    for _ in range(ROBUST_ITERATIONS):
        weight = confidence * robust
        # Smoothness is measured against the contour's overall (similarity) motion, so rotating, scaling or
        # translating the whole structure is free; only deviations from that motion are smoothed.
        scale, rotation, source_mean, target_mean = _fit_similarity(start, target, weight)
        moved_start = scale * (start - source_mean) @ rotation.T + target_mean
        system = np.diag(weight) + smooth + lam * laplacian.T @ laplacian + 1e-9 * identity
        rhs = weight[:, None] * target + smooth @ moved_start + lam * laplacian.T @ shape_target
        positions = np.linalg.solve(system, rhs)
        residual = np.linalg.norm(positions - target, axis=1)
        robust = 1.0 / (1.0 + (residual / ROBUST_SCALE) ** 2)  # iteratively reweighted: shrug off bad tracks
    return positions


def _fit_similarity(
    source: np.ndarray, target: np.ndarray, weight: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted least-squares scale + rotation + translation taking `source` to `target` (Umeyama).

    Returns (scale, rotation, source_mean, target_mean) so that target ~ scale * (source - source_mean) @ rotation.T
    + target_mean. With under 3 points of weight it falls back to a pure shift.
    """
    total = weight.sum()
    if total <= 0:
        return 1.0, np.eye(2), source.mean(0), source.mean(0)
    w = weight / total
    source_mean, target_mean = w @ source, w @ target
    if (weight > 0).sum() < 3:
        return 1.0, np.eye(2), source_mean, target_mean
    src, dst = source - source_mean, target - target_mean
    u, sigma, vt = np.linalg.svd((dst * w[:, None]).T @ src)
    sign = np.sign(np.linalg.det(u) * np.linalg.det(vt)) or 1.0
    rotation = u @ np.diag([1.0, sign]) @ vt
    spread = (w[:, None] * src**2).sum()
    scale = (sigma[0] + sign * sigma[1]) / spread if spread > 0 else 1.0
    return float(scale), rotation, source_mean, target_mean


def _reference_laplacian(
    reference: np.ndarray, target: np.ndarray, confidence: np.ndarray, laplacian: np.ndarray
) -> np.ndarray:
    """The reference shape's Laplacian coordinates, rotated and scaled by its weighted similarity fit to `target`."""
    scale, rotation, _, _ = _fit_similarity(reference, target, confidence)
    return scale * (laplacian @ reference) @ rotation.T
