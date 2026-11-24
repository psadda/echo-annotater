"""Grow a polygon within one frame with a morphological geodesic active contour.

The polygon's mask is inflated by a balloon force and stopped where the (smoothed) image has strong
edges, so on an echo frame it expands into the surrounding low-gradient tissue and halts at
bright boundaries such as the pericardium.

The snake follows Marquez-Neila, Baumela and Alvarez, "A Morphological Approach to Curvature-based
Evolution of Curves and Surfaces" (PAMI 2014), implemented here with OpenCV and numpy.
"""

from __future__ import annotations

import cv2
import numpy as np

from .annotations import Polygon, rasterize
from .geometry import polygon_area, resample_polygon, target_vertex_count

GROW_ITERATIONS = 10  # each iteration grows the mask by about a pixel, so this is roughly the growth per call
EDGE_SIGMA = 2.0  # px; smooths speckle before measuring edge strength
EDGE_ALPHA = 100.0  # how sharply edge strength stops the contour (image scaled to 0-1)
SIMPLIFY_TOLERANCE = 0.75  # px; keeps the grown polygon editable instead of one vertex per pixel
MIN_GROWTH = 1.0  # px^2; less than this counts as "nothing to grow"...
MIN_GROWTH_FRACTION = 0.02  # ...as does a gain of less than this fraction of the polygon's area
BALLOON_THRESHOLD_PERCENTILE = 40  # the balloon only inflates where edge-stopping is above this percentile

_BOX = np.ones((3, 3), np.uint8)
# Structuring elements of the curvature operators: the four lines through the centre of a 3x3 window.
_LINES = [
    np.eye(3, dtype=np.uint8),
    np.array([[0, 1, 0]] * 3, dtype=np.uint8),
    np.flipud(np.eye(3, dtype=np.uint8)),
    np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8),
]


def grow_polygon(
    gray: np.ndarray, polygon: Polygon, iterations: int = GROW_ITERATIONS
) -> Polygon | None:
    """Return `polygon` grown on `gray` (2-D uint8), or None if it could not grow.

    The grown region always contains the original (growth never shrinks it). The returned polygon has evenly
    spaced vertices, 16-32 of them, so it approximates that region to within a fraction of a pixel to a pixel or so.
    """
    height, width = gray.shape
    start = rasterize([polygon], height, width) > 0
    if not start.any():
        return None

    edges = _edge_stopping(gray.astype(np.float32) / 255.0, EDGE_ALPHA, EDGE_SIGMA)
    grown = _snake(edges, start, iterations) | start

    outline = _largest_outline(grown)
    if outline is None:
        return None
    # Evenly spaced vertices: far fewer than the pixel-level outline, and what the tracker works best with.
    result = resample_polygon(outline, target_vertex_count(outline))
    area = polygon_area(polygon)
    # Resampling cuts corners slightly, so a polygon that has filled its region still "regrows" a little
    # each time. Only a real gain counts, otherwise repeated presses would never report "nothing to grow".
    if polygon_area(result) - area < max(MIN_GROWTH, MIN_GROWTH_FRACTION * area):
        return None
    return result


def _edge_stopping(image: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    """1 / sqrt(1 + alpha * |gradient of the Gaussian-smoothed image|): near 1 in flat areas, near 0 on edges."""
    radius = int(4.0 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    smooth = np.exp(-0.5 * x**2 / sigma**2)
    smooth /= smooth.sum()
    derivative = -x / sigma**2 * smooth  # derivative of the Gaussian (sign is irrelevant for the magnitude)
    # Separable derivative-of-Gaussian filters; replicate the border as the image's own edge values.
    gy = cv2.sepFilter2D(image, -1, smooth, derivative, borderType=cv2.BORDER_REPLICATE)
    gx = cv2.sepFilter2D(image, -1, derivative, smooth, borderType=cv2.BORDER_REPLICATE)
    return 1.0 / np.sqrt(1.0 + alpha * np.hypot(gx, gy))


def _snake(edges: np.ndarray, start: np.ndarray, iterations: int) -> np.ndarray:
    """Morphological geodesic active contour (balloon = +1, one smoothing pass) from the mask `start`."""
    balloon_allowed = edges > np.percentile(edges, BALLOON_THRESHOLD_PERCENTILE)
    edge_dy, edge_dx = np.gradient(edges)
    u = start.astype(np.int8)
    smoothing_step = 0  # the two curvature operators alternate from one smoothing pass to the next
    for _ in range(iterations):
        # Balloon: inflate by one pixel wherever the edge-stopping function is large.
        inflated = _binary(cv2.dilate, u, _BOX)
        u[balloon_allowed] = inflated[balloon_allowed]
        # Image attachment: move the level set up the gradient of the edge-stopping function.
        du_dy, du_dx = np.gradient(u)
        pull = edge_dy * du_dy + edge_dx * du_dx
        u[pull > 0] = 1
        u[pull < 0] = 0
        # Curvature smoothing: SI after IS, then IS after SI, alternating.
        u = _sup_inf(_inf_sup(u)) if smoothing_step % 2 == 0 else _inf_sup(_sup_inf(u))
        smoothing_step += 1
    return u.astype(bool)


def _binary(op, u: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Binary dilation or erosion with everything outside the image counted as 0."""
    padded = np.pad(u.astype(np.uint8), 1)
    return op(padded, kernel)[1:-1, 1:-1].astype(np.int8)


def _sup_inf(u: np.ndarray) -> np.ndarray:
    return np.max([_binary(cv2.erode, u, line) for line in _LINES], axis=0).astype(np.int8)


def _inf_sup(u: np.ndarray) -> np.ndarray:
    return np.min([_binary(cv2.dilate, u, line) for line in _LINES], axis=0).astype(np.int8)


def _largest_outline(mask: np.ndarray) -> Polygon | None:
    """Outer outline of `mask` as a simplified polygon in continuous image coordinates.

    The outline follows the pixel edges between inside and outside, so it encloses exactly the mask's pixels.
    """
    inside = np.pad(mask, 1)  # padding closes outlines that touch the border
    # Directed unit edges with the inside on the right-hand side; a vertex (x, y) is a pixel corner.
    successors: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def add(rows, cols, start, end):
        for r, c in zip(rows.tolist(), cols.tolist()):
            successors.setdefault((c + start[0], r + start[1]), []).append((c + end[0], r + end[1]))

    add(*np.nonzero(inside[1:] & ~inside[:-1]), (0, 1), (1, 1))  # pixel's top edge (rows shift by one)
    add(*np.nonzero(inside[:-1] & ~inside[1:]), (1, 1), (0, 1))  # bottom edge
    add(*np.nonzero(inside[:, 1:] & ~inside[:, :-1]), (1, 1), (1, 0))  # left edge
    add(*np.nonzero(inside[:, :-1] & ~inside[:, 1:]), (1, 0), (1, 1))  # right edge
    if not successors:
        return None

    loops = []
    while successors:
        first = next(iter(successors))
        loop, current = [first], first
        while True:
            ends = successors[current]
            following = ends.pop()
            if not ends:
                del successors[current]
            if following == first:
                break
            loop.append(following)
            current = following
        loops.append(loop)

    outline = np.asarray(max(loops, key=lambda loop: _loop_area(np.asarray(loop))), dtype=np.float32)
    approx = cv2.approxPolyDP(outline.reshape(-1, 1, 2), SIMPLIFY_TOLERANCE, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    height, width = mask.shape
    # Undo the one-pixel padding; vertices are pixel corners, which are already continuous coordinates.
    return [
        (float(min(max(x - 1, 0), width)), float(min(max(y - 1, 0), height)))
        for x, y in approx
    ]


def _loop_area(points: np.ndarray) -> float:
    """Absolute shoelace area (holes and the outer boundary have opposite signs, so use the larger magnitude)."""
    x, y = points[:, 0].astype(np.float64), points[:, 1].astype(np.float64)
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2
