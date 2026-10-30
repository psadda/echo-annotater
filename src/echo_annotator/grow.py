"""Grow a polygon within one frame with a morphological geodesic active contour.

The polygon's mask is inflated by a balloon force and stopped where the (smoothed) image has strong
edges, so on an echo frame it expands into the surrounding low-gradient tissue and halts at
bright boundaries such as the pericardium.
"""

from __future__ import annotations

import numpy as np
from skimage.measure import approximate_polygon, find_contours
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour

from .annotations import Polygon, rasterize
from .geometry import polygon_area

GROW_ITERATIONS = 10  # each iteration grows the mask by at most ~1 px, so this is the max growth per call
EDGE_SIGMA = 2.0  # px; smooths speckle before measuring edge strength
EDGE_ALPHA = 100.0  # how sharply edge strength stops the contour (image scaled to 0-1)
SIMPLIFY_TOLERANCE = 0.75  # px; keeps the grown polygon editable instead of one vertex per pixel
MIN_GROWTH = 1.0  # px^2; less than this counts as "nothing to grow"


def grow_polygon(
    gray: np.ndarray, polygon: Polygon, iterations: int = GROW_ITERATIONS
) -> Polygon | None:
    """Return `polygon` grown on `gray` (2-D uint8), or None if it could not grow.

    The result always contains the original polygon: growth never shrinks it.
    """
    height, width = gray.shape
    start = rasterize([polygon], height, width) > 0
    if not start.any():
        return None

    edges = inverse_gaussian_gradient(gray.astype(np.float64) / 255.0, alpha=EDGE_ALPHA, sigma=EDGE_SIGMA)
    level_set = morphological_geodesic_active_contour(
        edges, num_iter=iterations, init_level_set=start, smoothing=1, balloon=1
    )
    grown = level_set.astype(bool) | start

    result = _largest_outline(grown)
    if result is None or polygon_area(result) - polygon_area(polygon) < MIN_GROWTH:
        return None
    return result


def _largest_outline(mask: np.ndarray) -> Polygon | None:
    """Outer outline of `mask` as a simplified polygon in continuous image coordinates."""
    padded = np.pad(mask, 1).astype(np.float64)  # padding closes outlines that touch the border
    contours = find_contours(padded, 0.5)
    if not contours:
        return None
    outline = max(contours, key=lambda c: polygon_area([(x, y) for y, x in c]))
    outline = approximate_polygon(outline, tolerance=SIMPLIFY_TOLERANCE)[:-1]  # drop the repeated end point
    if len(outline) < 3:
        return None
    height, width = mask.shape
    # find_contours gives (row, col) at pixel centres of the padded array; undo the pad, then +0.5 to
    # move from centre-indexed to the continuous coordinates used everywhere else.
    return [
        (float(min(max(col - 1 + 0.5, 0), width)), float(min(max(row - 1 + 0.5, 0), height)))
        for row, col in outline
    ]
