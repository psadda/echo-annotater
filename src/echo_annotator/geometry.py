"""Polygon geometry: hit-testing for editing, and arc-length resampling, in image coordinates."""

from __future__ import annotations

import math

import numpy as np

Point = tuple[float, float]
Polygon = list[Point]

RESAMPLE_SPACING = 6.0  # px between vertices when a polygon is resampled...
RESAMPLE_MIN, RESAMPLE_MAX = 16, 32  # ...within these vertex counts


def polygon_perimeter(polygon: Polygon) -> float:
    return sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]))


def target_vertex_count(polygon: Polygon) -> int:
    """How many vertices to resample `polygon` to: about one per RESAMPLE_SPACING px, within limits."""
    return min(max(round(polygon_perimeter(polygon) / RESAMPLE_SPACING), RESAMPLE_MIN), RESAMPLE_MAX)


def resample_polygon(polygon: Polygon, count: int) -> Polygon:
    """`count` vertices evenly spaced by arc length around the closed polygon.

    The first vertex stays exactly where it was and the direction is kept, so vertex i keeps meaning
    "the point i/count of the way round" from one call to the next.
    """
    points = np.asarray(polygon, dtype=np.float64)
    ring = np.vstack([points, points[:1]])
    distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ring, axis=0), axis=1))])
    if distance[-1] == 0:
        return [tuple(map(float, points[0]))] * count
    wanted = np.linspace(0.0, distance[-1], count, endpoint=False)
    xs, ys = np.interp(wanted, distance, ring[:, 0]), np.interp(wanted, distance, ring[:, 1])
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def polygon_area(polygon: Polygon) -> float:
    """Unsigned area by the shoelace formula."""
    twice = sum(
        ax * by - bx * ay
        for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1])
    )
    return abs(twice) / 2


def nearest_vertex(polygons: list[Polygon], x: float, y: float, radius: float) -> tuple[int, int] | None:
    """(polygon index, vertex index) of the vertex closest to (x, y) within `radius`."""
    best, best_dist = None, radius
    for pi, polygon in enumerate(polygons):
        for vi, (vx, vy) in enumerate(polygon):
            dist = math.hypot(x - vx, y - vy)
            if dist <= best_dist:
                best, best_dist = (pi, vi), dist
    return best


def nearest_edge(
    polygons: list[Polygon], x: float, y: float, radius: float
) -> tuple[int, int, Point] | None:
    """(polygon index, edge index, closest point on that edge) within `radius`.

    Edge `i` runs from vertex `i` to vertex `i + 1` (wrapping), so a new vertex goes at index `i + 1`.
    """
    best, best_dist = None, radius
    for pi, polygon in enumerate(polygons):
        for ei, (ax, ay) in enumerate(polygon):
            bx, by = polygon[(ei + 1) % len(polygon)]
            dx, dy = bx - ax, by - ay
            length_sq = dx * dx + dy * dy
            t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length_sq))
            px, py = ax + t * dx, ay + t * dy
            dist = math.hypot(x - px, y - py)
            if dist <= best_dist:
                best, best_dist = (pi, ei, (px, py)), dist
    return best


def polygon_at(polygons: list[Polygon], x: float, y: float) -> int | None:
    """Index of the topmost (most recently added) polygon containing (x, y), by even-odd rule."""
    for pi in range(len(polygons) - 1, -1, -1):
        polygon, inside = polygons[pi], False
        for i, (ax, ay) in enumerate(polygon):
            bx, by = polygon[(i + 1) % len(polygon)]
            if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
                inside = not inside
        if inside:
            return pi
    return None
