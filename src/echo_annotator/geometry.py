"""Hit-testing for polygon editing, in image coordinates."""

from __future__ import annotations

import math

Point = tuple[float, float]
Polygon = list[Point]


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
