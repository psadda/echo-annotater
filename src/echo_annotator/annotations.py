"""Per-frame polygon annotations: in-memory model, rasterization, and sidecar persistence.

Polygons are stored in continuous image coordinates, where pixel (col, row) covers
[col, col + 1) x [row, row + 1). Next to `<name>` one file is written, `<name>.annotations.json`,
holding the editable polygons. Masks are not saved; generate them from the JSON with `rasterize`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LABEL = "pericardial_fat"
FORMAT_VERSION = 1
MIN_VERTICES = 3
MAX_HISTORY = 200

Point = tuple[float, float]
Polygon = list[Point]
# (frame to show on undo/redo, polygons of every frame the edit touched, as they were before it)
Snapshot = tuple[int, dict[int, list[Polygon]]]


class AnnotationError(Exception):
    """Raised when an existing annotation sidecar can't be used for the current clip."""


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".annotations.json")


def rasterize(polygons: list[Polygon], height: int, width: int) -> np.ndarray:
    """Union of `polygons` as a (height, width) uint8 mask with values 0/255.

    A pixel is inside a polygon when its centre is (even-odd rule). Used to turn a polygon into the mask the
    active contour starts from, and the conversion to apply to a frame's polygons from the JSON when masks
    are needed.
    """
    crossings = np.zeros((height, width + 1), np.int16)  # per row: +1 where a span of inside pixels starts, -1 where it ends
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float64)
        x0, y0 = points.T
        x1, y1 = np.roll(points, -1, axis=0).T
        # Rows r whose centre line y = r + 0.5 crosses an edge, half-open so a shared vertex counts once.
        first = np.ceil(np.minimum(y0, y1) - 0.5).astype(int)
        last = np.ceil(np.maximum(y0, y1) - 0.5).astype(int)  # exclusive
        edge = np.repeat(np.arange(len(points)), np.maximum(last - first, 0))
        if len(edge) == 0:
            continue
        starts = np.cumsum(np.concatenate([[0], np.maximum(last - first, 0)[:-1]]))
        rows = first[edge] + (np.arange(len(edge)) - starts[edge])
        keep = (rows >= 0) & (rows < height)
        edge, rows = edge[keep], rows[keep]
        xs = x0[edge] + (rows + 0.5 - y0[edge]) * (x1[edge] - x0[edge]) / (y1[edge] - y0[edge])
        order = np.lexsort((xs, rows))  # by row, then left to right: crossings alternate enter / leave
        rows, xs = rows[order], xs[order]
        group_start = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
        rank = np.arange(len(rows)) - np.repeat(group_start, np.diff(np.r_[group_start, len(rows)]))
        enter = np.flatnonzero(rank % 2 == 0)
        leave = enter + 1  # a closed polygon crosses each centre line an even number of times
        c0 = np.clip(np.ceil(xs[enter] - 0.5).astype(int), 0, width)
        c1 = np.clip(np.ceil(xs[leave] - 0.5).astype(int), 0, width)
        np.add.at(crossings, (rows[enter], c0), 1)
        np.add.at(crossings, (rows[enter], c1), -1)
    return np.where(np.cumsum(crossings, axis=1)[:, :width] > 0, 255, 0).astype(np.uint8)


@dataclass
class AnnotationSet:
    path: Path
    n_frames: int
    height: int
    width: int
    polygons: dict[int, list[Polygon]] = field(default_factory=dict)
    # Undo/redo stacks. In-memory only: never written to disk.
    _history: list[Snapshot] = field(default_factory=list, repr=False)
    _redo: list[Snapshot] = field(default_factory=list, repr=False)
    _stamp: tuple[int, int] | None = field(default=None, repr=False)  # sidecar (mtime, size) we last read/wrote

    def _disk_stamp(self) -> tuple[int, int] | None:
        try:
            stat = sidecar_path(self.path).stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def matches_disk(self) -> bool:
        """False if the sidecar was changed or removed behind our back (so in-memory state is stale)."""
        return self._disk_stamp() == self._stamp

    def on_frame(self, frame: int) -> list[Polygon]:
        return self.polygons.get(frame, [])

    def annotated_frames(self) -> list[int]:
        return sorted(self.polygons)

    # ---- undo ----------------------------------------------------------------------------

    def _snapshot(self, focus: int, frames: Iterable[int]) -> Snapshot:
        return focus, {f: [list(p) for p in self.on_frame(f)] for f in frames}

    def checkpoint(self, frame: int, also: Iterable[int] = ()) -> None:
        """Call just before editing `frame` (and any `also` frames changed by the same edit).

        The whole edit is one undo step. Discards any redo history.
        """
        self._history.append(self._snapshot(frame, {frame, *also}))
        del self._history[:-MAX_HISTORY]
        self._redo.clear()

    def undo(self) -> int | None:
        """Revert the latest edit and return the frame to show, or None if nothing to undo."""
        return self._travel(self._history, self._redo)

    def redo(self) -> int | None:
        """Re-apply the latest undone edit and return the frame to show, or None if nothing to redo."""
        return self._travel(self._redo, self._history)

    def _travel(self, source: list[Snapshot], other: list[Snapshot]) -> int | None:
        if not source:
            return None
        focus, snapshot = source.pop()
        other.append(self._snapshot(focus, snapshot))  # so the opposite move can undo this one
        for frame, polygons in snapshot.items():
            if polygons:
                self.polygons[frame] = polygons
            else:
                self.polygons.pop(frame, None)
        return focus

    # ---- edits (each is one undo step) ---------------------------------------------------

    def add(self, frame: int, polygon: Polygon) -> None:
        self.checkpoint(frame)
        self.polygons.setdefault(frame, []).append(polygon)

    def clear(self, frame: int) -> bool:
        if frame not in self.polygons:
            return False
        self.checkpoint(frame)
        del self.polygons[frame]
        return True

    def delete_polygon(self, frame: int, polygon_index: int) -> None:
        self.checkpoint(frame)
        polys = self.polygons[frame]
        del polys[polygon_index]
        if not polys:
            del self.polygons[frame]

    def insert_vertex(self, frame: int, polygon_index: int, index: int, point: Point) -> None:
        self.checkpoint(frame)
        self.polygons[frame][polygon_index].insert(index, point)

    def delete_vertex(self, frame: int, polygon_index: int, vertex_index: int) -> bool:
        """Remove a vertex; refused (False) if the polygon would drop below 3 vertices."""
        polygon = self.polygons[frame][polygon_index]
        if len(polygon) <= MIN_VERTICES:
            return False
        self.checkpoint(frame)
        del polygon[vertex_index]
        return True

    def replace_polygon(self, frame: int, polygon_index: int, polygon: Polygon) -> None:
        self.checkpoint(frame)
        self.polygons[frame][polygon_index] = polygon

    def set_frames(self, source_frame: int, polygons: dict[int, Polygon]) -> None:
        """Make each given frame hold exactly its one polygon (replacing what was there), as one undo step.

        `source_frame` is the frame undo/redo will bring back into view.
        """
        self.checkpoint(source_frame, also=polygons)
        for frame, polygon in polygons.items():
            self.polygons[frame] = [polygon]

    def move_vertex(self, frame: int, polygon_index: int, vertex_index: int, point: Point) -> None:
        """Move a vertex without a checkpoint: callers checkpoint once, then call this while dragging."""
        self.polygons[frame][polygon_index][vertex_index] = point

    def save(self) -> None:
        """Write all polygons to the sidecar JSON."""
        json_path = sidecar_path(self.path)
        payload = {
            "version": FORMAT_VERSION,
            "source": self.path.name,
            "label": LABEL,
            "image_size": {"width": self.width, "height": self.height},
            "n_frames": self.n_frames,
            "frames": {
                str(f): [[[x, y] for x, y in poly] for poly in polys]
                for f, polys in sorted(self.polygons.items())
            },
        }
        tmp = json_path.with_name(json_path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, json_path)  # atomic, so a crash can't leave a half-written file
        self._stamp = self._disk_stamp()

    @classmethod
    def load(cls, path: Path, n_frames: int, height: int, width: int) -> AnnotationSet:
        """Read the sidecar if present (else an empty set); AnnotationError if it doesn't match."""
        result = cls(path, n_frames, height, width)
        json_path = sidecar_path(path)
        result._stamp = result._disk_stamp()
        if not json_path.exists():
            return result
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            size = data["image_size"]
            if (data["n_frames"], size["height"], size["width"]) != (n_frames, height, width):
                raise AnnotationError(
                    f"{json_path.name} was made for a {data['n_frames']}-frame "
                    f"{size['width']}x{size['height']} clip"
                )
            for key, polys in data["frames"].items():
                frame = int(key)
                if not 0 <= frame < n_frames:
                    raise AnnotationError(f"{json_path.name} has out-of-range frame {frame}")
                result.polygons[frame] = [[(float(x), float(y)) for x, y in poly] for poly in polys]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise AnnotationError(f"{json_path.name} is unreadable: {exc}") from exc
        return result
