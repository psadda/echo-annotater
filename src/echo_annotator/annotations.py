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
from PIL import Image, ImageDraw

LABEL = "pericardial_fat"
FORMAT_VERSION = 1
MIN_VERTICES = 3
MAX_HISTORY = 200
SUPERSAMPLE = 4  # fat stripes are only a few pixels thick, so rasterize with sub-pixel accuracy

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

    Not used by the app itself; this is the conversion to apply to a frame's polygons from the JSON.
    """
    s = SUPERSAMPLE
    big = Image.new("L", (width * s, height * s), 0)
    draw = ImageDraw.Draw(big)
    for polygon in polygons:
        # PIL treats integer coordinates as pixel centres, hence the -0.5
        draw.polygon([(x * s - 0.5, y * s - 0.5) for x, y in polygon], fill=255)
    small = np.asarray(big.resize((width, height), Image.Resampling.BOX))
    return np.where(small >= 128, 255, 0).astype(np.uint8)


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
