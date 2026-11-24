"""Main window: drag-and-drop / Ctrl+O loading, playback, folder navigation, polygon annotation."""

from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QFontDatabase, QKeyEvent, QKeySequence
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox

from .annotations import AnnotationError, AnnotationSet, Polygon
from .clip_io import Clip, ClipLoadError, load_clip, sibling_clips
from .geometry import nearest_edge, nearest_vertex, polygon_area, polygon_at
from .grow import grow_polygon
from .tracking import propagate_polygon
from .viewer import ViewerWidget

TITLE = "Echo Annotator"
HIT_RADIUS_PX = 8  # on-screen distance for grabbing vertices/edges and closing a polygon
ZOOM_KEY_STEP = 1.5
PRESERVE_CURVATURE_KEY = "preserve_curvature"

CONTROLS = """\
Files
  Ctrl+O / drag and drop    open a DICOM or AVI
  Up / Down                 next / previous DICOM or AVI in the folder
Playback
  Space                     pause / resume
  Left / Right (paused)     previous / next frame
View
  Mouse wheel               zoom at the cursor
  Ctrl+= / Ctrl+-           zoom in / out
  Middle-drag or Alt+drag   pan
  Ctrl+0                    reset to fit
Drawing (paused)
  Click                     add a vertex
  Enter, double-click, or click the first vertex    close the polygon
  Backspace                 remove the last vertex
  Esc                       discard the polygon (or deselect)
Editing (paused)
  Drag a vertex             move it
  Shift+click an edge       insert a vertex and drag it
  Right-click a vertex      delete it (polygons keep at least 3)
  Ctrl+click in a polygon   select it (Ctrl+click elsewhere deselects)
  Delete                    delete the selected polygon, or clear the frame
  Ctrl+Z / Ctrl+Y           undo / redo (kept per file while the app is open)
Tools (paused; act on the selected polygon, or the frame's only polygon)
  Ctrl+R                    grow it on this frame with an active contour (press again to grow more)
  Ctrl+T                    track it to all other frames with optical flow, forward and back;
                            replaces any polygons already on those frames. The tracked polygons have
                            evenly spaced vertices (16-32); Tools > Preserve contour curvature keeps the
                            source shape's local curvature (stronger de-noising, less shape freedom)
"""


@contextmanager
def _busy():
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


@dataclass
class Drag:
    frame: int
    polygon: int
    vertex: int
    # Checkpointed once the polygon actually changes: at the first move, or already for an inserted vertex.
    # A plain click on a vertex therefore never touches undo/redo history.
    checkpointed: bool = False


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(TITLE)
        self.resize(960, 720)
        self.setAcceptDrops(True)

        self.viewer = ViewerWidget()
        self.viewer.pressed.connect(self._on_pressed)
        self.viewer.dragged.connect(self._on_dragged)
        self.viewer.released.connect(self._end_drag)
        self.viewer.right_pressed.connect(self._on_right_pressed)
        self.viewer.hovered.connect(self._on_hovered)
        self.viewer.finish_requested.connect(self.finish_polygon)
        self.viewer.view_changed.connect(self._update_info)
        self.setCentralWidget(self.viewer)

        self._info = QLabel("Drop a DICOM or AVI here or press Ctrl+O")
        self.statusBar().addPermanentWidget(self._info, 1)

        open_action = QAction("&Open echo...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.open_dialog)
        self.menuBar().addMenu("&File").addAction(open_action)
        # The Ctrl+R / Ctrl+T shortcuts are handled in keyPressEvent; the tab just displays them in the menu.
        tools_menu = self.menuBar().addMenu("&Tools")
        grow_action = QAction("&Grow polygon\tCtrl+R", self)
        grow_action.triggered.connect(self.grow_current)
        tools_menu.addAction(grow_action)
        propagate_action = QAction("&Propagate polygon\tCtrl+T", self)
        propagate_action.triggered.connect(self.propagate_current)
        tools_menu.addAction(propagate_action)
        self._settings = QSettings("EchoAnnotator", "EchoAnnotator")
        self._curvature_action = QAction("Preserve contour &curvature when propagating", self, checkable=True)
        self._curvature_action.setChecked(self._settings.value(PRESERVE_CURVATURE_KEY, False, type=bool))
        self._curvature_action.toggled.connect(lambda on: self._settings.setValue(PRESERVE_CURVATURE_KEY, on))
        tools_menu.addAction(self._curvature_action)
        help_action = QAction("&Controls...", self)
        help_action.triggered.connect(self.show_controls)
        self.menuBar().addMenu("&Help").addAction(help_action)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        self.clip: Clip | None = None
        self.frame = 0
        self.playing = False
        self.annotations: AnnotationSet | None = None
        self._annotation_cache: dict[Path, AnnotationSet] = {}  # per file, in memory for this session
        self.pending: Polygon = []  # polygon being drawn on the current frame
        self.selected: int | None = None  # index into the current frame's polygons
        self._drag: Drag | None = None
        self._siblings: list[Path] = []
        self._index = -1

    def show_controls(self) -> None:
        box = QMessageBox(QMessageBox.Icon.Information, "Controls", CONTROLS, parent=self)
        box.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        box.exec()

    # ---- loading -------------------------------------------------------------------------

    def open_dialog(self) -> None:
        start = str(self.clip.path.parent) if self.clip else ""
        name, _ = QFileDialog.getOpenFileName(
            self, "Open echo", start, "All files (*);;Echo files (*.dcm *.dicom *.avi)"
        )
        if name:
            self.open_path(Path(name))

    def open_path(self, path: Path) -> bool:
        """Load `path` and start playing it. Leaves the current clip untouched on failure."""
        try:
            clip = load_clip(path)
        except ClipLoadError as exc:
            self.statusBar().showMessage(f"Could not open: {exc}", 8000)
            return False
        self._siblings = sibling_clips(path)
        self._index = self._siblings.index(path)
        self._show_clip(clip)
        return True

    def _show_clip(self, clip: Clip) -> None:
        self._end_drag()  # finish any drag against the old annotation set first
        self.pending = []
        self.selected = None
        self.clip = clip
        self.frame = 0
        self.viewer.reset_view()
        self.annotations = self._annotations_for(clip)
        self.setWindowTitle(f"{clip.path.name} - {TITLE}")
        self._render()
        self.set_playing(True)

    def _annotations_for(self, clip: Clip) -> AnnotationSet | None:
        """The clip's annotation set, reusing the one from earlier in this session if it matches.

        Reusing the live object (rather than re-reading the sidecar) is what keeps each DICOM's
        undo/redo history when you switch away and back. It is never written to disk. If the
        sidecar changed on disk in the meantime, the set is reloaded and its history starts over.
        """
        _, height, width = clip.frames.shape[:3]
        key = clip.path.resolve()
        cached = self._annotation_cache.get(key)
        if (
            cached
            and (cached.n_frames, cached.height, cached.width) == (clip.n_frames, height, width)
            and cached.matches_disk()  # an externally edited sidecar invalidates the history too
        ):
            return cached
        try:
            annotations = AnnotationSet.load(clip.path, clip.n_frames, height, width)
        except AnnotationError as exc:
            self._annotation_cache.pop(key, None)
            self.statusBar().showMessage(f"Annotation disabled for this file: {exc}", 10000)
            return None
        self._annotation_cache[key] = annotations
        return annotations

    def open_sibling(self, step: int) -> None:
        """Move `step` (+1 next / -1 previous) files through the folder, skipping unreadable ones."""
        if not self._siblings:
            return
        i = self._index + step
        while 0 <= i < len(self._siblings):
            try:
                clip = load_clip(self._siblings[i])
            except ClipLoadError:
                i += step
                continue
            self._index = i
            self._show_clip(clip)
            return
        self.statusBar().showMessage(
            "Last file in folder" if step > 0 else "First file in folder", 2000
        )

    # ---- playback ------------------------------------------------------------------------

    def set_playing(self, playing: bool) -> None:
        if self.clip is None:
            return
        if playing:
            self._end_drag()
            self._discard_pending()
            self._timer.start(round(1000 / self.clip.fps))
        else:
            self._timer.stop()
        self.playing = playing
        self._set_cursor(False)
        self._update_info()

    def step(self, delta: int) -> None:
        if self.clip is None:
            return
        self._end_drag()
        self._discard_pending()
        self.selected = None
        self.frame = (self.frame + delta) % self.clip.n_frames
        self._render()

    def _on_tick(self) -> None:
        self.step(1)

    def _render(self) -> None:
        if self.clip is not None:
            self.viewer.set_frame(self.clip.frames[self.frame])
            self._refresh_overlay()
            self._update_info()

    def _update_info(self) -> None:
        if self.clip is None:
            return
        state = "Playing" if self.playing else "Paused - click to draw"
        text = (
            f"{self.clip.path.name}   file {self._index + 1}/{len(self._siblings)}   "
            f"frame {self.frame + 1}/{self.clip.n_frames}   {self.clip.fps:.0f} fps   {state}"
        )
        if self.viewer.zoom > 1:
            text += f"   zoom {self.viewer.zoom * 100:.0f}%"
        if self.annotations is not None:
            text += (
                f"   polygons {len(self.annotations.on_frame(self.frame))}"
                f"   annotated frames {len(self.annotations.annotated_frames())}"
            )
        self._info.setText(text)

    # ---- annotation ----------------------------------------------------------------------

    def _refresh_overlay(self) -> None:
        polygons = self.annotations.on_frame(self.frame) if self.annotations else []
        self.viewer.set_overlay(polygons, self.pending, self.selected)

    def _can_edit(self) -> bool:
        if self.clip is None:
            return False
        if self.annotations is None:
            self.statusBar().showMessage("Annotation is disabled for this file", 3000)
            return False
        if self.playing:
            self.statusBar().showMessage("Pause (Space) to annotate", 2000)
            return False
        return True

    def _set_cursor(self, over_vertex: bool) -> None:
        if self.playing or not self.annotations:
            shape = Qt.CursorShape.ArrowCursor
        elif over_vertex:
            shape = Qt.CursorShape.SizeAllCursor
        else:
            shape = Qt.CursorShape.CrossCursor
        self.viewer.setCursor(shape)

    def _hit_radius(self) -> float:
        return HIT_RADIUS_PX / self.viewer.scale()

    def _after_edit(self) -> None:
        self._save()
        self._refresh_overlay()
        self._update_info()

    def _discard_pending(self) -> None:
        if self.pending:
            self.pending = []
            self.statusBar().showMessage("Unfinished polygon discarded", 2000)
            self._refresh_overlay()

    def _on_pressed(self, x: float, y: float, modifiers) -> None:
        if not self._can_edit():
            return
        if self.pending:
            self._add_vertex(x, y)
            return
        polygons = self.annotations.on_frame(self.frame)
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self.selected = polygon_at(polygons, x, y)
            self._refresh_overlay()
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            hit = nearest_edge(polygons, x, y, self._hit_radius())
            if hit:
                polygon, edge, point = hit
                self.annotations.insert_vertex(self.frame, polygon, edge + 1, point)
                self.selected = polygon
                self._drag = Drag(self.frame, polygon, edge + 1, checkpointed=True)
                self._refresh_overlay()
        elif hit := nearest_vertex(polygons, x, y, self._hit_radius()):
            polygon, vertex = hit
            self.selected = polygon
            self._drag = Drag(self.frame, polygon, vertex)
            self._refresh_overlay()
        else:
            self.selected = None
            self._add_vertex(x, y)

    def _add_vertex(self, x: float, y: float) -> None:
        if len(self.pending) >= 3:
            fx, fy = self.pending[0]
            if math.hypot(x - fx, y - fy) * self.viewer.scale() <= HIT_RADIUS_PX:
                self.finish_polygon()
                return
        self.pending.append((x, y))
        self._refresh_overlay()

    def _on_dragged(self, x: float, y: float) -> None:
        if self._drag is None:
            return
        d = self._drag
        if not d.checkpointed:
            self.annotations.checkpoint(d.frame)
            d.checkpointed = True
        self.annotations.move_vertex(d.frame, d.polygon, d.vertex, (x, y))
        self._refresh_overlay()

    def _end_drag(self) -> None:
        d, self._drag = self._drag, None
        if d is not None and d.checkpointed:
            self._after_edit()

    def _on_right_pressed(self, x: float, y: float) -> None:
        if self.pending or not self._can_edit():
            return
        hit = nearest_vertex(self.annotations.on_frame(self.frame), x, y, self._hit_radius())
        if hit is None:
            return
        polygon, vertex = hit
        if self.annotations.delete_vertex(self.frame, polygon, vertex):
            self.selected = polygon
            self._after_edit()
        else:
            self.statusBar().showMessage(
                "A polygon needs at least 3 vertices - Ctrl+click it and press Delete to remove it", 4000
            )

    def _on_hovered(self, x: float, y: float) -> None:
        over_vertex = bool(
            not self.pending
            and self.annotations
            and nearest_vertex(self.annotations.on_frame(self.frame), x, y, self._hit_radius())
        )
        self._set_cursor(over_vertex)

    def finish_polygon(self) -> None:
        if not self.pending or not self._can_edit():
            return
        if len(self.pending) < 3:
            self.statusBar().showMessage("A polygon needs at least 3 points", 2000)
            return
        self.annotations.add(self.frame, self.pending)
        self.pending = []
        self.selected = None
        self._after_edit()

    def undo(self) -> None:
        """Remove the last vertex of the polygon being drawn, else undo the last edit."""
        if not self._can_edit():
            return
        if self.pending:
            self.pending.pop()
            self._refresh_overlay()
            return
        self._travel_history(self.annotations.undo, "Nothing to undo")

    def redo(self) -> None:
        if not self._can_edit() or self.pending:  # a half-drawn polygon has no redo
            return
        self._travel_history(self.annotations.redo, "Nothing to redo")

    def _travel_history(self, move, empty_message: str) -> None:
        frame = move()
        if frame is None:
            self.statusBar().showMessage(empty_message, 2000)
            return
        self.frame = frame  # show the frame that changed
        self.selected = None
        self._render()
        self._save()

    def _tool_target(self) -> tuple[int, Polygon] | None:
        """(index, polygon) the Ctrl+R / Ctrl+T tools act on: the selected one, else the frame's only one."""
        if not self._can_edit():
            return None
        if self.pending:
            self.statusBar().showMessage("Finish the polygon (Enter) or discard it (Esc) first", 3000)
            return None
        polygons = self.annotations.on_frame(self.frame)
        if self.selected is not None and self.selected < len(polygons):
            index = self.selected
        elif len(polygons) == 1:
            index = 0
        else:
            self.statusBar().showMessage(
                "Select a polygon first (Ctrl+click)" if polygons else "No polygon on this frame", 3000
            )
            return None
        return index, polygons[index]

    def grow_current(self) -> None:
        """Ctrl+R: grow the current polygon on this frame with an active contour."""
        target = self._tool_target()
        if target is None:
            return
        index, polygon = target
        with _busy():
            grown = grow_polygon(self.clip.gray(self.frame), polygon)
        if grown is None:
            self.statusBar().showMessage("Nothing to grow: the polygon already meets strong edges", 3000)
            return
        self.annotations.replace_polygon(self.frame, index, grown)
        self.selected = index
        self._after_edit()
        self.statusBar().showMessage(
            f"Grew polygon from {polygon_area(polygon):.0f} to {polygon_area(grown):.0f} px²", 4000
        )

    @property
    def preserve_curvature(self) -> bool:
        return self._curvature_action.isChecked()

    def propagate_current(self) -> None:
        """Ctrl+T: track the current polygon to every other frame with sparse optical flow."""
        target = self._tool_target()
        if target is None:
            return
        index, polygon = target
        with _busy():
            tracked = propagate_polygon(
                self.clip.gray, self.clip.n_frames, self.frame, polygon, self.preserve_curvature
            )
        if not tracked:
            self.statusBar().showMessage("Could not track the polygon to any other frame", 4000)
            return
        replaced = sum(1 for frame in tracked if self.annotations.on_frame(frame))
        self.annotations.set_frames(self.frame, tracked)
        self.selected = index
        self._after_edit()
        forward = sum(1 for frame in tracked if frame > self.frame)
        message = f"Propagated to {forward} frames forward and {len(tracked) - forward} back"
        if self.preserve_curvature:
            message += " (curvature preserved)"
        if replaced:
            message += f" (replaced annotations on {replaced})"
        if len(tracked) < self.clip.n_frames - 1:
            message += " - tracking was lost before the clip ended"
        self.statusBar().showMessage(message, 6000)

    def delete_selected_or_clear(self) -> None:
        if not self._can_edit():
            return
        self.pending = []
        if self.selected is not None:
            self.annotations.delete_polygon(self.frame, self.selected)
            self.selected = None
        else:
            self.annotations.clear(self.frame)
        self._after_edit()

    def escape(self) -> None:
        if self.pending:
            self._discard_pending()
        elif self.selected is not None:
            self.selected = None
            self._refresh_overlay()

    def _save(self) -> None:
        try:
            self.annotations.save()
        except OSError as exc:
            self.statusBar().showMessage(f"Could not save annotations: {exc}", 10000)

    # ---- events --------------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if key == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self.set_playing(not self.playing)
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            if not self.playing:
                self.step(-1 if key == Qt.Key.Key_Left else 1)
        elif key == Qt.Key.Key_Up:
            self.open_sibling(+1)
        elif key == Qt.Key.Key_Down:
            self.open_sibling(-1)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.finish_polygon()
        elif key == Qt.Key.Key_Escape:
            self.escape()
        elif key == Qt.Key.Key_Backspace:
            if self.pending:  # only ever removes a vertex, never a finished polygon
                self.undo()
        elif ctrl and key == Qt.Key.Key_Z:
            self.undo()
        elif ctrl and key == Qt.Key.Key_Y:
            self.redo()
        elif ctrl and key == Qt.Key.Key_R:
            self.grow_current()
        elif ctrl and key == Qt.Key.Key_T:
            self.propagate_current()
        elif ctrl and key in (Qt.Key.Key_Equal, Qt.Key.Key_Plus):
            self.viewer.zoom_by(ZOOM_KEY_STEP)
        elif ctrl and key == Qt.Key.Key_Minus:
            self.viewer.zoom_by(1 / ZOOM_KEY_STEP)
        elif ctrl and key == Qt.Key.Key_0:
            self.viewer.reset_view()
        elif key == Qt.Key.Key_Delete:
            self.delete_selected_or_clear()
        else:
            super().keyPressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.open_path(Path(url.toLocalFile()))
                event.acceptProposedAction()
                return


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if len(sys.argv) > 1:
        window.open_path(Path(sys.argv[1]))
    sys.exit(app.exec())
