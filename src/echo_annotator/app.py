"""Main window: drag-and-drop / Ctrl+O loading, playback, folder navigation, polygon annotation."""

from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import wx

from .annotations import AnnotationError, AnnotationSet, Polygon
from .clip_io import Clip, ClipLoadError, load_clip, sibling_clips
from .geometry import nearest_edge, nearest_vertex, polygon_area, polygon_at
from .grow import grow_polygon
from .tracking import propagate_polygon
from .viewer import MOD_CTRL, MOD_SHIFT, ViewerWidget

TITLE = "Echo Annotator"
HIT_RADIUS_PX = 8  # on-screen distance for grabbing vertices/edges and closing a polygon
ZOOM_KEY_STEP = 1.5
PRESERVE_CURVATURE_KEY = "preserve_curvature"
APP_NAME = "EchoAnnotator"

CONTROLS = """\
Files
  Ctrl+O / drag and drop    open an AVI
  Up / Down                 next / previous AVI in the folder
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
    with wx.BusyCursor():
        yield


class _FileDrop(wx.FileDropTarget):
    """Opens the first file dropped on the window it is attached to."""

    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self._window = window

    def OnDropFiles(self, x: int, y: int, filenames: list[str]) -> bool:
        if not filenames:
            return False
        self._window.open_path(Path(filenames[0]))
        return True


@dataclass
class Drag:
    frame: int
    polygon: int
    vertex: int
    # Checkpointed once the polygon actually changes: at the first move, or already for an inserted vertex.
    # A plain click on a vertex therefore never touches undo/redo history.
    checkpointed: bool = False


class MainWindow(wx.Frame):
    def __init__(self, config: wx.ConfigBase | None = None) -> None:
        super().__init__(None, title=TITLE)
        self.SetClientSize((960, 720))

        self.viewer = ViewerWidget(self)
        self.viewer.pressed.connect(self._on_pressed)
        self.viewer.dragged.connect(self._on_dragged)
        self.viewer.released.connect(self._end_drag)
        self.viewer.right_pressed.connect(self._on_right_pressed)
        self.viewer.hovered.connect(self._on_hovered)
        self.viewer.finish_requested.connect(self.finish_polygon)
        self.viewer.view_changed.connect(self._update_info)
        # Drops go to the window under the pointer, so both need a target.
        self.SetDropTarget(_FileDrop(self))
        self.viewer.SetDropTarget(_FileDrop(self))

        self.CreateStatusBar()
        self._info_text = "Drop an AVI here or press Ctrl+O"
        self._message = ""  # temporary status message; the info text shows again once it expires
        self._message_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_message_expired, source=self._message_timer)
        self.SetStatusText(self._info_text)

        # Injectable so tests never touch the real user profile (the Windows registry).
        self._settings = config if config is not None else wx.Config(APP_NAME, APP_NAME)
        # Ctrl+O / Ctrl+R / Ctrl+T are menu accelerators (Cmd on macOS); the other keys go through _on_char_hook.
        menubar = wx.MenuBar()
        file_menu, tools_menu, help_menu = wx.Menu(), wx.Menu(), wx.Menu()
        self._add_item(file_menu, "&Open echo...\tCtrl+O", self.open_dialog)
        self._add_item(tools_menu, "&Grow polygon\tCtrl+R", self.grow_current)
        self._add_item(tools_menu, "&Propagate polygon\tCtrl+T", self.propagate_current)
        self._curvature_item = tools_menu.AppendCheckItem(wx.ID_ANY, "Preserve contour &curvature when propagating")
        self._curvature_item.Check(self._settings.ReadBool(PRESERVE_CURVATURE_KEY, False))
        self.Bind(
            wx.EVT_MENU,
            lambda event: self.set_preserve_curvature(self._curvature_item.IsChecked()),
            self._curvature_item,
        )
        self._add_item(help_menu, "&Controls...", self.show_controls)
        menubar.Append(file_menu, "&File")
        menubar.Append(tools_menu, "&Tools")
        menubar.Append(help_menu, "&Help")
        self.SetMenuBar(menubar)

        self._timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, lambda event: self._on_tick(), source=self._timer)
        self.Bind(wx.EVT_CHAR_HOOK, self._on_char_hook)
        self.Bind(wx.EVT_CLOSE, self._on_close)

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

    def _add_item(self, menu: wx.Menu, label: str, handler) -> None:
        item = menu.Append(wx.ID_ANY, label)
        self.Bind(wx.EVT_MENU, lambda event: handler(), item)

    def _on_close(self, event: wx.CloseEvent) -> None:
        self._timer.Stop()
        self._message_timer.Stop()
        event.Skip()  # carry on with the default close

    # ---- status bar ----------------------------------------------------------------------

    def _say(self, text: str, timeout_ms: int) -> None:
        """Show `text` in the status bar for `timeout_ms`, then go back to the info line."""
        self._message = text
        self.SetStatusText(text)
        self._message_timer.StartOnce(timeout_ms)

    def _on_message_expired(self, event: wx.TimerEvent) -> None:
        self._message = ""
        self.SetStatusText(self._info_text)

    def current_message(self) -> str:
        return self._message

    @property
    def preserve_curvature(self) -> bool:
        return self._curvature_item.IsChecked()

    def set_preserve_curvature(self, on: bool) -> None:
        self._curvature_item.Check(on)
        self._settings.WriteBool(PRESERVE_CURVATURE_KEY, on)
        self._settings.Flush()

    def show_controls(self) -> None:
        with wx.Dialog(self, title="Controls", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER) as dialog:
            text = wx.TextCtrl(dialog, value=CONTROLS, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP)
            text.SetFont(wx.Font(wx.FontInfo(10).Family(wx.FONTFAMILY_TELETYPE)))
            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(text, 1, wx.EXPAND | wx.ALL, 8)
            sizer.Add(dialog.CreateStdDialogButtonSizer(wx.OK), 0, wx.EXPAND | wx.ALL, 8)
            dialog.SetSizer(sizer)
            dialog.SetEscapeId(wx.ID_OK)
            dialog.SetClientSize(text.GetTextExtent("x" * 100).Width + 40, 560)
            dialog.ShowModal()

    # ---- loading -------------------------------------------------------------------------

    def open_dialog(self) -> None:
        start = str(self.clip.path.parent) if self.clip else ""
        with wx.FileDialog(
            self,
            "Open echo",
            defaultDir=start,
            wildcard="Echo videos (*.avi)|*.avi|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dialog:
            if dialog.ShowModal() == wx.ID_OK:
                self.open_path(Path(dialog.GetPath()))

    def open_path(self, path: Path) -> bool:
        """Load `path` and start playing it. Leaves the current clip untouched on failure."""
        try:
            clip = load_clip(path)
        except ClipLoadError as exc:
            self._say(f"Could not open: {exc}", 8000)
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
        self.SetTitle(f"{clip.path.name} - {TITLE}")
        self._render()
        self.set_playing(True)

    def _annotations_for(self, clip: Clip) -> AnnotationSet | None:
        """The clip's annotation set, reusing the one from earlier in this session if it matches.

        Reusing the live object (rather than re-reading the sidecar) is what keeps each file's
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
            self._say(f"Annotation disabled for this file: {exc}", 10000)
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
        self._say(
            "Last file in folder" if step > 0 else "First file in folder", 2000
        )

    # ---- playback ------------------------------------------------------------------------

    def set_playing(self, playing: bool) -> None:
        if self.clip is None:
            return
        if playing:
            self._end_drag()
            self._discard_pending()
            self._timer.Start(round(1000 / self.clip.fps))
        else:
            self._timer.Stop()
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
        self._info_text = text
        if not self._message:
            self.SetStatusText(text)

    # ---- annotation ----------------------------------------------------------------------

    def _refresh_overlay(self) -> None:
        polygons = self.annotations.on_frame(self.frame) if self.annotations else []
        self.viewer.set_overlay(polygons, self.pending, self.selected)

    def _can_edit(self) -> bool:
        if self.clip is None:
            return False
        if self.annotations is None:
            self._say("Annotation is disabled for this file", 3000)
            return False
        if self.playing:
            self._say("Pause (Space) to annotate", 2000)
            return False
        return True

    def _set_cursor(self, over_vertex: bool) -> None:
        if self.playing or not self.annotations:
            shape = wx.CURSOR_ARROW
        elif over_vertex:
            shape = wx.CURSOR_SIZING
        else:
            shape = wx.CURSOR_CROSS
        self.viewer.SetCursor(wx.Cursor(shape))

    def _hit_radius(self) -> float:
        return HIT_RADIUS_PX / self.viewer.scale()

    def _after_edit(self) -> None:
        self._save()
        self._refresh_overlay()
        self._update_info()

    def _discard_pending(self) -> None:
        if self.pending:
            self.pending = []
            self._say("Unfinished polygon discarded", 2000)
            self._refresh_overlay()

    def _on_pressed(self, x: float, y: float, modifiers: int) -> None:
        if not self._can_edit():
            return
        if self.pending:
            self._add_vertex(x, y)
            return
        polygons = self.annotations.on_frame(self.frame)
        if modifiers & MOD_CTRL:
            self.selected = polygon_at(polygons, x, y)
            self._refresh_overlay()
        elif modifiers & MOD_SHIFT:
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
            self._say(
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
            self._say("A polygon needs at least 3 points", 2000)
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
            self._say(empty_message, 2000)
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
            self._say("Finish the polygon (Enter) or discard it (Esc) first", 3000)
            return None
        polygons = self.annotations.on_frame(self.frame)
        if self.selected is not None and self.selected < len(polygons):
            index = self.selected
        elif len(polygons) == 1:
            index = 0
        else:
            self._say(
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
            self._say("Nothing to grow: the polygon already meets strong edges", 3000)
            return
        self.annotations.replace_polygon(self.frame, index, grown)
        self.selected = index
        self._after_edit()
        self._say(
            f"Grew polygon from {polygon_area(polygon):.0f} to {polygon_area(grown):.0f} px²", 4000
        )

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
            self._say("Could not track the polygon to any other frame", 4000)
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
        self._say(message, 6000)

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
            self._say(f"Could not save annotations: {exc}", 10000)

    # ---- events --------------------------------------------------------------------------

    def _on_char_hook(self, event: wx.KeyEvent) -> None:
        key = event.GetKeyCode()
        cmd = event.CmdDown()
        if key == wx.WXK_SPACE:
            if not event.IsAutoRepeat():
                self.set_playing(not self.playing)
        elif key in (wx.WXK_LEFT, wx.WXK_RIGHT):
            if not self.playing:
                self.step(-1 if key == wx.WXK_LEFT else 1)
        elif key == wx.WXK_UP:
            self.open_sibling(+1)
        elif key == wx.WXK_DOWN:
            self.open_sibling(-1)
        elif key in (wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER):
            self.finish_polygon()
        elif key == wx.WXK_ESCAPE:
            self.escape()
        elif key == wx.WXK_BACK:
            if self.pending:  # only ever removes a vertex, never a finished polygon
                self.undo()
        elif cmd and key == ord("Z"):
            self.undo()
        elif cmd and key == ord("Y"):
            self.redo()
        elif cmd and key in (ord("="), ord("+"), wx.WXK_ADD, wx.WXK_NUMPAD_ADD):
            self.viewer.zoom_by(ZOOM_KEY_STEP)
        elif cmd and key in (ord("-"), wx.WXK_SUBTRACT, wx.WXK_NUMPAD_SUBTRACT):
            self.viewer.zoom_by(1 / ZOOM_KEY_STEP)
        elif cmd and key in (ord("0"), wx.WXK_NUMPAD0):
            self.viewer.reset_view()
        elif key == wx.WXK_DELETE:
            self.delete_selected_or_clear()
        else:
            event.Skip()  # includes the menu accelerators (Ctrl+O / Ctrl+R / Ctrl+T)


class _App(wx.App):
    window: MainWindow | None = None

    def MacOpenFiles(self, filenames: list[str]) -> None:  # Finder "Open With" / dropping a file on the Dock icon
        if self.window is not None and filenames:
            self.window.open_path(Path(filenames[0]))


def main() -> None:
    app = _App(False)
    app.SetAppName(APP_NAME)
    window = app.window = MainWindow()
    window.Show()
    if len(sys.argv) > 1:
        window.open_path(Path(sys.argv[1]))
    app.MainLoop()
