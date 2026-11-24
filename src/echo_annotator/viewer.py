"""Widget that displays a single frame scaled to fit, with a polygon overlay and mouse input."""

from __future__ import annotations

import math
from typing import NamedTuple

import cv2
import numpy as np
import wx

Polygon = list[tuple[float, float]]

MAX_ZOOM = 20.0
ZOOM_STEP = 1.25  # per wheel notch
SMOOTH_BELOW_SCALE = 2.0  # widget pixels per image pixel

# Modifier bit flags passed with the `pressed` signal.
MOD_CTRL = 1  # Ctrl on Windows/Linux, Cmd on macOS
MOD_SHIFT = 2
MOD_ALT = 4

FILL = wx.Colour(255, 200, 0, 70)
OUTLINE = wx.Colour(255, 200, 0)
SELECTED = wx.Colour(255, 255, 255)
PENDING = wx.Colour(0, 220, 255)


class Rect(NamedTuple):
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def empty(self) -> bool:
        return self.w <= 0 or self.h <= 0

    def contains(self, point: tuple[float, float]) -> bool:
        return not self.empty and self.x <= point[0] <= self.right and self.y <= point[1] <= self.bottom


class Signal:
    """Minimal stand-in for a signal: `connect` callbacks, `emit` calls them in order."""

    def __init__(self) -> None:
        self._slots: list = []

    def connect(self, slot) -> None:
        self._slots.append(slot)

    def emit(self, *args) -> None:
        for slot in list(self._slots):
            slot(*args)


def _modifiers(event: wx.MouseEvent) -> int:
    return (
        (MOD_CTRL if event.CmdDown() else 0)
        | (MOD_SHIFT if event.ShiftDown() else 0)
        | (MOD_ALT if event.AltDown() else 0)
    )


class ViewerWidget(wx.Window):
    # All coordinates are image coordinates.
    #   pressed(x, y, modifiers)  left press inside the image; modifiers is a MOD_* bit mask
    #   dragged(x, y)             left button held; clamped to the image
    #   released()
    #   right_pressed(x, y)
    #   hovered(x, y)             no button held
    #   finish_requested()        double-click
    #   view_changed()            zoom or pan changed

    def __init__(self, parent: wx.Window) -> None:
        super().__init__(parent)
        self.SetMinSize((320, 240))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.pressed = Signal()
        self.dragged = Signal()
        self.released = Signal()
        self.right_pressed = Signal()
        self.hovered = Signal()
        self.finish_requested = Signal()
        self.view_changed = Signal()

        self._image: np.ndarray | None = None  # H x W x 3 uint8, RGB
        self._polygons: list[Polygon] = []
        self._pending: Polygon = []
        self._selected: int | None = None
        self._cursor: tuple[float, float] | None = None
        self._zoom = 1.0  # 1 = fit to window
        self._pan = (0.0, 0.0)  # offset in widget pixels from the centred position
        self._pan_from: tuple[float, float] | None = None

        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda event: None)
        self.Bind(wx.EVT_SIZE, lambda event: (self.Refresh(), event.Skip()))
        self.Bind(wx.EVT_MOUSEWHEEL, self._on_wheel)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_left_down)
        self.Bind(wx.EVT_LEFT_DCLICK, self._on_left_dclick)
        self.Bind(wx.EVT_LEFT_UP, self._on_left_up)
        self.Bind(wx.EVT_RIGHT_DOWN, self._on_right_down)
        self.Bind(wx.EVT_MIDDLE_DOWN, self._on_middle_down)
        self.Bind(wx.EVT_MIDDLE_UP, self._on_middle_up)
        self.Bind(wx.EVT_MOTION, self._on_motion)
        self.Bind(wx.EVT_LEAVE_WINDOW, self._on_leave)
        self.Bind(wx.EVT_MOUSE_CAPTURE_LOST, self._on_capture_lost)

    def AcceptsFocus(self) -> bool:  # keys are handled by the main window
        return False

    def set_frame(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        self._image = np.ascontiguousarray(frame)
        self.Refresh()

    def set_overlay(self, polygons: list[Polygon], pending: Polygon, selected: int | None) -> None:
        self._polygons = polygons
        self._pending = pending
        self._selected = selected
        self.Refresh()

    # ---- geometry ------------------------------------------------------------------------

    @property
    def zoom(self) -> float:
        return self._zoom

    def _size(self) -> tuple[int, int]:
        return tuple(self.GetClientSize())

    def _image_size(self) -> tuple[int, int]:
        height, width = self._image.shape[:2]
        return width, height

    def _view_size(self) -> tuple[float, float]:
        """Size in widget pixels of the whole image at the current zoom."""
        (iw, ih), (ww, wh) = self._image_size(), self._size()
        fit = min(ww / iw, wh / ih)
        return iw * fit * self._zoom, ih * fit * self._zoom

    def _clamp_pan(self, pan: tuple[float, float], w: float, h: float) -> tuple[float, float]:
        """Keep a zoomed image covering the widget; an image smaller than the widget stays centred."""
        ww, wh = self._size()
        limit_x, limit_y = max(0.0, (w - ww) / 2), max(0.0, (h - wh) / 2)
        return min(max(pan[0], -limit_x), limit_x), min(max(pan[1], -limit_y), limit_y)

    def image_rect(self) -> Rect:
        """Widget-space rectangle the image is drawn into (empty if nothing is loaded)."""
        if self._image is None:
            return Rect(0, 0, 0, 0)
        w, h = self._view_size()
        ww, wh = self._size()
        px, py = self._clamp_pan(self._pan, w, h)
        return Rect((ww - w) / 2 + px, (wh - h) / 2 + py, w, h)

    def zoom_by(self, factor: float, anchor: tuple[float, float] | None = None) -> None:
        """Zoom by `factor`, keeping the image point under `anchor` (default: widget centre) fixed."""
        if self._image is None:
            return
        ww, wh = self._size()
        anchor = anchor if anchor is not None else (ww / 2, wh / 2)
        ix, iy = self.to_image(anchor)
        self._zoom = min(max(self._zoom * factor, 1.0), MAX_ZOOM)
        w, h = self._view_size()
        s = w / self._image_size()[0]
        pan = (anchor[0] - ix * s - (ww - w) / 2, anchor[1] - iy * s - (wh - h) / 2)
        self._pan = self._clamp_pan(pan, w, h)
        self.Refresh()
        self.view_changed.emit()

    def reset_view(self) -> None:
        self._zoom, self._pan = 1.0, (0.0, 0.0)
        self.Refresh()
        self.view_changed.emit()

    def scale(self) -> float:
        """Widget pixels per image pixel."""
        return self.image_rect().w / self._image_size()[0] if self._image is not None else 1.0

    def to_image(self, pos: tuple[float, float], clamp: bool = False) -> tuple[float, float]:
        rect, s = self.image_rect(), self.scale()
        x, y = (pos[0] - rect.x) / s, (pos[1] - rect.y) / s
        if clamp:
            x, y = min(max(x, 0.0), rect.w / s), min(max(y, 0.0), rect.h / s)
        return x, y

    def to_widget(self, x: float, y: float) -> tuple[float, float]:
        rect, s = self.image_rect(), self.scale()
        return rect.x + x * s, rect.y + y * s

    # ---- events --------------------------------------------------------------------------

    def _grab(self) -> None:
        if not self.HasCapture():  # keeps drags going when the pointer leaves the widget
            self.CaptureMouse()

    def _ungrab(self) -> None:
        if self.HasCapture():
            self.ReleaseMouse()

    def _on_wheel(self, event: wx.MouseEvent) -> None:
        if event.GetWheelAxis() == wx.MOUSE_WHEEL_VERTICAL:
            delta = event.GetWheelDelta() or 120
            rotation = event.GetWheelRotation()
            if rotation:
                self.zoom_by(ZOOM_STEP ** (rotation / delta), tuple(event.GetPosition()))

    def _start_pan(self, event: wx.MouseEvent) -> None:
        self._pan_from = tuple(event.GetPosition())
        self._grab()

    def _on_left_down(self, event: wx.MouseEvent) -> None:
        if event.AltDown():
            self._start_pan(event)
            return
        pos = tuple(event.GetPosition())
        if not self.image_rect().contains(pos):
            return
        self._grab()
        self.pressed.emit(*self.to_image(pos), _modifiers(event))

    def _on_middle_down(self, event: wx.MouseEvent) -> None:
        self._start_pan(event)

    def _on_right_down(self, event: wx.MouseEvent) -> None:
        pos = tuple(event.GetPosition())
        if self.image_rect().contains(pos):
            self.right_pressed.emit(*self.to_image(pos))

    def _on_left_dclick(self, event: wx.MouseEvent) -> None:
        # The first click of the double-click already added the final vertex.
        self.finish_requested.emit()

    def _on_motion(self, event: wx.MouseEvent) -> None:
        pos = tuple(event.GetPosition())
        self._cursor = pos
        if self._pan_from is not None:
            w, h = self._view_size()
            moved = (self._pan[0] + pos[0] - self._pan_from[0], self._pan[1] + pos[1] - self._pan_from[1])
            self._pan = self._clamp_pan(moved, w, h)
            self._pan_from = pos
            self.Refresh()
            self.view_changed.emit()
            return
        if event.LeftIsDown():
            self.dragged.emit(*self.to_image(pos, clamp=True))
        else:
            self.hovered.emit(*self.to_image(pos))
        if self._pending:
            self.Refresh()

    def _on_left_up(self, event: wx.MouseEvent) -> None:
        was_panning = self._pan_from is not None
        self._pan_from = None
        self._ungrab()
        if not was_panning:
            self.released.emit()

    def _on_middle_up(self, event: wx.MouseEvent) -> None:
        self._pan_from = None
        self._ungrab()

    def _on_capture_lost(self, event: wx.MouseCaptureLostEvent) -> None:
        self._pan_from = None

    def _on_leave(self, event: wx.MouseEvent) -> None:
        self._cursor = None
        if self._pending:
            self.Refresh()
        event.Skip()

    # ---- painting ------------------------------------------------------------------------

    def _on_paint(self, event: wx.PaintEvent) -> None:
        self.paint(wx.AutoBufferedPaintDC(self))

    def paint(self, dc: wx.DC) -> None:
        dc.SetBackground(wx.BLACK_BRUSH)
        dc.Clear()
        if self._image is None:
            return
        self._draw_image(dc)
        gc = wx.GraphicsContext.Create(dc)
        if gc is not None:
            self._draw_overlay(gc)

    def _draw_image(self, dc: wx.DC) -> None:
        rect, s = self.image_rect(), self.scale()
        ww, wh = self._size()
        # Draw only the visible part: at high zoom the full image rect would be enormous.
        # The region is aligned to whole widget pixels so it can be blitted at an integer position.
        x0, y0 = math.floor(max(rect.x, 0)), math.floor(max(rect.y, 0))
        x1, y1 = math.ceil(min(rect.right, ww)), math.ceil(min(rect.bottom, wh))
        if x1 <= x0 or y1 <= y0:
            return
        cs = max(1.0, self.GetContentScaleFactor())  # render at device resolution on HiDPI/Retina screens
        w, h, d = round((x1 - x0) * cs), round((y1 - y0) * cs), s * cs
        # Map device pixel (u, v) of the region back to image pixel coordinates (pixel centres at integers).
        matrix = np.array(
            [
                [1 / d, 0, (0.5 + x0 * cs - rect.x * cs) / d - 0.5],
                [0, 1 / d, (0.5 + y0 * cs - rect.y * cs) / d - 0.5],
            ]
        )
        interpolation = cv2.INTER_LINEAR if s < SMOOTH_BELOW_SCALE else cv2.INTER_NEAREST  # crisp pixels when zoomed in
        pixels = cv2.warpAffine(
            self._image, matrix, (w, h), flags=interpolation | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
        )
        bitmap = wx.Bitmap.FromBuffer(w, h, pixels)
        if cs != 1.0:
            bitmap.SetScaleFactor(cs)
        dc.DrawBitmap(bitmap, x0, y0)

    def _draw_overlay(self, gc: wx.GraphicsContext) -> None:
        def pen(colour: wx.Colour, width: float, style=wx.PENSTYLE_SOLID) -> wx.GraphicsPenInfo:
            return gc.CreatePen(wx.GraphicsPenInfo(colour).Width(width).Style(style))

        gc.SetAntialiasMode(wx.ANTIALIAS_DEFAULT)
        gc.SetBrush(wx.Brush(FILL))
        for i, polygon in enumerate(self._polygons):
            selected = i == self._selected
            gc.SetPen(pen(SELECTED if selected else OUTLINE, 2.5 if selected else 1.5))
            path = gc.CreatePath()
            points = [self.to_widget(x, y) for x, y in polygon]
            path.MoveToPoint(*points[0])
            for point in points[1:]:
                path.AddLineToPoint(*point)
            path.CloseSubpath()
            gc.DrawPath(path)

        gc.SetBrush(wx.TRANSPARENT_BRUSH)
        if self._selected is not None and self._selected < len(self._polygons):
            gc.SetPen(pen(SELECTED, 1.5))
            for x, y in self._polygons[self._selected]:  # square handles show what can be dragged
                px, py = self.to_widget(x, y)
                gc.DrawRectangle(px - 4, py - 4, 8, 8)

        if self._pending:
            points = [self.to_widget(x, y) for x, y in self._pending]
            gc.SetPen(pen(PENDING, 1.5))
            if len(points) > 1:
                gc.StrokeLines([wx.Point2D(*p) for p in points])
            if self._cursor is not None and self.image_rect().contains(self._cursor):
                gc.SetPen(pen(PENDING, 1, wx.PENSTYLE_SHORT_DASH))
                gc.StrokeLine(*points[-1], *self._cursor)
            gc.SetPen(pen(PENDING, 1.5))
            gc.SetBrush(wx.Brush(PENDING))
            for px, py in points[1:]:
                gc.DrawEllipse(px - 3, py - 3, 6, 6)
            gc.SetBrush(wx.TRANSPARENT_BRUSH)
            gc.DrawEllipse(points[0][0] - 5, points[0][1] - 5, 10, 10)  # ring marks the vertex that closes the polygon
