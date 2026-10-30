"""Widget that displays a single frame scaled to fit, with a polygon overlay and mouse input."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPolygonF, QWheelEvent
from PySide6.QtWidgets import QWidget

Polygon = list[tuple[float, float]]

MAX_ZOOM = 20.0
ZOOM_STEP = 1.25  # per wheel notch
SMOOTH_BELOW_SCALE = 2.0  # widget pixels per image pixel

FILL = QColor(255, 200, 0, 70)
OUTLINE = QColor(255, 200, 0)
SELECTED = QColor(255, 255, 255)
PENDING = QColor(0, 220, 255)


class ViewerWidget(QWidget):
    # All coordinates are image coordinates.
    pressed = Signal(float, float, object)  # left press inside the image, with Qt.KeyboardModifier flags
    dragged = Signal(float, float)  # left button held; clamped to the image
    released = Signal()
    right_pressed = Signal(float, float)
    hovered = Signal(float, float)  # no button held
    finish_requested = Signal()  # double-click
    view_changed = Signal()  # zoom or pan changed

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)  # for the rubber-band line while drawing
        self._image: QImage | None = None
        self._polygons: list[Polygon] = []
        self._pending: Polygon = []
        self._selected: int | None = None
        self._cursor: QPointF | None = None
        self._zoom = 1.0  # 1 = fit to window
        self._pan = QPointF(0, 0)  # offset in widget pixels from the centred position
        self._pan_from: QPointF | None = None

    def set_frame(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        fmt = QImage.Format.Format_RGB888 if frame.ndim == 3 else QImage.Format.Format_Grayscale8
        # copy() so the QImage owns its pixels and never outlives the numpy buffer
        self._image = QImage(frame.data, width, height, frame.strides[0], fmt).copy()
        self.update()

    def set_overlay(self, polygons: list[Polygon], pending: Polygon, selected: int | None) -> None:
        self._polygons = polygons
        self._pending = pending
        self._selected = selected
        self.update()

    # ---- geometry ------------------------------------------------------------------------

    @property
    def zoom(self) -> float:
        return self._zoom

    def _view_size(self) -> tuple[float, float]:
        """Size in widget pixels of the whole image at the current zoom."""
        fit = min(self.width() / self._image.width(), self.height() / self._image.height())
        return self._image.width() * fit * self._zoom, self._image.height() * fit * self._zoom

    def _clamp_pan(self, pan: QPointF, w: float, h: float) -> QPointF:
        """Keep a zoomed image covering the widget; an image smaller than the widget stays centred."""
        limit_x, limit_y = max(0.0, (w - self.width()) / 2), max(0.0, (h - self.height()) / 2)
        return QPointF(min(max(pan.x(), -limit_x), limit_x), min(max(pan.y(), -limit_y), limit_y))

    def image_rect(self) -> QRectF:
        """Widget-space rectangle the image is drawn into (empty if nothing is loaded)."""
        if self._image is None:
            return QRectF()
        w, h = self._view_size()
        pan = self._clamp_pan(self._pan, w, h)
        return QRectF((self.width() - w) / 2 + pan.x(), (self.height() - h) / 2 + pan.y(), w, h)

    def zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        """Zoom by `factor`, keeping the image point under `anchor` (default: widget centre) fixed."""
        if self._image is None:
            return
        anchor = anchor if anchor is not None else QPointF(self.width() / 2, self.height() / 2)
        ix, iy = self.to_image(anchor)
        self._zoom = min(max(self._zoom * factor, 1.0), MAX_ZOOM)
        w, h = self._view_size()
        s = w / self._image.width()
        pan = QPointF(anchor.x() - ix * s - (self.width() - w) / 2, anchor.y() - iy * s - (self.height() - h) / 2)
        self._pan = self._clamp_pan(pan, w, h)
        self.update()
        self.view_changed.emit()

    def reset_view(self) -> None:
        self._zoom, self._pan = 1.0, QPointF(0, 0)
        self.update()
        self.view_changed.emit()

    def scale(self) -> float:
        """Widget pixels per image pixel."""
        return self.image_rect().width() / self._image.width() if self._image else 1.0

    def to_image(self, pos: QPointF, clamp: bool = False) -> tuple[float, float]:
        rect, s = self.image_rect(), self.scale()
        x, y = (pos.x() - rect.x()) / s, (pos.y() - rect.y()) / s
        if clamp:
            x, y = min(max(x, 0.0), rect.width() / s), min(max(y, 0.0), rect.height() / s)
        return x, y

    def to_widget(self, x: float, y: float) -> QPointF:
        rect, s = self.image_rect(), self.scale()
        return QPointF(rect.x() + x * s, rect.y() + y * s)

    # ---- events --------------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta:
            self.zoom_by(ZOOM_STEP ** (delta / 120), event.position())
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        button = event.button()
        if button == Qt.MouseButton.MiddleButton or (
            button == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.AltModifier
        ):
            self._pan_from = event.position()
            return
        if not self.image_rect().contains(event.position()):
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressed.emit(*self.to_image(event.position()), event.modifiers())
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_pressed.emit(*self.to_image(event.position()))

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        # The first click of the double-click already added the final vertex.
        if event.button() == Qt.MouseButton.LeftButton:
            self.finish_requested.emit()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._cursor = event.position()
        if self._pan_from is not None:
            w, h = self._view_size()
            self._pan = self._clamp_pan(self._pan + event.position() - self._pan_from, w, h)
            self._pan_from = event.position()
            self.update()
            self.view_changed.emit()
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.dragged.emit(*self.to_image(event.position(), clamp=True))
        else:
            self.hovered.emit(*self.to_image(event.position()))
        if self._pending:
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._pan_from is not None:
            if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
                self._pan_from = None
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.released.emit()

    def leaveEvent(self, event) -> None:
        self._cursor = None
        if self._pending:
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self._image is None:
            return
        rect, s = self.image_rect(), self.scale()
        # Draw only the visible part: at high zoom the full target rect would be enormous.
        target = rect.intersected(QRectF(self.rect()))
        source = QRectF((target.x() - rect.x()) / s, (target.y() - rect.y()) / s, target.width() / s, target.height() / s)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, s < SMOOTH_BELOW_SCALE)  # show crisp pixels when zoomed in
        painter.drawImage(target, self._image, source)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setBrush(FILL)
        for i, polygon in enumerate(self._polygons):
            painter.setPen(QPen(SELECTED if i == self._selected else OUTLINE, 2.5 if i == self._selected else 1.5))
            painter.drawPolygon(QPolygonF([self.to_widget(x, y) for x, y in polygon]))

        if self._selected is not None and self._selected < len(self._polygons):
            painter.setPen(QPen(SELECTED, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for x, y in self._polygons[self._selected]:  # square handles show what can be dragged
                p = self.to_widget(x, y)
                painter.drawRect(QRectF(p.x() - 4, p.y() - 4, 8, 8))

        if self._pending:
            points = [self.to_widget(x, y) for x, y in self._pending]
            painter.setPen(QPen(PENDING, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolyline(QPolygonF(points))
            if self._cursor is not None and self.image_rect().contains(self._cursor):
                painter.setPen(QPen(PENDING, 1, Qt.PenStyle.DashLine))
                painter.drawLine(points[-1], self._cursor)
            painter.setPen(QPen(PENDING, 1.5))
            painter.setBrush(PENDING)
            for point in points[1:]:
                painter.drawEllipse(point, 3, 3)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(points[0], 5, 5)  # ring marks the vertex that closes the polygon
