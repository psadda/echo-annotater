"""Widget that displays a single frame scaled to fit, preserving aspect ratio."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QWidget


class ViewerWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._image: QImage | None = None

    def set_frame(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        fmt = QImage.Format.Format_RGB888 if frame.ndim == 3 else QImage.Format.Format_Grayscale8
        # copy() so the QImage owns its pixels and never outlives the numpy buffer
        self._image = QImage(frame.data, width, height, frame.strides[0], fmt).copy()
        self.update()

    def image_rect(self) -> QRectF:
        """Widget-space rectangle the image is drawn into (empty if nothing is loaded)."""
        if self._image is None:
            return QRectF()
        scale = min(self.width() / self._image.width(), self.height() / self._image.height())
        w, h = self._image.width() * scale, self._image.height() * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self._image is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(self.image_rect(), self._image)
