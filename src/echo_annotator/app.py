"""Main window: drag-and-drop / Ctrl+O loading, playback, and folder navigation."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeyEvent, QKeySequence
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QMainWindow

from .dicom_io import Clip, DicomLoadError, load_clip, sibling_dicoms
from .viewer import ViewerWidget

TITLE = "Echo Annotator"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(TITLE)
        self.resize(960, 720)
        self.setAcceptDrops(True)

        self.viewer = ViewerWidget()
        self.setCentralWidget(self.viewer)

        self._info = QLabel("Drop a DICOM here or press Ctrl+O")
        self.statusBar().addPermanentWidget(self._info, 1)

        open_action = QAction("&Open DICOM...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.open_dialog)
        self.menuBar().addMenu("&File").addAction(open_action)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        self.clip: Clip | None = None
        self.frame = 0
        self.playing = False
        self._siblings: list[Path] = []
        self._index = -1

    # ---- loading -------------------------------------------------------------------------

    def open_dialog(self) -> None:
        start = str(self.clip.path.parent) if self.clip else ""
        name, _ = QFileDialog.getOpenFileName(
            self, "Open DICOM", start, "All files (*);;DICOM files (*.dcm *.dicom)"
        )
        if name:
            self.open_path(Path(name))

    def open_path(self, path: Path) -> bool:
        """Load `path` and start playing it. Leaves the current clip untouched on failure."""
        try:
            clip = load_clip(path)
        except DicomLoadError as exc:
            self.statusBar().showMessage(f"Could not open: {exc}", 8000)
            return False
        self._siblings = sibling_dicoms(path)
        self._index = self._siblings.index(path)
        self._show_clip(clip)
        return True

    def _show_clip(self, clip: Clip) -> None:
        self.clip = clip
        self.frame = 0
        self.setWindowTitle(f"{clip.path.name} - {TITLE}")
        self._render()
        self.set_playing(True)

    def open_sibling(self, step: int) -> None:
        """Move `step` (+1 next / -1 previous) files through the folder, skipping unreadable ones."""
        if not self._siblings:
            return
        i = self._index + step
        while 0 <= i < len(self._siblings):
            try:
                clip = load_clip(self._siblings[i])
            except DicomLoadError:
                i += step
                continue
            self._index = i
            self._show_clip(clip)
            return
        self.statusBar().showMessage(
            "Last DICOM in folder" if step > 0 else "First DICOM in folder", 2000
        )

    # ---- playback ------------------------------------------------------------------------

    def set_playing(self, playing: bool) -> None:
        if self.clip is None:
            return
        self.playing = playing
        if playing:
            self._timer.start(round(1000 / self.clip.fps))
        else:
            self._timer.stop()
        self._update_info()

    def step(self, delta: int) -> None:
        if self.clip is None:
            return
        self.frame = (self.frame + delta) % self.clip.n_frames
        self._render()

    def _on_tick(self) -> None:
        self.step(1)

    def _render(self) -> None:
        if self.clip is not None:
            self.viewer.set_frame(self.clip.frames[self.frame])
            self._update_info()

    def _update_info(self) -> None:
        if self.clip is None:
            return
        state = "Playing" if self.playing else "Paused"
        self._info.setText(
            f"{self.clip.path.name}   file {self._index + 1}/{len(self._siblings)}   "
            f"frame {self.frame + 1}/{self.clip.n_frames}   {self.clip.fps:.0f} fps   {state}"
        )

    # ---- events --------------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
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
