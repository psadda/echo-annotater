import pytest
import wx

import synthetic as syn
from driver import Driver, pump


@pytest.fixture(scope="session")
def wxapp():
    return wx.GetApp() or wx.App(False)


@pytest.fixture
def parent(wxapp):
    """A hidden top-level window to host widgets that are tested on their own."""
    frame = wx.Frame(None)
    yield frame
    frame.Destroy()


@pytest.fixture
def make_window(wxapp, tmp_path):
    """Create windows that share one throwaway settings file, so the real user profile is never touched."""
    from echo_annotator.app import MainWindow

    settings_path = str(tmp_path / "settings.ini")
    windows = []

    def make():
        window = MainWindow(wx.FileConfig(localFilename=settings_path))
        window.SetClientSize((960, 720))
        window.ShowWithoutActivating()
        window.Layout()
        pump()
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.Close()
        window.Destroy()
    pump()


@pytest.fixture
def window(make_window):
    return make_window()


@pytest.fixture
def ui(window):
    return Driver(window)


@pytest.fixture
def clip_path(tmp_path):
    """A 6-frame 160x120 AVI whose frame i is uniformly about (i + 1) * 10 (MJPG is lossy by a level or two)."""
    return syn.write_avi(tmp_path / "a.avi", syn.numbered_frames(6), fps=25)
