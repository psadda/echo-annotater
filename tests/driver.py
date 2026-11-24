"""Drive a MainWindow with real mouse and keyboard events, in image coordinates.

Events are built by hand and delivered straight to the widget's event handler, so no real input device is needed.
"""

from __future__ import annotations

import sys
import time

import wx

from echo_annotator.viewer import MOD_ALT, MOD_CTRL, MOD_SHIFT

NONE, CTRL, SHIFT, ALT = 0, MOD_CTRL, MOD_SHIFT, MOD_ALT
LEFT, RIGHT, MIDDLE = "left", "right", "middle"

_BUTTON_EVENTS = {
    (LEFT, "down"): wx.wxEVT_LEFT_DOWN,
    (LEFT, "up"): wx.wxEVT_LEFT_UP,
    (LEFT, "dclick"): wx.wxEVT_LEFT_DCLICK,
    (RIGHT, "down"): wx.wxEVT_RIGHT_DOWN,
    (RIGHT, "up"): wx.wxEVT_RIGHT_UP,
    (MIDDLE, "down"): wx.wxEVT_MIDDLE_DOWN,
    (MIDDLE, "up"): wx.wxEVT_MIDDLE_UP,
}


class K:
    """Key codes under the names the tests use."""

    Key_Space = wx.WXK_SPACE
    Key_Left, Key_Right, Key_Up, Key_Down = wx.WXK_LEFT, wx.WXK_RIGHT, wx.WXK_UP, wx.WXK_DOWN
    Key_Return = wx.WXK_RETURN
    Key_Escape = wx.WXK_ESCAPE
    Key_Backspace = wx.WXK_BACK
    Key_Delete = wx.WXK_DELETE
    Key_R, Key_T, Key_Y, Key_Z = ord("R"), ord("T"), ord("Y"), ord("Z")


def _set_modifiers(event, modifiers: int) -> None:
    # Ctrl is Cmd on macOS, which is what the app's CmdDown() checks.
    if sys.platform == "darwin":
        event.SetMetaDown(bool(modifiers & CTRL))
    else:
        event.SetControlDown(bool(modifiers & CTRL))
    event.SetShiftDown(bool(modifiers & SHIFT))
    event.SetAltDown(bool(modifiers & ALT))


def pump(ms: int = 0) -> None:
    """Run the event loop (timers, paints) for at least `ms` milliseconds."""
    end = time.monotonic() + ms / 1000
    loop = wx.GUIEventLoop()
    with wx.EventLoopActivator(loop):  # wx.Yield() outside MainLoop() does not fire timers
        while True:
            while loop.Pending():
                loop.Dispatch()
            if time.monotonic() >= end:
                return
            time.sleep(0.005)


def send_mouse(window: wx.Window, kind: str, button: str, pos, modifiers: int = NONE, left_held: bool = False) -> None:
    """Deliver a mouse event to `window`; `kind` is "down", "up", "dclick" or "move"."""
    event_type = wx.wxEVT_MOTION if kind == "move" else _BUTTON_EVENTS[(button, kind)]
    event = wx.MouseEvent(event_type)
    event.SetEventObject(window)
    event.SetPosition(wx.Point(round(pos[0]), round(pos[1])))
    _set_modifiers(event, modifiers)
    event.SetLeftDown(left_held or (button == LEFT and kind in ("down", "dclick")))
    window.GetEventHandler().ProcessEvent(event)


def send_wheel(window: wx.Window, pos, notches: float) -> None:
    event = wx.MouseEvent(wx.wxEVT_MOUSEWHEEL)
    event.SetEventObject(window)
    event.SetPosition(wx.Point(round(pos[0]), round(pos[1])))
    event.SetWheelAxis(wx.MOUSE_WHEEL_VERTICAL)
    event.SetWheelDelta(120)
    event.SetWheelRotation(int(120 * notches))
    window.GetEventHandler().ProcessEvent(event)


class Driver:
    def __init__(self, window) -> None:
        self.w = window
        self.v = window.viewer

    # ---- conversions ----
    def px(self, x: float, y: float) -> tuple[int, int]:
        p = self.v.to_widget(x, y)
        return round(p[0]), round(p[1])

    # ---- keyboard ----
    def key(self, key: int, modifier: int = NONE) -> None:
        event = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
        event.SetEventObject(self.w)
        event.SetKeyCode(key)
        event.SetUnicodeKey(key)
        _set_modifiers(event, modifier)
        if not self.w.GetEventHandler().ProcessEvent(event) and modifier == CTRL:
            self._menu_accelerator(key)

    def _menu_accelerator(self, key: int) -> None:
        """What the OS does for keys the window did not take: fire the menu item with that accelerator."""
        wanted = "Ctrl+" + chr(key)
        for menu, _ in self.w.GetMenuBar().GetMenus():
            for item in menu.GetMenuItems():
                if item.GetItemLabel().endswith("\t" + wanted):
                    self.w.GetEventHandler().ProcessEvent(wx.CommandEvent(wx.wxEVT_MENU, item.GetId()))
                    return

    def undo(self) -> None:
        self.key(K.Key_Z, CTRL)

    def redo(self) -> None:
        self.key(K.Key_Y, CTRL)

    # ---- mouse (image coordinates unless stated) ----
    def mouse_down(self, pos, modifier: int = NONE, button: str = LEFT) -> None:
        send_mouse(self.v, "down", button, pos, modifier)

    def mouse_move(self, pos, left_held: bool = False) -> None:
        send_mouse(self.v, "move", LEFT, pos, left_held=left_held)

    def mouse_up(self, pos, modifier: int = NONE, button: str = LEFT) -> None:
        send_mouse(self.v, "up", button, pos, modifier)

    def click(self, x: float, y: float, modifier: int = NONE, button: str = LEFT) -> None:
        self.mouse_down(self.px(x, y), modifier, button)
        self.mouse_up(self.px(x, y), modifier, button)

    def double_click(self, x: float, y: float) -> None:
        pos = self.px(x, y)
        self.mouse_down(pos)
        self.mouse_up(pos)
        send_mouse(self.v, "dclick", LEFT, pos)
        self.mouse_up(pos)

    def drag(self, x0: float, y0: float, x1: float, y1: float, modifier: int = NONE, button: str = LEFT) -> None:
        held = button == LEFT
        self.mouse_down(self.px(x0, y0), modifier, button)
        self.mouse_move(self.px((x0 + x1) / 2, (y0 + y1) / 2), left_held=held)
        self.mouse_move(self.px(x1, y1), left_held=held)
        self.mouse_up(self.px(x1, y1), modifier, button)

    def draw(self, points, close: bool = True) -> None:
        """Click vertices; press Enter to close the polygon."""
        for x, y in points:
            self.click(x, y)
        if close:
            self.key(K.Key_Return)

    def wheel(self, pos, notches: float) -> None:
        send_wheel(self.v, pos, notches)

    # ---- state shortcuts ----
    def pause(self) -> None:
        if self.w.playing:
            self.key(K.Key_Space)

    def goto(self, frame: int) -> None:
        self.w.frame = frame
        self.w._render()

    @property
    def polys(self):
        return self.w.annotations.on_frame(self.w.frame)

    def message(self) -> str:
        return self.w.current_message()

    def wait(self, ms: int) -> None:
        pump(ms)

    def wait_until(self, condition, timeout_ms: int = 2000) -> bool:
        end = time.monotonic() + timeout_ms / 1000
        while not condition():
            if time.monotonic() >= end:
                return condition()
            pump(10)
        return True
