import numpy as np
import pytest
import wx

from driver import ALT, LEFT, MIDDLE, NONE, Driver, send_mouse
from echo_annotator.viewer import MAX_ZOOM, ViewerWidget


def make_viewer(parent, width=800, height=600):
    v = ViewerWidget(parent)
    v.SetSize((width, height))
    v.set_frame(np.zeros((120, 160), np.uint8))  # 4:3 image
    return v


@pytest.fixture
def viewer(parent):
    return make_viewer(parent)  # 4:3 image in a 4:3 widget -> scale 5


def test_fit_preserves_aspect_and_centres(parent):
    v = make_viewer(parent, 900, 300)
    rect = v.image_rect()
    assert rect.h == pytest.approx(300) and rect.w == pytest.approx(400)
    assert rect.x + rect.w / 2 == pytest.approx(450)
    assert v.scale() == pytest.approx(2.5)


def test_empty_viewer_has_empty_rect(parent):
    assert ViewerWidget(parent).image_rect().empty


def test_widget_image_round_trip(viewer):
    for x, y in [(0, 0), (80, 60), (159.5, 119.5), (33.3, 71.9)]:
        p = viewer.to_widget(x, y)
        assert viewer.to_image(p) == pytest.approx((x, y))


def test_to_image_clamps_on_request(viewer):
    assert viewer.to_image((-50, 9999), clamp=True) == pytest.approx((0, 120))
    assert viewer.to_image((-50, 9999)) != pytest.approx((0, 120))


class TestZoom:
    def test_zoom_keeps_the_point_under_the_cursor_fixed(self, viewer):
        anchor = (600, 150)
        before = viewer.to_image(anchor)
        viewer.zoom_by(3, anchor)
        assert viewer.zoom == pytest.approx(3)
        assert viewer.to_image(anchor) == pytest.approx(before, abs=1e-6)

    def test_zoom_is_clamped_to_fit_and_max(self, viewer):
        viewer.zoom_by(0.1)
        assert viewer.zoom == 1.0
        viewer.zoom_by(1e6)
        assert viewer.zoom == MAX_ZOOM

    def test_zoomed_image_always_covers_the_widget(self, viewer):
        viewer.zoom_by(4, (10, 10))
        r = viewer.image_rect()
        assert r.x <= 1e-6 and r.y <= 1e-6
        assert r.right >= 800 - 1e-6 and r.bottom >= 600 - 1e-6

    def test_reset_returns_to_fit(self, viewer):
        fit = viewer.image_rect()
        viewer.zoom_by(5, (100, 100))
        viewer.reset_view()
        assert viewer.zoom == 1.0 and viewer.image_rect() == fit

    def test_view_changed_signal(self, viewer):
        seen = []
        viewer.view_changed.connect(lambda: seen.append(1))
        viewer.zoom_by(2)
        viewer.reset_view()
        assert len(seen) == 2

    def test_wheel_zooms_about_the_cursor(self, window):
        window.viewer.set_frame(np.zeros((120, 160), np.uint8))
        d = Driver(window)
        anchor = (700, 200)
        before = window.viewer.to_image(anchor)
        d.wheel(anchor, 3)
        assert window.viewer.zoom == pytest.approx(1.25**3)
        assert window.viewer.to_image(anchor) == pytest.approx(before, abs=0.05)
        d.wheel(anchor, -30)
        assert window.viewer.zoom == 1.0


def middle_drag(viewer, start, delta_steps, button=MIDDLE, modifier=NONE):
    send_mouse(viewer, "down", button, start, modifier)
    end = start
    for dx, dy in delta_steps:
        end = (end[0] + dx, end[1] + dy)
        send_mouse(viewer, "move", button, end, left_held=button == LEFT)
    send_mouse(viewer, "up", button, end, modifier)


class TestPan:
    def test_middle_drag_pans_by_the_drag_delta(self, viewer):
        viewer.zoom_by(4)
        before = viewer.image_rect()
        middle_drag(viewer, (400, 300), [(40, 25), (80, 55)])
        after = viewer.image_rect()
        assert after.x - before.x == pytest.approx(120, abs=1)
        assert after.y - before.y == pytest.approx(80, abs=1)

    def test_pan_is_clamped_at_the_image_corner(self, viewer):
        viewer.zoom_by(4)
        middle_drag(viewer, (400, 300), [(5000, 5000)])
        r = viewer.image_rect()
        assert abs(r.x) < 1e-6 and abs(r.y) < 1e-6

    def test_alt_left_drag_pans_without_emitting_clicks(self, viewer):
        presses = []
        viewer.pressed.connect(lambda *a: presses.append(a))
        viewer.zoom_by(4)
        before = viewer.image_rect()
        middle_drag(viewer, (400, 300), [(-100, -60)], button=LEFT, modifier=ALT)
        assert viewer.image_rect().x < before.x - 50
        assert presses == []

    def test_no_pan_at_fit_zoom(self, viewer):
        before = viewer.image_rect()
        middle_drag(viewer, (400, 300), [(100, 100)])
        assert viewer.image_rect() == before


class TestMouseSignals:
    def test_left_press_inside_image_emits_image_coordinates(self, viewer):
        got = []
        viewer.pressed.connect(lambda x, y, mods: got.append((x, y)))
        p = viewer.to_widget(40, 30)
        send_mouse(viewer, "down", LEFT, p)
        send_mouse(viewer, "up", LEFT, p)
        assert got[0] == pytest.approx((40, 30), abs=0.2)

    def test_press_outside_the_image_is_ignored(self, parent):
        v = make_viewer(parent, 900, 300)  # pillarboxed: image spans x 250..650
        got = []
        v.pressed.connect(lambda *a: got.append(a))
        send_mouse(v, "down", LEFT, (50, 150))
        send_mouse(v, "up", LEFT, (50, 150))
        assert got == []


class TestPainting:
    def paint(self, viewer, width=800, height=600):
        bitmap = wx.Bitmap(width, height)
        dc = wx.MemoryDC(bitmap)
        viewer.paint(dc)
        dc.SelectObject(wx.NullBitmap)
        return bitmap.ConvertToImage()

    def test_image_fills_its_rect_and_the_rest_is_black(self, parent):
        v = ViewerWidget(parent)
        v.SetSize((900, 300))
        v.set_frame(np.full((120, 160, 3), (200, 100, 50), np.uint8))  # pillarboxed: image spans x 250..650
        image = self.paint(v, 900, 300)
        assert (image.GetRed(450, 150), image.GetGreen(450, 150), image.GetBlue(450, 150)) == (200, 100, 50)
        assert (image.GetRed(100, 150), image.GetGreen(100, 150), image.GetBlue(100, 150)) == (0, 0, 0)

    def test_zoomed_view_shows_the_right_part_of_the_image(self, viewer):
        frame = np.zeros((120, 160, 3), np.uint8)
        frame[:, 80:] = (255, 255, 255)  # right half white
        viewer.set_frame(frame)
        viewer.zoom_by(4, (10, 300))  # anchored at the left edge: only the left (black) part is visible
        assert self.paint(viewer).GetRed(790, 300) == 0
        viewer.zoom_by(1, (790, 300))
        viewer.reset_view()
        viewer.zoom_by(4, (790, 300))  # anchored at the right edge: only the white part is visible
        assert self.paint(viewer).GetRed(10, 300) == 255

    def test_overlay_does_not_crash_with_every_element_present(self, viewer):
        viewer.set_overlay([[(10, 10), (60, 10), (30, 50)], [(80, 20), (120, 20), (100, 60)]], [(5, 5), (20, 5), (20, 20)], 0)
        send_mouse(viewer, "move", NONE, viewer.to_widget(40, 40))
        self.paint(viewer)
