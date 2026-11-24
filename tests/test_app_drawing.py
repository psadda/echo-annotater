"""Drawing polygons and the vertex/polygon editing gestures."""

import json

import pytest

from driver import CTRL, K, LEFT, NONE, RIGHT, SHIFT
from echo_annotator.annotations import sidecar_path

RECT = [(20, 20), (100, 20), (100, 60), (20, 60)]


@pytest.fixture
def ui(ui, clip_path):
    ui.w.open_path(clip_path)
    ui.pause()
    ui.goto(0)
    return ui


def saved_frames(ui):
    return json.loads(sidecar_path(ui.w.clip.path).read_text())["frames"]


class TestDrawing:
    def test_clicks_while_playing_are_ignored(self, ui):
        ui.key(K.Key_Space)
        ui.click(20, 20)
        assert ui.w.pending == [] and "Pause" in ui.message()

    def test_enter_closes_the_polygon_and_saves(self, ui):
        ui.draw(RECT)
        assert len(ui.polys) == 1 and ui.w.pending == []
        assert len(saved_frames(ui)["0"][0]) == 4

    def test_clicks_map_to_exact_image_coordinates(self, ui):
        for pt in RECT:
            ui.click(*pt)
        assert all(abs(a[0] - b[0]) < 0.2 and abs(a[1] - b[1]) < 0.2 for a, b in zip(ui.w.pending, RECT))

    def test_no_mask_files_are_ever_written(self, ui):
        ui.draw(RECT)
        folder = sidecar_path(ui.w.clip.path).parent
        assert not list(folder.glob("*.masks")) and not list(folder.glob("*.png"))

    def test_backspace_removes_only_the_last_vertex(self, ui):
        for pt in RECT:
            ui.click(*pt)
        ui.key(K.Key_Backspace)
        assert len(ui.w.pending) == 3

    def test_backspace_never_removes_a_finished_polygon(self, ui):
        ui.draw(RECT)
        ui.key(K.Key_Backspace)
        assert len(ui.polys) == 1

    def test_clicking_the_first_vertex_closes_the_polygon(self, ui):
        for pt in [(30, 80), (70, 80), (70, 100)]:
            ui.click(*pt)
        ui.click(30.5, 80.5)
        assert len(ui.polys) == 1 and ui.w.pending == []

    def test_double_click_closes_the_polygon(self, ui):
        for pt in [(110, 60), (140, 60)]:
            ui.click(*pt)
        ui.click(140, 90)
        ui.double_click(140, 90)
        assert len(ui.polys) == 1

    def test_fewer_than_three_points_cannot_close(self, ui):
        ui.click(5, 5)
        ui.key(K.Key_Return)
        assert len(ui.w.pending) == 1 and ui.polys == []
        assert "at least 3" in ui.message()

    def test_escape_discards_the_polygon_in_progress(self, ui):
        ui.click(5, 5)
        ui.key(K.Key_Escape)
        assert ui.w.pending == []

    def test_stepping_a_frame_discards_the_polygon_in_progress(self, ui):
        ui.click(5, 5)
        ui.key(K.Key_Right)
        assert ui.w.pending == [] and ui.w.frame == 1

    def test_resuming_playback_discards_the_polygon_in_progress(self, ui):
        ui.click(5, 5)
        ui.key(K.Key_Space)
        assert ui.w.pending == []

    def test_polygons_belong_to_their_frame(self, ui):
        ui.draw(RECT)
        ui.goto(2)
        ui.draw([(10, 10), (30, 10), (20, 30)])
        assert ui.w.annotations.annotated_frames() == [0, 2]
        assert set(saved_frames(ui)) == {"0", "2"}

    def test_clicks_outside_the_image_are_ignored(self, ui):
        ui.w.viewer.SetSize((900, 300))  # pillarboxed: the image only covers the middle
        ui.v.set_frame(ui.w.clip.frames[0])
        ui.mouse_down((20, 150))
        ui.mouse_up((20, 150))
        assert ui.w.pending == []

    def test_delete_without_a_selection_clears_the_frame(self, ui):
        ui.draw(RECT)
        ui.draw([(30, 80), (90, 80), (60, 110)])
        ui.key(K.Key_Delete)
        assert ui.w.annotations.annotated_frames() == [] and saved_frames(ui) == {}


class TestVertexEditing:
    def test_dragging_a_vertex_moves_selects_and_saves(self, ui):
        ui.draw(RECT)
        ui.drag(100, 60, 140, 100)
        x, y = ui.polys[0][2]
        assert (abs(x - 140) < 1.5 and abs(y - 100) < 1.5) and ui.w.selected == 0
        assert abs(saved_frames(ui)["0"][0][2][0] - 140) < 1.5
        assert len(ui.polys) == 1  # did not start a new polygon

    def test_a_plain_click_on_a_vertex_changes_nothing(self, ui):
        ui.draw(RECT)
        history = len(ui.w.annotations._history)
        ui.click(100, 60)
        assert len(ui.w.annotations._history) == history and ui.w.selected == 0
        assert ui.polys[0][2] == pytest.approx((100, 60), abs=0.2)

    def test_dragging_outside_the_image_is_clamped(self, ui):
        ui.draw(RECT)
        ui.drag(100, 60, 5000, 5000)
        x, y = ui.polys[0][2]
        assert x <= 160.0001 and y <= 120.0001

    def test_shift_click_on_an_edge_inserts_a_vertex(self, ui):
        ui.draw(RECT)
        ui.click(60, 20, SHIFT)
        assert len(ui.polys[0]) == 5
        assert ui.polys[0][1] == pytest.approx((60, 20), abs=1.5)
        assert len(saved_frames(ui)["0"][0]) == 5

    def test_shift_drag_inserts_and_moves_in_one_undo_step(self, ui):
        ui.draw(RECT)
        ui.drag(60, 20, 60, 5, SHIFT)
        assert len(ui.polys[0]) == 5 and abs(ui.polys[0][1][1] - 5) < 1.5
        ui.undo()
        assert len(ui.polys[0]) == 4

    def test_shift_click_away_from_edges_does_nothing(self, ui):
        ui.draw(RECT)
        ui.click(60, 40, SHIFT)
        assert len(ui.polys[0]) == 4 and ui.w.pending == []

    def test_right_click_deletes_a_vertex_but_never_below_three(self, ui):
        ui.draw(RECT)
        ui.click(100, 60, button=RIGHT)
        assert len(ui.polys[0]) == 3
        ui.click(20, 60, button=RIGHT)
        assert len(ui.polys[0]) == 3 and "at least 3" in ui.message()

    def test_hit_radius_is_in_screen_pixels_so_zoom_gives_precision(self, ui):
        ui.draw(RECT)
        ui.v.zoom_by(5, ui.v.to_widget(100, 20))
        s = ui.v.scale()
        ui.drag(100 + 6 / s, 20, 100 + 9 / s, 20 + 3 / s)  # 6 screen px away: grabs the vertex
        assert ui.polys[0][1] != (100, 20) and len(ui.polys) == 1
        ui.click(100 + 14 / s, 20 + 14 / s)  # 14 screen px away: starts a new polygon instead
        assert len(ui.w.pending) == 1

    def test_stepping_ends_a_drag_in_progress_and_keeps_the_edit(self, ui):
        ui.draw(RECT)
        ui.mouse_down(ui.px(100, 20))
        ui.mouse_move(ui.px(110, 25), left_held=True)
        ui.key(K.Key_Right)
        assert ui.w._drag is None and ui.w.frame == 1
        ui.mouse_up(ui.px(110, 25))  # harmless
        ui.goto(0)
        assert abs(ui.polys[0][1][0] - 110) < 1.5


class TestSelection:
    def test_ctrl_click_selects_deselects_and_picks_the_topmost(self, ui):
        ui.draw(RECT)
        ui.draw([(30, 80), (90, 80), (60, 110)])
        ui.click(60, 40, CTRL)
        assert ui.w.selected == 0
        ui.click(150, 5, CTRL)
        assert ui.w.selected is None
        ui.click(60, 95, CTRL)
        assert ui.w.selected == 1
        ui.key(K.Key_Escape)
        assert ui.w.selected is None

    def test_delete_removes_only_the_selected_polygon(self, ui):
        ui.draw(RECT)
        ui.draw([(30, 80), (90, 80), (60, 110)])
        ui.click(60, 95, CTRL)
        ui.key(K.Key_Delete)
        assert len(ui.polys) == 1 and ui.w.selected is None
        assert ui.polys[0][0] == pytest.approx((20, 20), abs=0.2)
