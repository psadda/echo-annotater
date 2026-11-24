"""Ctrl+R (grow), Ctrl+T (propagate) and the curvature option, through the window."""

import json

import numpy as np
import pytest

import synthetic as syn
from driver import CTRL, K
from echo_annotator.annotations import rasterize, sidecar_path
from echo_annotator.geometry import polygon_area
from synthetic import curve_distance

N, SRC = 24, 10
STRIPE_POLY = [(90, 108), (200, 108), (200, 114), (90, 114)]
OTHER_POLY = [(30, 20), (60, 20), (45, 40)]


@pytest.fixture(scope="module")
def scene():
    return syn.scene_frames(N)


@pytest.fixture
def ui(ui, scene, tmp_path):
    path = syn.write_avi(tmp_path / "scene.avi", scene)
    ui.w.open_path(path)
    ui.pause()
    ui.goto(SRC)
    return ui


def grow(ui):
    ui.key(K.Key_R, CTRL)


def propagate(ui):
    ui.key(K.Key_T, CTRL)


def snapshot(ui):
    return {f: [list(p) for p in ps] for f, ps in ui.w.annotations.polygons.items()}


def saved_frames(ui):
    return json.loads(sidecar_path(ui.w.clip.path).read_text())["frames"]


class TestGuards:
    @pytest.mark.parametrize("tool", [grow, propagate])
    def test_no_polygon(self, ui, tool):
        tool(ui)
        assert "No polygon" in ui.message()

    @pytest.mark.parametrize("tool", [grow, propagate])
    def test_two_polygons_and_none_selected_asks_to_select(self, ui, tool):
        ui.draw(STRIPE_POLY)
        ui.draw(OTHER_POLY)
        before = snapshot(ui)
        tool(ui)
        assert "Select a polygon" in ui.message() and snapshot(ui) == before

    @pytest.mark.parametrize("tool", [grow, propagate])
    def test_refused_while_drawing(self, ui, tool):
        ui.draw(STRIPE_POLY)
        ui.click(10, 10)
        tool(ui)
        assert "Finish the polygon" in ui.message()

    @pytest.mark.parametrize("tool", [grow, propagate])
    def test_refused_while_playing(self, ui, tool):
        ui.draw(STRIPE_POLY)
        ui.key(K.Key_Space)
        tool(ui)
        assert "Pause" in ui.message()


class TestGrow:
    def test_grows_inside_the_stripe_and_is_one_undo_step(self, ui):
        ui.draw(STRIPE_POLY)
        ui.draw(OTHER_POLY)
        ui.click(120, 111, CTRL)
        assert ui.w.selected == 0
        before, other = [list(p) for p in ui.polys[0]], [list(p) for p in ui.polys[1]]
        history = len(ui.w.annotations._history)
        grow(ui)
        grown = ui.polys[0]

        assert polygon_area(grown) > 2 * polygon_area(before)
        mask = rasterize([grown], syn.SCENE_H, syn.SCENE_W) > 0
        assert (mask & (syn.stripe_mask(SRC) > 0)).sum() / mask.sum() > 0.9  # stays in the dark stripe
        assert 16 <= len(grown) <= 32
        assert [list(p) for p in ui.polys[1]] == other  # the other polygon is untouched
        assert ui.w.selected == 0
        assert len(ui.w.annotations._history) == history + 1
        assert ui.w.annotations.annotated_frames() == [SRC]
        assert len(saved_frames(ui)[str(SRC)][0]) == len(grown)

        ui.undo()
        assert [list(p) for p in ui.polys[0]] == before  # exact
        ui.redo()
        assert [list(p) for p in ui.polys[0]] == [list(p) for p in grown]

    def test_single_polygon_needs_no_selection(self, ui):
        ui.draw(STRIPE_POLY)
        grow(ui)
        assert polygon_area(ui.polys[0]) > 6 * 110

    def test_repeated_presses_end_with_nothing_to_grow(self, ui):
        ui.draw(STRIPE_POLY)
        for _ in range(30):  # ~10 px per press; the stripe takes a dozen presses to fill
            grow(ui)
            if "Nothing to grow" in ui.message():
                break
        assert "Nothing to grow" in ui.message()
        assert ui.w.annotations.annotated_frames() == [SRC]


class TestPropagate:
    def test_fills_every_frame_as_one_undo_step_with_exact_restore(self, ui):
        ui.goto(SRC + 3)
        ui.draw(OTHER_POLY)  # pre-existing annotation on another frame: must be replaced, and restorable
        ui.goto(SRC)
        ui.draw(STRIPE_POLY)
        ui.click(120, 111, CTRL)
        source_polygon = [tuple(p) for p in ui.polys[ui.w.selected]]
        before = snapshot(ui)
        history = len(ui.w.annotations._history)

        propagate(ui)
        assert ui.w.annotations.annotated_frames() == list(range(N))
        assert "forward" in ui.message() and "replaced annotations on 1" in ui.message()
        assert len(ui.w.annotations._history) == history + 1
        assert [tuple(p) for p in ui.polys[ui.w.selected]] == source_polygon  # source frame untouched
        assert all(len(ui.w.annotations.on_frame(t)) == 1 for t in range(N) if t != SRC)
        assert sorted(map(int, saved_frames(ui))) == list(range(N))
        after = snapshot(ui)

        ui.goto(5)
        ui.undo()
        assert ui.w.frame == SRC and snapshot(ui) == before
        assert sorted(map(int, saved_frames(ui))) == sorted(before)
        ui.goto(2)
        ui.redo()
        assert ui.w.frame == SRC and snapshot(ui) == after

    def test_tracked_polygons_follow_the_true_motion(self, ui):
        ui.draw(STRIPE_POLY)
        source = np.asarray([tuple(p) for p in ui.polys[0]]) - 0.5
        propagate(ui)
        errors = []
        for t in range(N):
            if t == SRC:
                continue
            m = syn.scene_motion(t) @ np.linalg.inv(syn.scene_motion(SRC))
            truth = source @ m[:2, :2].T + m[:2, 2] + 0.5
            errors.append(curve_distance(ui.w.annotations.on_frame(t)[0], [tuple(q) for q in truth]))
        assert max(errors) < 3.0
        counts = {len(ui.w.annotations.on_frame(t)[0]) for t in range(N) if t != SRC}
        assert len(counts) == 1 and 16 <= counts.pop() <= 32

    def test_a_new_edit_after_undoing_a_propagation_clears_redo(self, ui):
        ui.draw(STRIPE_POLY)
        propagate(ui)
        ui.undo()
        ui.draw(OTHER_POLY)
        ui.redo()
        assert "Nothing to redo" in ui.message()


class TestOtherClipKinds:
    def test_single_frame_clip(self, ui, scene, tmp_path):
        ui.w.open_path(syn.write_avi(tmp_path / "one.avi", scene[:1]))
        ui.pause()
        ui.draw(STRIPE_POLY)
        history = len(ui.w.annotations._history)
        propagate(ui)
        assert "Could not track" in ui.message() and len(ui.w.annotations._history) == history
        grow(ui)
        assert polygon_area(ui.polys[0]) > 6 * 110 and len(ui.w.annotations._history) == history + 1

    def test_colour_video(self, ui, scene, tmp_path):
        ui.w.open_path(syn.write_avi(tmp_path / "rgb.avi", scene[:4], color=True))
        ui.pause()
        ui.goto(0)
        ui.draw(STRIPE_POLY)
        grow(ui)
        propagate(ui)
        assert ui.w.annotations.annotated_frames() == [0, 1, 2, 3]


class TestCurvatureOption:
    def test_off_by_default_and_flag_reaches_the_tracker(self, ui, monkeypatch):
        from echo_annotator import app as app_module

        seen = []
        real = app_module.propagate_polygon
        monkeypatch.setattr(app_module, "propagate_polygon", lambda g, n, s, p, preserve=False: (seen.append(preserve), real(g, n, s, p, preserve))[1])
        assert not ui.w.preserve_curvature
        ui.draw(STRIPE_POLY)
        propagate(ui)
        assert seen == [False] and "curvature preserved" not in ui.message()

        ui.undo()
        ui.w.set_preserve_curvature(True)
        assert ui.w.preserve_curvature
        propagate(ui)
        assert seen == [False, True] and "curvature preserved" in ui.message()
        assert ui.w.annotations.annotated_frames() == list(range(N))

    def test_choice_persists_to_new_windows(self, make_window):
        first = make_window()
        assert not first.preserve_curvature
        first.set_preserve_curvature(True)
        assert make_window().preserve_curvature
        first.set_preserve_curvature(False)
        assert not make_window().preserve_curvature
