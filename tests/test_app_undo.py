"""Undo/redo through the window, including per-file histories kept in memory."""

import json

import pytest

import synthetic as syn
from driver import CTRL, K
from echo_annotator.annotations import sidecar_path

A = [(20, 20), (100, 20), (100, 60), (20, 60)]
B = [(30, 80), (90, 80), (60, 110)]
C = [(120, 90), (150, 90), (135, 110)]


@pytest.fixture
def ui(ui, clip_path):
    ui.w.open_path(clip_path)
    ui.pause()
    ui.goto(0)
    return ui


def saved(ui):
    return json.loads(sidecar_path(ui.w.clip.path).read_text())["frames"]


def test_undo_and_redo_polygons_and_vertex_edits(ui):
    ui.draw(A)
    ui.draw(B)
    ui.drag(100, 60, 140, 100)
    assert abs(ui.polys[0][2][0] - 140) < 1.5
    ui.undo()
    assert abs(ui.polys[0][2][0] - 100) < 1.5 and abs(saved(ui)["0"][0][2][0] - 100) < 1.5
    ui.redo()
    assert abs(ui.polys[0][2][0] - 140) < 1.5 and abs(saved(ui)["0"][0][2][0] - 140) < 1.5
    ui.undo(), ui.undo()
    assert len(ui.polys) == 1
    ui.redo()
    assert len(ui.polys) == 2


def test_undo_all_the_way_back_empties_the_json(ui):
    ui.draw(A)
    ui.undo()
    assert ui.w.annotations.annotated_frames() == [] and saved(ui) == {}
    ui.redo()
    assert len(ui.polys) == 1 and len(saved(ui)["0"]) == 1


def test_redo_at_the_end_and_undo_at_the_start_report_it(ui):
    ui.undo()
    assert "Nothing to undo" in ui.message()
    ui.draw(A)
    ui.redo()
    assert "Nothing to redo" in ui.message()


def test_a_new_edit_discards_redo(ui):
    ui.draw(A)
    ui.undo()
    ui.draw(B)
    ui.redo()
    assert "Nothing to redo" in ui.message() and len(ui.polys) == 1


def test_a_plain_vertex_click_does_not_discard_redo(ui):
    ui.draw(A)
    ui.draw(C)
    ui.undo()
    ui.click(100, 20)  # selects polygon 0, edits nothing
    ui.redo()
    assert len(ui.polys) == 2


def test_ctrl_z_while_drawing_removes_a_vertex_and_ctrl_y_is_ignored(ui):
    ui.draw(A)
    ui.click(10, 10)
    ui.click(30, 10)
    ui.undo()
    assert len(ui.w.pending) == 1 and len(ui.polys) == 1
    ui.redo()
    assert len(ui.w.pending) == 1 and len(ui.polys) == 1


def test_undo_and_redo_jump_to_the_frame_that_changed(ui):
    ui.goto(2)
    ui.draw([(10, 10), (40, 10), (25, 30)])
    ui.goto(4)
    ui.undo()
    assert ui.w.frame == 2 and ui.polys == []
    ui.goto(0)
    ui.redo()
    assert ui.w.frame == 2 and len(ui.polys) == 1


def test_undo_is_refused_while_playing(ui):
    ui.draw(A)
    ui.key(K.Key_Space)
    ui.undo()
    assert "Pause" in ui.message() and len(ui.polys) == 1


class TestPerFileHistory:
    @pytest.fixture
    def two_files(self, ui, tmp_path):
        syn.write_avi(tmp_path / "b.avi", syn.numbered_frames(6))
        ui.w.open_path(ui.w.clip.path)  # the folder listing is taken when a file is opened
        ui.pause()
        ui.goto(0)
        return ui

    def test_each_file_keeps_its_own_stacks_across_switches(self, two_files):
        ui = two_files
        a_set = ui.w.annotations
        ui.draw(A)
        ui.draw(B)
        ui.undo()  # a.avi: 1 undo, 1 redo available
        a_state = (len(a_set._history), len(a_set._redo))
        assert a_state == (1, 1)

        ui.key(K.Key_Up)  # b.avi
        assert ui.w.clip.path.name == "b.avi" and ui.w.annotations is not a_set
        assert ui.w.annotations._history == [] and ui.w.annotations._redo == []
        ui.pause()
        ui.undo()
        assert "Nothing to undo" in ui.message()  # cannot undo a.avi's edits from here
        ui.draw(A)
        ui.draw(B)
        ui.undo()
        b_set = ui.w.annotations
        assert (len(b_set._history), len(b_set._redo)) == (1, 1)

        ui.key(K.Key_Down)  # back to a.avi: same object, same stacks
        assert ui.w.annotations is a_set and (len(a_set._history), len(a_set._redo)) == a_state
        ui.pause()
        ui.redo()
        assert len(ui.polys) == 2  # continues where it left off
        ui.undo()
        ui.undo()
        assert ui.polys == []

        ui.key(K.Key_Up)
        assert ui.w.annotations is b_set and (len(b_set._history), len(b_set._redo)) == (1, 1)

    def test_undoing_a_propagation_still_works_after_switching(self, ui, tmp_path):
        syn.write_avi(tmp_path / "t.avi", syn.drifting_frames(6))  # textured, so tracking has something to hold
        syn.write_avi(tmp_path / "u.avi", syn.numbered_frames(3))
        ui.w.open_path(tmp_path / "t.avi")
        ui.pause()
        ui.goto(0)
        ui.draw(A)
        ui.key(K.Key_T, CTRL)
        assert ui.w.annotations.annotated_frames() == list(range(6))
        ui.key(K.Key_Up)
        assert ui.w.clip.path.name == "u.avi"
        ui.key(K.Key_Down)
        ui.pause()
        ui.undo()
        assert ui.w.annotations.annotated_frames() == [0]

    def test_history_is_memory_only(self, ui, make_window, clip_path):
        ui.draw(A)
        fresh = make_window()  # as if the application had been closed and reopened
        fresh.open_path(clip_path)
        assert fresh.annotations.annotated_frames() == [0]  # the polygons were saved...
        assert fresh.annotations._history == [] and fresh.annotations._redo == []  # ...the history was not
        fresh.set_playing(False)
        fresh.undo()
        assert "Nothing to undo" in fresh.current_message()

    def test_an_externally_changed_sidecar_resets_that_files_history(self, two_files):
        ui = two_files
        ui.draw(A)
        first = ui.w.annotations
        ui.key(K.Key_Up)
        path = sidecar_path(first.path)
        data = json.loads(path.read_text())
        data["frames"] = {}
        path.write_text(json.dumps(data))
        ui.key(K.Key_Down)
        assert ui.w.annotations is not first and ui.w.annotations._history == []
        assert ui.w.annotations.annotated_frames() == []
