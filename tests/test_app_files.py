"""Opening files, playback, frame stepping, folder navigation."""

import json

import numpy as np
import pytest

import synthetic as syn
from driver import K
from echo_annotator.annotations import sidecar_path


def drop(window, path):
    """Drop `path` on the viewer, which fills the window."""
    return window.viewer.GetDropTarget().OnDropFiles(0, 0, [str(path)])


def shown_value(window):
    """The grey level of the frame currently displayed (numbered_frames: (index + 1) * 10)."""
    return int(window.viewer._image[0, 0, 0])


class TestOpening:
    def test_drop_loads_and_autoplays(self, window, clip_path):
        assert drop(window, clip_path)
        assert window.clip.path == clip_path and window.playing
        assert window.GetTitle().startswith("a.avi")

    def test_window_and_viewer_both_accept_drops(self, window, clip_path):
        assert window.GetDropTarget().OnDropFiles(0, 0, [str(clip_path)])
        assert window.clip.path == clip_path

    def test_an_empty_drop_is_not_accepted(self, window):
        assert not window.GetDropTarget().OnDropFiles(0, 0, [])
        assert window.clip is None

    def test_open_path_returns_false_and_keeps_current_clip_on_failure(self, window, clip_path, tmp_path):
        window.open_path(clip_path)
        bad = tmp_path / "notes.txt"
        bad.write_text("not an echo")
        assert not window.open_path(bad)
        assert "only .avi" in window.current_message()
        assert window.clip.path == clip_path
        assert "Could not open" in window.current_message()

    def test_avi_opens_by_drop_and_uses_its_frame_rate(self, window, tmp_path):
        path = syn.write_avi(tmp_path / "a.avi", syn.drifting_frames(10), fps=25)
        drop(window, path)
        assert window.clip.n_frames == 10 and window.playing
        assert abs(window._timer.GetInterval() - 40) <= 1

    def test_frame_rate_sets_the_timer(self, window, tmp_path):
        window.open_path(syn.write_avi(tmp_path / "a.avi", syn.numbered_frames(4), fps=10))
        assert window._timer.GetInterval() == 100

    def test_zoom_resets_when_a_new_clip_opens(self, window, clip_path, tmp_path):
        window.open_path(clip_path)
        window.viewer.zoom_by(3)
        window.open_path(syn.write_avi(tmp_path / "b.avi", syn.numbered_frames(3)))
        assert window.viewer.zoom == 1.0


class TestPlayback:
    def test_frames_advance_while_playing(self, ui, clip_path):
        ui.w.open_path(clip_path)
        start = ui.w.frame
        assert ui.wait_until(lambda: ui.w.frame != start)

    def test_space_pauses_and_resumes(self, ui, clip_path):
        ui.w.open_path(clip_path)
        ui.key(K.Key_Space)
        assert not ui.w.playing
        frame = ui.w.frame
        ui.wait(250)
        assert ui.w.frame == frame  # really stopped
        ui.key(K.Key_Space)
        assert ui.w.playing

    def test_arrows_step_with_wraparound_when_paused(self, ui, clip_path):
        ui.w.open_path(clip_path)
        ui.pause()
        ui.goto(0)
        ui.key(K.Key_Left)
        assert ui.w.frame == 5  # wraps backwards
        ui.key(K.Key_Right)
        assert ui.w.frame == 0
        ui.key(K.Key_Right)
        assert ui.w.frame == 1

    def test_displayed_pixels_follow_the_frame(self, ui, clip_path):
        ui.w.open_path(clip_path)
        ui.pause()
        for frame in range(6):
            ui.goto(frame)
            assert abs(shown_value(ui.w) - (frame + 1) * 10) <= 3  # MJPG is lossy by a level or two

    def test_arrows_are_ignored_while_playing(self, ui, tmp_path):
        ui.w.open_path(syn.write_avi(tmp_path / "slow.avi", syn.numbered_frames(6), fps=1))  # 1 s per frame
        frame = ui.w.frame
        ui.key(K.Key_Right)
        ui.key(K.Key_Left)
        assert ui.w.frame == frame and ui.w.playing

    def test_new_file_starts_playing_from_frame_zero(self, ui, clip_path, tmp_path):
        ui.w.open_path(clip_path)
        ui.pause()
        ui.goto(3)
        ui.w.open_path(syn.write_avi(tmp_path / "b.avi", syn.numbered_frames(3)))
        assert ui.w.playing and ui.w.frame == 0

    def test_status_bar_describes_the_state(self, ui, clip_path):
        ui.w.open_path(clip_path)
        ui.pause()
        ui.goto(2)
        text = ui.w._info_text
        assert "a.avi" in text and "frame 3/6" in text and "Paused" in text


class TestFolderNavigation:
    @pytest.fixture
    def folder(self, tmp_path):
        syn.write_avi(tmp_path / "a.avi", syn.drifting_frames(20), fps=25)
        syn.write_avi(tmp_path / "b_color.avi", syn.drifting_frames(6), fps=10, color=True)
        syn.write_avi(tmp_path / "c.avi", syn.numbered_frames(5))
        syn.write_avi(tmp_path / "d_lower.AVI", syn.drifting_frames(4), fps=50)
        (tmp_path / "e_broken.avi").write_bytes(b"RIFF" + b"\0" * 200)
        (tmp_path / "f_empty.avi").write_bytes(b"")
        (tmp_path / "notes.txt").write_text("hi")
        return tmp_path

    def test_up_and_down_walk_mixed_formats_in_order_skipping_unreadable(self, ui, folder):
        ui.w.open_path(folder / "a.avi")
        seen = [ui.w.clip.path.name]
        for _ in range(8):
            ui.key(K.Key_Up)
            if ui.w.clip.path.name != seen[-1]:
                seen.append(ui.w.clip.path.name)
        assert seen == ["a.avi", "b_color.avi", "c.avi", "d_lower.AVI"]
        ui.key(K.Key_Down)
        assert ui.w.clip.path.name == "c.avi"

    def test_ends_of_the_folder_do_not_wrap(self, ui, folder):
        ui.w.open_path(folder / "a.avi")
        ui.key(K.Key_Down)
        assert ui.w.clip.path.name == "a.avi" and "First file" in ui.message()
        ui.w.open_path(folder / "d_lower.AVI")
        ui.key(K.Key_Up)
        assert ui.w.clip.path.name == "d_lower.AVI" and "Last file" in ui.message()

    def test_natural_sort(self, ui, tmp_path):
        for name in ("file10.avi", "file2.avi", "file1.avi"):
            syn.write_avi(tmp_path / name, syn.numbered_frames(2))
        ui.w.open_path(tmp_path / "file1.avi")
        order = [ui.w.clip.path.name]
        for _ in range(2):
            ui.key(K.Key_Up)
            order.append(ui.w.clip.path.name)
        assert order == ["file1.avi", "file2.avi", "file10.avi"]

    def test_status_shows_position_in_folder(self, ui, folder):
        ui.w.open_path(folder / "b_color.avi")
        assert "file 2/6" in ui.w._info_text  # the broken ones still count as siblings

    def test_navigating_resets_zoom_and_plays(self, ui, folder):
        ui.w.open_path(folder / "a.avi")
        ui.w.viewer.zoom_by(3)
        ui.key(K.Key_Up)
        assert ui.w.viewer.zoom == 1.0 and ui.w.playing and ui.w.frame == 0


class TestSidecarOnOpen:
    def test_annotations_reload_from_the_sidecar(self, make_window, clip_path):
        first = make_window()
        first.open_path(clip_path)
        first.set_playing(False)
        first.annotations.add(2, [(10.0, 10.0), (50.0, 10.0), (30.0, 40.0)])
        first.annotations.save()
        second = make_window()
        second.open_path(clip_path)
        assert second.annotations.annotated_frames() == [2]

    def test_sidecar_for_a_different_clip_disables_annotation_and_is_not_overwritten(self, ui, clip_path):
        ui.w.open_path(clip_path)
        ui.pause()
        ui.w.annotations.add(0, [(10.0, 10.0), (50.0, 10.0), (30.0, 40.0)])
        ui.w.annotations.save()
        path = sidecar_path(clip_path)
        data = json.loads(path.read_text())
        data["n_frames"] = 99
        path.write_text(json.dumps(data))
        ui.w.open_path(clip_path)  # reloads because the file changed on disk
        assert ui.w.annotations is None
        assert "Annotation disabled" in ui.message()
        ui.pause()
        ui.click(20, 20)
        assert ui.w.pending == []
        assert json.loads(path.read_text())["n_frames"] == 99  # untouched
