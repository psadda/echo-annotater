import numpy as np
import pytest

import synthetic as syn
from echo_annotator.clip_io import ClipLoadError, _clamp_fps, load_clip, sibling_clips


class TestLoadAvi:
    def test_grayscale_video_keeps_one_channel(self, tmp_path):
        clip = load_clip(syn.write_avi(tmp_path / "a.avi", syn.drifting_frames(20), fps=25))
        assert clip.frames.shape == (20, syn.H, syn.W) and clip.frames.dtype == np.uint8
        assert clip.fps == pytest.approx(25, abs=0.01)
        assert abs(float(clip.frames[0].mean()) - float(syn.drifting_frames(1)[0].mean())) < 3
        assert clip.path == tmp_path / "a.avi"

    def test_frame_order_is_preserved(self, tmp_path):
        clip = load_clip(syn.write_avi(tmp_path / "a.avi", syn.numbered_frames(5)))
        values = [int(clip.frames[i].mean()) for i in range(5)]
        assert all(abs(v - (i + 1) * 10) <= 3 for i, v in enumerate(values))  # MJPG is lossy by a level or two

    def test_single_frame_video(self, tmp_path):
        clip = load_clip(syn.write_avi(tmp_path / "one.avi", syn.drifting_frames(1)))
        assert clip.frames.shape == (1, syn.H, syn.W)

    def test_color_video_is_rgb(self, tmp_path):
        clip = load_clip(syn.write_avi(tmp_path / "c.avi", syn.drifting_frames(6), fps=10, color=True))
        assert clip.frames.shape == (6, syn.H, syn.W, 3)
        assert clip.fps == pytest.approx(10, abs=0.01)
        red = clip.frames[0, syn.H // 2, syn.W // 2]
        assert red[0] > 150 and red[2] < red[0]  # the writer's B,G,R = (f, f/2, 200): R must come out large

    def test_uppercase_extension(self, tmp_path):
        clip = load_clip(syn.write_avi(tmp_path / "A.AVI", syn.drifting_frames(4)))
        assert clip.n_frames == 4

    @pytest.mark.parametrize("content", [b"", b"RIFF" + b"\0" * 200])
    def test_unreadable_video_raises(self, tmp_path, content):
        path = tmp_path / "bad.avi"
        path.write_bytes(content)
        with pytest.raises(ClipLoadError):
            load_clip(path)

    @pytest.mark.parametrize("name", ["notes.txt", "echo.dcm", "movie.mp4", "noextension"])
    def test_other_file_types_are_refused_with_a_clear_message(self, tmp_path, name):
        path = tmp_path / name
        path.write_bytes(b"whatever")
        with pytest.raises(ClipLoadError, match="only .avi"):
            load_clip(path)


class TestClipHelpers:
    def test_gray_for_gray_and_color_clips(self, tmp_path):
        gray = load_clip(syn.write_avi(tmp_path / "g.avi", syn.numbered_frames(2)))
        color = load_clip(syn.write_avi(tmp_path / "c.avi", syn.numbered_frames(2), color=True))
        assert gray.gray(1).shape == (syn.H, syn.W)
        assert color.gray(1).shape == (syn.H, syn.W) and color.gray(1).dtype == np.uint8

    @pytest.mark.parametrize("value,expected", [(0, 30.0), (-5, 30.0), (float("nan"), 30.0), (0.1, 1.0), (500, 240.0), (50, 50)])
    def test_clamp_fps(self, value, expected):
        assert _clamp_fps(value) == expected


class TestSiblings:
    def test_natural_order_and_filtering(self, tmp_path):
        for name in ("file10.avi", "file2.avi", "file1.avi", "UPPER.AVI"):
            syn.write_avi(tmp_path / name, syn.drifting_frames(2))
        (tmp_path / "notes.txt").write_text("x")
        (tmp_path / "echo.dcm").write_bytes(b"x")
        (tmp_path / "file1.avi.annotations.json").write_text("{}")
        names = [p.name for p in sibling_clips(tmp_path / "file2.avi")]
        assert names == ["file1.avi", "file2.avi", "file10.avi", "UPPER.AVI"]

    def test_always_includes_the_opened_path(self, tmp_path):
        odd = tmp_path / "weird.bin"
        odd.write_bytes(b"x")
        assert odd in sibling_clips(odd)
