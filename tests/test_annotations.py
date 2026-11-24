import json

import numpy as np
import pytest

from echo_annotator import annotations as ann
from echo_annotator.annotations import AnnotationError, AnnotationSet, rasterize, sidecar_path

POLY = [(10.0, 10.0), (50.0, 10.0), (50.0, 30.0), (10.0, 30.0)]


@pytest.fixture
def aset(tmp_path):
    return AnnotationSet(tmp_path / "clip.avi", n_frames=6, height=60, width=80)


class TestRasterize:
    def test_aligned_rectangle_is_exact(self):
        mask = rasterize([[(2, 2), (6, 2), (6, 5), (2, 5)]], 10, 12)
        assert mask.sum() // 255 == 12 and mask[2:5, 2:6].all()
        assert set(np.unique(mask)) <= {0, 255}

    def test_triangle_area(self):
        mask = rasterize([[(0, 0), (10, 0), (0, 10)]], 10, 10)
        assert abs(int(mask.sum() // 255) - 50) <= 6

    def test_union_of_overlapping_polygons(self):
        a, b = [(0, 0), (6, 0), (6, 6), (0, 6)], [(3, 3), (9, 3), (9, 9), (3, 9)]
        mask = rasterize([a, b], 10, 10)
        assert mask.sum() // 255 == 36 + 36 - 9


class TestRasterizeGeometry:
    def test_pixels_are_inside_when_their_centre_is(self):
        # a strip from x=1.0 to 3.4: centres at 1.5 and 2.5 are inside, 3.5 is just outside
        mask = rasterize([[(1.0, 1.0), (3.4, 1.0), (3.4, 3.0), (1.0, 3.0)]], 5, 6)
        assert mask[1:3, 1:3].all() and not mask[1:3, 3].any() and mask.sum() // 255 == 4

    def test_area_is_unbiased_on_random_convex_polygons(self):
        import cv2

        from echo_annotator.geometry import polygon_area

        rng = np.random.default_rng(1)
        errors = []
        for _ in range(150):
            hull = cv2.convexHull(rng.uniform([3, 3], [97, 77], (7, 2)).astype(np.float32)).reshape(-1, 2)
            polygon = [tuple(map(float, p)) for p in hull]
            if len(polygon) >= 3 and polygon_area(polygon) > 100:
                errors.append((rasterize([polygon], 80, 100) > 0).sum() / polygon_area(polygon) - 1)
        assert abs(np.mean(errors)) < 0.005 and np.max(np.abs(errors)) < 0.06

    def test_polygon_partly_outside_the_image_is_clipped(self):
        mask = rasterize([[(-5.0, -5.0), (4.0, -5.0), (4.0, 4.0), (-5.0, 4.0)]], 10, 10)
        assert mask[:4, :4].all() and mask.sum() // 255 == 16

    def test_polygon_entirely_outside_is_empty(self):
        assert not rasterize([[(50.0, 50.0), (60.0, 50.0), (55.0, 60.0)]], 10, 10).any()

    def test_degenerate_polygons_are_empty(self):
        assert not rasterize([[(3.0, 3.0), (3.0, 3.0), (3.0, 3.0)]], 10, 10).any()
        assert not rasterize([[(1.0, 5.0), (8.0, 5.0), (4.0, 5.0)]], 10, 10).any()  # zero height

    def test_vertex_exactly_on_a_pixel_row_centre_is_not_double_counted(self):
        # the shared vertex (5, 2.5) sits exactly on a row centre line
        mask = rasterize([[(0.0, 0.0), (5.0, 2.5), (0.0, 5.0)]], 6, 6)
        assert 0 < mask.sum() // 255 <= 13  # a triangle of area 12.5, never a stripe or a hole

    def test_concave_polygon_uses_even_odd_fill(self):
        notch = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.0, 4.0), (0.0, 10.0)]
        mask = rasterize([notch], 12, 12) > 0
        assert mask[2, 2] and mask[2, 8] and not mask[8, 5]  # below the inner vertex is the notch, outside


class TestEditing:
    def test_add_and_undo_redo(self, aset):
        aset.add(2, POLY)
        assert aset.annotated_frames() == [2]
        assert aset.undo() == 2 and aset.annotated_frames() == []
        assert aset.redo() == 2 and aset.on_frame(2) == [POLY]

    def test_nothing_to_undo_or_redo(self, aset):
        assert aset.undo() is None and aset.redo() is None

    def test_new_edit_discards_redo(self, aset):
        aset.add(0, POLY)
        aset.undo()
        aset.add(1, POLY)
        assert aset.redo() is None

    def test_clear_and_delete_polygon(self, aset):
        aset.add(0, POLY)
        aset.add(0, POLY[::-1])
        aset.delete_polygon(0, 0)
        assert aset.on_frame(0) == [POLY[::-1]]
        assert aset.clear(0) and aset.annotated_frames() == []
        assert not aset.clear(0)  # nothing left to clear

    def test_insert_move_and_delete_vertex(self, aset):
        aset.add(0, list(POLY))
        aset.insert_vertex(0, 0, 1, (30.0, 10.0))
        assert aset.on_frame(0)[0][1] == (30.0, 10.0)
        aset.checkpoint(0)
        aset.move_vertex(0, 0, 1, (30.0, 5.0))
        assert aset.on_frame(0)[0][1] == (30.0, 5.0)
        assert aset.delete_vertex(0, 0, 1)
        assert len(aset.on_frame(0)[0]) == 4

    def test_delete_vertex_refused_below_three(self, aset):
        aset.add(0, [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)])
        before = len(aset._history)
        assert not aset.delete_vertex(0, 0, 0)
        assert len(aset.on_frame(0)[0]) == 3 and len(aset._history) == before

    def test_replace_polygon_is_one_undo_step(self, aset):
        aset.add(0, POLY)
        aset.replace_polygon(0, 0, POLY[::-1])
        assert aset.undo() == 0 and aset.on_frame(0) == [POLY]

    def test_set_frames_is_one_multi_frame_undo_step(self, aset):
        aset.add(1, POLY)
        aset.add(3, [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)])
        tracked = {0: POLY, 2: POLY, 3: POLY}  # frame 3 already had something
        aset.set_frames(1, tracked)
        assert aset.annotated_frames() == [0, 1, 2, 3]
        assert aset.on_frame(3) == [POLY]
        assert aset.undo() == 1  # the frame to show is the source frame
        assert aset.annotated_frames() == [1, 3]
        assert aset.on_frame(3) == [[(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)]]
        assert aset.redo() == 1 and aset.annotated_frames() == [0, 1, 2, 3]

    def test_history_is_capped(self, aset):
        for _ in range(ann.MAX_HISTORY + 25):
            aset.add(0, POLY)
        assert len(aset._history) == ann.MAX_HISTORY

    def test_undo_snapshots_are_not_aliased_to_live_data(self, aset):
        aset.add(0, list(POLY))
        aset.checkpoint(0)
        aset.move_vertex(0, 0, 0, (0.0, 0.0))
        aset.undo()
        assert aset.on_frame(0)[0][0] == (10.0, 10.0)


class TestPersistence:
    def test_roundtrip(self, aset):
        aset.add(0, POLY)
        aset.add(4, [(1.5, 2.5), (9.0, 2.5), (5.0, 8.0)])
        aset.save()
        loaded = AnnotationSet.load(aset.path, 6, 60, 80)
        assert loaded.polygons == aset.polygons
        assert loaded._history == [] and loaded._redo == []  # undo history is never persisted

    def test_json_contents_and_no_masks(self, aset):
        aset.add(2, POLY)
        aset.save()
        path = sidecar_path(aset.path)
        assert path.name == "clip.avi.annotations.json"
        data = json.loads(path.read_text())
        assert data["label"] == "pericardial_fat" and data["n_frames"] == 6 and data["source"] == "clip.avi"
        assert data["image_size"] == {"width": 80, "height": 60}
        assert list(data["frames"]) == ["2"]
        assert not list(path.parent.glob("*.masks")) and not list(path.parent.glob("*.png"))
        assert "undo" not in path.read_text() and "redo" not in path.read_text()

    def test_save_leaves_no_temp_file(self, aset):
        aset.add(0, POLY)
        aset.save()
        assert not list(aset.path.parent.glob("*.tmp"))

    def test_load_without_sidecar_is_empty(self, tmp_path):
        loaded = AnnotationSet.load(tmp_path / "x.avi", 3, 10, 10)
        assert loaded.polygons == {}

    @pytest.mark.parametrize("n_frames,height,width", [(7, 60, 80), (6, 61, 80), (6, 60, 81)])
    def test_mismatched_sidecar_is_refused(self, aset, n_frames, height, width):
        aset.add(0, POLY)
        aset.save()
        with pytest.raises(AnnotationError):
            AnnotationSet.load(aset.path, n_frames, height, width)

    def test_out_of_range_frame_is_refused(self, aset):
        aset.add(0, POLY)
        aset.save()
        path = sidecar_path(aset.path)
        data = json.loads(path.read_text())
        data["frames"] = {"99": data["frames"]["0"]}
        path.write_text(json.dumps(data))
        with pytest.raises(AnnotationError):
            AnnotationSet.load(aset.path, 6, 60, 80)

    @pytest.mark.parametrize("text", ["not json", "{}", '{"image_size": 3}'])
    def test_corrupt_sidecar_is_refused(self, aset, text):
        sidecar_path(aset.path).write_text(text)
        with pytest.raises(AnnotationError):
            AnnotationSet.load(aset.path, 6, 60, 80)

    def test_matches_disk_detects_outside_changes(self, aset):
        aset.add(0, POLY)
        aset.save()
        assert aset.matches_disk()
        path = sidecar_path(aset.path)
        path.write_text(path.read_text() + "\n")
        assert not aset.matches_disk()
        path.unlink()
        assert not aset.matches_disk()
