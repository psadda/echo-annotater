import numpy as np
import pytest

from echo_annotator.geometry import (
    nearest_edge,
    nearest_vertex,
    polygon_area,
    polygon_at,
    polygon_perimeter,
    resample_polygon,
    target_vertex_count,
)

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_area_and_perimeter():
    assert polygon_area(SQUARE) == 100
    assert polygon_area(SQUARE[::-1]) == 100  # unsigned
    assert polygon_perimeter(SQUARE) == 40


def test_nearest_vertex_respects_radius():
    assert nearest_vertex([SQUARE], 9.5, 0.4, 1) == (0, 1)
    assert nearest_vertex([SQUARE], 5, 5, 1) is None


def test_nearest_vertex_picks_closest_across_polygons():
    other = [(9.0, 0.0), (19.0, 0.0), (19.0, 10.0)]
    assert nearest_vertex([SQUARE, other], 9.1, 0.0, 2) == (1, 0)


def test_nearest_edge_projects_onto_segment():
    polygon, edge, point = nearest_edge([SQUARE], 5, 0.5, 1)
    assert (polygon, edge) == (0, 0)
    assert point == pytest.approx((5.0, 0.0))


def test_nearest_edge_includes_closing_edge():
    assert nearest_edge([SQUARE], 0.2, 5, 1)[1] == 3


def test_nearest_edge_none_when_far():
    assert nearest_edge([SQUARE], 5, 5, 1) is None


def test_polygon_at_inside_outside_and_topmost():
    big = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
    assert polygon_at([SQUARE], 5, 5) == 0
    assert polygon_at([SQUARE], 15, 5) is None
    assert polygon_at([big, SQUARE], 5, 5) == 1  # later polygons are on top


def test_polygon_at_concave_notch():
    concave = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.0, 4.0), (0.0, 10.0)]
    assert polygon_at([concave], 2, 2) == 0
    assert polygon_at([concave], 5, 8) is None  # inside the notch


class TestResample:
    def test_count_first_vertex_and_even_spacing(self):
        r = resample_polygon(SQUARE, 20)
        gaps = np.linalg.norm(np.roll(np.asarray(r), -1, 0) - np.asarray(r), axis=1)
        assert len(r) == 20 and r[0] == (0.0, 0.0)
        assert np.allclose(gaps[:-1], 2.0)  # 40 px perimeter / 20

    def test_keeps_area_of_a_dense_circle(self):
        circle = [(50 + 30 * np.cos(a), 50 + 30 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 400, endpoint=False)]
        r = resample_polygon(circle, 24)
        assert polygon_area(r) == pytest.approx(np.pi * 900, rel=0.02)

    def test_preserves_direction_and_start(self):
        circle = [(50 + 30 * np.cos(a), 50 + 30 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 400, endpoint=False)]
        r = resample_polygon(circle, 24)
        assert r[0] == circle[0]
        radial = np.asarray(circle[0]) - 50
        tangent = np.array([-radial[1], radial[0]])  # counter-clockwise
        assert np.dot(np.asarray(r[1]) - np.asarray(r[0]), tangent) > 0

    def test_degenerate_polygon(self):
        assert resample_polygon([(3.0, 4.0)] * 3, 5) == [(3.0, 4.0)] * 5


def test_target_vertex_count_clamps():
    tiny = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    huge = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)]
    assert target_vertex_count(tiny) == 16
    assert target_vertex_count(huge) == 32
    assert 16 <= target_vertex_count(SQUARE) <= 32
