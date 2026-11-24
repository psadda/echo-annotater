import numpy as np
import pytest

from echo_annotator import grow
from echo_annotator.annotations import rasterize
from echo_annotator.geometry import polygon_area
from echo_annotator.grow import grow_polygon

H, W = 200, 240
STRIPE = (80, 100)  # dark rows between two bright bands


@pytest.fixture(scope="module")
def stripe_image():
    rng = np.random.default_rng(0)
    img = np.full((H, W), 110.0)
    img[70:80, 20:220] = 210  # bright band above (pericardium-like)
    img[80:100, 20:220] = 45  # dark stripe (fat-like)
    img[100:110, 20:220] = 210  # bright band below
    speckle = rng.gamma(shape=6.0, scale=1 / 6.0, size=(H, W))  # multiplicative speckle
    return np.clip(img * speckle, 0, 255).astype(np.uint8)


SEED_POLYGON = [(70.0, 87.0), (170.0, 87.0), (170.0, 93.0), (70.0, 93.0)]


def test_grows_to_fill_the_stripe_without_crossing_the_bright_edges(stripe_image):
    grown = grow_polygon(stripe_image, SEED_POLYGON)
    mask = rasterize([grown], H, W) > 0
    rows = np.nonzero(mask.any(axis=1))[0]
    assert polygon_area(grown) > 2 * polygon_area(SEED_POLYGON)
    assert rows.min() >= STRIPE[0] and rows.max() <= STRIPE[1] - 1  # stays between the bright bands
    assert not mask[70:80].any() and not mask[100:110].any()


def test_result_contains_the_original_and_has_even_vertices(stripe_image):
    grown = grow_polygon(stripe_image, SEED_POLYGON)
    assert 16 <= len(grown) <= 32
    seed_mask = rasterize([SEED_POLYGON], H, W) > 0
    grown_mask = rasterize([grown], H, W) > 0
    assert (seed_mask & grown_mask).sum() >= 0.97 * seed_mask.sum()  # resampling may shave a corner pixel or two
    ring = np.asarray(grown)
    gaps = np.linalg.norm(np.roll(ring, -1, 0) - ring, axis=1)
    assert gaps.max() < 1.6 * gaps.mean()


def test_input_is_not_modified(stripe_image):
    seed = list(SEED_POLYGON)
    grow_polygon(stripe_image, seed)
    assert seed == SEED_POLYGON


def test_repeated_growth_eventually_reports_nothing_to_grow(stripe_image):
    polygon = SEED_POLYGON
    for _ in range(12):
        polygon = grow_polygon(stripe_image, polygon)
        if polygon is None:
            break
        assert polygon_area(polygon) > 0
    assert polygon is None


def test_each_press_grows_by_a_bounded_amount(stripe_image):
    first = grow_polygon(stripe_image, SEED_POLYGON)
    # ~10 iterations of ~1 px: the left edge can move at most ~10 px (plus rasterization slack) per press
    assert min(x for x, _ in first) >= SEED_POLYGON[0][0] - 12


def test_polygon_already_filling_the_stripe_cannot_grow(stripe_image):
    full = [(20.0, 80.0), (219.0, 80.0), (219.0, 100.0), (20.0, 100.0)]
    assert grow_polygon(stripe_image, full) is None


def test_empty_polygon_cannot_grow(stripe_image):
    assert grow_polygon(stripe_image, [(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)]) is None


class TestEdgeStopping:
    def test_flat_image_does_not_stop_the_contour(self):
        assert np.allclose(grow._edge_stopping(np.full((30, 40), 0.5, np.float32), 100.0, 2.0), 1.0)

    def test_a_step_edge_is_strongly_stopping_and_flat_sides_are_not(self):
        image = np.zeros((40, 80), np.float32)
        image[:, 40:] = 1.0
        stopping = grow._edge_stopping(image, 100.0, 2.0)
        assert stopping[20, 40] < 0.25  # on the edge: a unit step smoothed by sigma=2 has gradient ~0.2, so 1/sqrt(1+100*0.2)
        assert stopping[20, 5] > 0.99 and stopping[20, 75] > 0.99  # far from it

    def test_stronger_edges_stop_more(self):
        weak, strong = np.zeros((40, 80), np.float32), np.zeros((40, 80), np.float32)
        weak[:, 40:], strong[:, 40:] = 0.2, 1.0
        assert grow._edge_stopping(strong, 100.0, 2.0)[20, 40] < grow._edge_stopping(weak, 100.0, 2.0)[20, 40]

    def test_values_are_in_zero_one_and_borders_are_handled(self):
        stopping = grow._edge_stopping(np.random.default_rng(0).random((30, 30)).astype(np.float32), 100.0, 2.0)
        assert stopping.min() > 0 and stopping.max() <= 1.0 and np.isfinite(stopping).all()


@pytest.fixture(scope="module")
def setup(stripe_image):
    edges = grow._edge_stopping(stripe_image.astype(np.float32) / 255.0, grow.EDGE_ALPHA, grow.EDGE_SIGMA)
    return edges, rasterize([SEED_POLYGON], H, W) > 0


class TestSnake:
    @pytest.mark.parametrize(
        "iterations,pixels,rows,cols",
        [  # regression values, identical to scikit-image's morphological_geodesic_active_contour when this was ported
            (4, 1869, (81, 98), (65, 175)),
            (10, 2389, (80, 99), (58, 182)),
            (30, 3192, (80, 99), (41, 204)),
            (60, 3984, (80, 100), (20, 219)),
        ],
    )
    def test_matches_the_reference_implementation(self, setup, iterations, pixels, rows, cols):
        edges, start = setup
        grown = grow._snake(edges, start, iterations)
        found_rows, found_cols = np.nonzero(grown.any(axis=1))[0], np.nonzero(grown.any(axis=0))[0]
        assert int(grown.sum()) == pixels
        assert (found_rows.min(), found_rows.max()) == rows and (found_cols.min(), found_cols.max()) == cols

    def test_growth_per_iteration_is_bounded(self, setup):
        edges, start = setup
        for iterations in (3, 6, 12):  # about a pixel per iteration; the smoothing steps add a little
            grown = grow._snake(edges, start, iterations)
            cols = np.nonzero(grown.any(axis=0))[0]
            assert cols.min() >= 70 - 2 * iterations and cols.max() <= 169 + 2 * iterations

    def test_bright_bands_stop_growth_however_long_it_runs(self, setup):
        edges, start = setup
        grown = grow._snake(edges, start, 200)
        assert not grown[:80].any() and not grown[101:].any()  # stays between the bright bands (rows 80-99, one pixel of slack)

    def test_more_iterations_never_grow_less(self, setup):
        edges, start = setup
        sizes = [int(grow._snake(edges, start, n).sum()) for n in (2, 4, 8, 16)]
        assert sizes == sorted(sizes)

    def test_input_mask_is_not_modified_and_output_is_boolean(self, setup):
        edges, start = setup
        before = start.copy()
        out = grow._snake(edges, start, 5)
        assert (start == before).all() and out.dtype == bool and out.shape == start.shape

    def test_without_edge_variation_nothing_inflates(self):
        # Like scikit-image: the balloon acts only where the edge-stopping value exceeds its 40th percentile,
        # which a perfectly flat image never does (curvature smoothing may even erode a small mask; grow_polygon
        # always keeps the original). Real echo frames are never perfectly flat.
        start = np.zeros((40, 40), bool)
        start[15:25, 15:25] = True
        assert not (grow._snake(np.ones((40, 40), np.float32), start, 8) & ~start).any()


class TestLargestOutline:
    def test_rectangle_outline_is_exactly_its_pixel_edges(self):
        mask = np.zeros((30, 40), bool)
        mask[5:15, 8:30] = True
        outline = grow._largest_outline(mask)
        assert sorted(outline) == sorted([(8.0, 5.0), (30.0, 5.0), (30.0, 15.0), (8.0, 15.0)])
        assert polygon_area(outline) == 10 * 22

    def test_outline_area_matches_pixel_count_for_a_blob_within_simplification(self):
        yy, xx = np.mgrid[:60, :60]
        mask = (xx - 30) ** 2 + (yy - 28) ** 2 < 18**2
        outline = grow._largest_outline(mask)
        assert polygon_area(outline) == pytest.approx(mask.sum(), rel=0.03)

    def test_holes_are_ignored_and_the_largest_blob_wins(self):
        mask = np.zeros((40, 60), bool)
        mask[5:20, 8:30] = True
        mask[10:14, 12:16] = False  # a hole
        mask[25:35, 40:55] = True  # a smaller second blob
        outline = grow._largest_outline(mask)
        assert polygon_area(outline) == 15 * 22
        assert max(x for x, _ in outline) <= 30

    def test_blob_touching_the_image_border_is_closed(self):
        mask = np.zeros((20, 20), bool)
        mask[:, :8] = True
        outline = grow._largest_outline(mask)
        assert polygon_area(outline) == 20 * 8
        assert min(x for x, _ in outline) == 0 and min(y for _, y in outline) == 0

    def test_empty_mask_has_no_outline(self):
        assert grow._largest_outline(np.zeros((10, 10), bool)) is None

    def test_small_block(self):
        mask = np.zeros((10, 10), bool)
        mask[3:6, 4:7] = True
        assert polygon_area(grow._largest_outline(mask)) == 9

    def test_a_single_pixel_does_not_crash(self):
        mask = np.zeros((10, 10), bool)
        mask[4, 6] = True
        outline = grow._largest_outline(mask)  # may simplify away entirely, but must not raise
        assert outline is None or len(outline) >= 3
