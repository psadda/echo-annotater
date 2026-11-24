import numpy as np
import pytest

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
