import cv2
import numpy as np
import pytest

import synthetic as syn
from synthetic import curve_distance
from echo_annotator import tracking
from echo_annotator.geometry import polygon_area, resample_polygon

N = 12
DRIFT = np.array([1.0, 0.5])  # px per frame, see synthetic.drifting_frames
POLY = [(50.0, 40.0), (100.0, 38.0), (110.0, 70.0), (70.0, 85.0), (45.0, 65.0)]


@pytest.fixture(scope="module")
def frames():
    return syn.drifting_frames(N)


class TestPropagate:
    @pytest.mark.parametrize("preserve", [False, True])
    @pytest.mark.parametrize("source", [0, 5, N - 1])
    def test_follows_rigid_drift_in_both_directions(self, frames, source, preserve):
        start = [(x + DRIFT[0] * source, y + DRIFT[1] * source) for x, y in POLY]
        out = tracking.propagate_polygon(lambda i: frames[i], N, source, start, preserve_curvature=preserve)
        assert sorted(out) == [t for t in range(N) if t != source]  # every other frame, never the source
        for t, polygon in out.items():
            truth = [(x + DRIFT[0] * t, y + DRIFT[1] * t) for x, y in POLY]
            assert curve_distance(polygon, truth) < 1.0

    def test_tracked_polygons_have_a_constant_even_vertex_set(self, frames):
        out = tracking.propagate_polygon(lambda i: frames[i], N, 0, POLY)
        counts = {len(p) for p in out.values()}
        assert len(counts) == 1 and 16 <= counts.pop() <= 32
        ring = np.asarray(out[N - 1])
        gaps = np.linalg.norm(np.roll(ring, -1, 0) - ring, axis=1)
        assert gaps.max() < 1.6 * gaps.mean()

    def test_tracking_is_lost_on_featureless_frames(self):
        flat = [np.zeros((syn.H, syn.W), np.uint8)] * 5
        assert tracking.propagate_polygon(lambda i: flat[i], 5, 2, POLY) == {}

    def test_single_frame_clip_has_nothing_to_track(self):
        frame = syn.drifting_frames(1)[0]
        assert tracking.propagate_polygon(lambda i: frame, 1, 0, POLY) == {}

    def test_polygons_stay_inside_the_image(self, frames):
        edge = [(2.0, 2.0), (30.0, 3.0), (20.0, 30.0)]
        out = tracking.propagate_polygon(lambda i: frames[i], N, 0, edge)
        for polygon in out.values():
            pts = np.asarray(polygon)
            assert pts.min() >= 0 and pts[:, 0].max() <= syn.W and pts[:, 1].max() <= syn.H


RING = np.asarray(resample_polygon([(50 + 30 * np.cos(a), 50 + 30 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 400, endpoint=False)], 24)) - 0.5
CONFIDENT = np.ones(24)


def roughness(points):
    return float(np.linalg.norm(np.roll(points, -1, 0) - 2 * points + np.roll(points, 1, 0), axis=1).mean())


class TestRegularizedSolve:
    @pytest.mark.parametrize("reference", [None, RING])
    def test_translation_is_free_and_does_not_shrink(self, reference):
        shifted = RING + np.array([3.0, -2.0])
        out = tracking._regularized_positions(RING, shifted, CONFIDENT, reference)
        assert np.allclose(out, shifted, atol=1e-6)

    @pytest.mark.parametrize("reference", [None, RING])
    def test_uniform_scaling_is_followed_without_lag(self, reference):
        scale = 1.3
        scaled = (RING - RING.mean(0)) * scale + RING.mean(0)
        out = tracking._regularized_positions(RING, scaled, CONFIDENT, reference)
        ratio = polygon_area([tuple(p) for p in out]) / polygon_area([tuple(p) for p in RING])
        assert ratio == pytest.approx(scale**2, abs=0.03)

    @pytest.mark.parametrize("reference", [None, RING])
    def test_rotation_is_followed(self, reference):
        angle = np.deg2rad(20)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = (RING - RING.mean(0)) @ rot.T + RING.mean(0)
        out = tracking._regularized_positions(RING, rotated, CONFIDENT, reference)
        assert np.abs(out - rotated).max() < 0.2

    def test_zero_confidence_vertex_is_interpolated_from_neighbours(self):
        target = RING + np.array([1.0, 0.5])
        wild = target.copy()
        wild[7] += np.array([9.0, -9.0])
        weights = CONFIDENT.copy()
        weights[7] = 0.0
        out = tracking._regularized_positions(RING, wild, weights, None)
        assert np.linalg.norm(out[7] - target[7]) < 1.0

    def test_confident_but_wild_track_is_largely_rejected(self):
        target = RING + np.array([1.0, 0.5])
        wild = target.copy()
        wild[7] += np.array([9.0, -9.0])
        out = tracking._regularized_positions(RING, wild, CONFIDENT, None)
        assert np.linalg.norm(out[7] - target[7]) < 0.5 * np.linalg.norm(wild[7] - target[7])

    def test_jitter_is_removed(self):
        noisy = RING + np.array([2.0, 1.0]) + np.random.default_rng(0).normal(0, 1.2, RING.shape)
        out = tracking._regularized_positions(RING, noisy, CONFIDENT, None)
        base = roughness(RING)  # a 24-gon is itself "rough"; only the excess is jitter
        assert roughness(out) - base < 0.25 * (roughness(noisy) - base)

    def test_curvature_term_smooths_at_least_as_much_without_flattening(self):
        noisy = RING + np.array([2.0, 1.0]) + np.random.default_rng(0).normal(0, 1.2, RING.shape)
        plain = tracking._regularized_positions(RING, noisy, CONFIDENT, None)
        kept = tracking._regularized_positions(RING, noisy, CONFIDENT, RING)
        assert roughness(kept) <= roughness(plain) + 1e-9
        assert roughness(kept) > 0.9 * roughness(RING)

    def test_an_existing_bend_survives_when_it_is_the_reference(self):
        notched = RING.copy()
        notched[5:8] = (notched[5:8] - RING.mean(0)) * 0.8 + RING.mean(0)
        target = notched + np.array([2.0, 0.0])
        out = tracking._regularized_positions(notched, target, CONFIDENT, notched)
        assert np.allclose(out, target, atol=1e-6)

    def test_all_zero_confidence_falls_back_without_crashing(self):
        out = tracking._regularized_positions(RING, RING + 1.0, np.zeros(24), None)
        assert np.isfinite(out).all()


class TestSimilarityFit:
    def test_recovers_scale_rotation_and_shift(self):
        angle, scale = np.deg2rad(25), 1.4
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        target = scale * (RING - RING.mean(0)) @ rot.T + RING.mean(0) + np.array([5.0, -3.0])
        s, r, source_mean, target_mean = tracking._fit_similarity(RING, target, CONFIDENT)
        assert s == pytest.approx(scale)
        assert np.allclose(r, rot, atol=1e-9)
        assert np.allclose(s * (RING - source_mean) @ r.T + target_mean, target)

    def test_few_points_fall_back_to_a_pure_shift(self):
        weights = np.zeros(24)
        weights[:2] = 1.0
        s, r, _, _ = tracking._fit_similarity(RING, RING + 4.0, weights)
        assert s == 1.0 and np.allclose(r, np.eye(2))


# ---- the jagged-outpouching problem: dense contour, deforming speckle, fresh noise every frame ----

HH, WW, FRAMES = 200, 280, 60


def backward_map(t):
    """Smooth non-rigid map from frame-t coordinates to frame-0 coordinates."""
    s = t / (FRAMES - 1)

    def g(p):
        p = np.asarray(p, float)
        x, y = p[..., 0], p[..., 1]
        dx = -14 * s + 0.05 * s * (y - 100) + 3.0 * s * np.sin(2 * np.pi * y / 110 + 1.0)
        dy = 6 * s * np.sin(s * 3) + 0.04 * s * (x - 140) + 2.5 * s * np.sin(2 * np.pi * x / 130)
        return np.stack([x + dx, y + dy], axis=-1)

    return g


def forward_points(points, t):
    g, x = backward_map(t), np.asarray(points, float).copy()
    for _ in range(30):  # invert g_t by fixed-point iteration
        x = x - (g(x) - points)
    return x


def banana(n=120):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 70 + 8 * np.cos(angles)
    return np.stack([140 + r * np.cos(angles), 120 + 0.45 * r * np.sin(angles) - 25 * np.cos(angles) ** 2], axis=-1)


def noisy_deforming_frames(seed=0, noise=45):
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.gamma(4.0, 1 / 4.0, (HH, WW)).astype(np.float32), (0, 0), 1.2)
    tex = (tex - tex.min()) / (tex.max() - tex.min()) * 100 + 70
    ys, xs = np.mgrid[0:HH, 0:WW].astype(np.float32)
    frames = []
    for t in range(FRAMES):
        g = backward_map(t)(np.stack([xs, ys], axis=-1))
        warped = cv2.remap(tex, g[..., 0].astype(np.float32), g[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        warped = warped * rng.gamma(40.0, 1 / 40.0, warped.shape).astype(np.float32) + rng.normal(0, noise, warped.shape)
        frames.append(np.clip(warped, 0, 255).astype(np.uint8))
    return frames


def bump_amplitude(polygon, count=96, harmonics=6):
    """Largest distance between the contour and a version of itself keeping only the lowest Fourier harmonics."""
    p = np.asarray(resample_polygon([tuple(q) for q in polygon], count))
    z = p[:, 0] + 1j * p[:, 1]
    f = np.fft.fft(z)
    keep = np.zeros(count, bool)
    keep[: harmonics + 1] = True
    keep[-harmonics:] = True
    return float(np.abs(z - np.fft.ifft(np.where(keep, f, 0))).max())


def naive_propagate(frames, polygon):
    """Each vertex tracked independently by LK, as the first version of the tracker did."""
    out, current = {}, np.asarray(polygon, np.float32) - 0.5
    for t in range(1, len(frames)):
        moved, ok, _ = cv2.calcOpticalFlowPyrLK(frames[t - 1], frames[t], current.reshape(-1, 1, 2), None, winSize=(31, 31), maxLevel=3)
        moved = moved.reshape(-1, 2)
        current = np.where(ok.reshape(-1, 1) == 1, moved, current)
        out[t] = [tuple(p + 0.5) for p in current]
    return out


@pytest.fixture(scope="module")
def runs():
    frames = noisy_deforming_frames()
    dense = [tuple(p) for p in banana()]
    gray = lambda i: frames[i]
    return {
        "frames": frames,
        "naive": naive_propagate(frames, dense),
        "smooth": tracking.propagate_polygon(gray, FRAMES, 0, dense),
        "curvature": tracking.propagate_polygon(gray, FRAMES, 0, dense, preserve_curvature=True),
        "true_bump": float(np.mean([bump_amplitude(forward_points(banana(), t)) for t in range(FRAMES)])),
    }


class TestJaggedContourRegression:
    @staticmethod
    def mean_bump(out):
        return float(np.mean([bump_amplitude(p) for p in out.values()]))

    @staticmethod
    def mean_error(out):
        return float(np.mean([curve_distance(p, [tuple(q) for q in forward_points(banana(), t)]) for t, p in out.items()]))

    def test_setup_reproduces_the_outpouchings(self, runs):
        assert self.mean_bump(runs["naive"]) > 2.5 * runs["true_bump"]

    @pytest.mark.parametrize("variant", ["smooth", "curvature"])
    def test_regularized_tracker_keeps_bumps_near_the_true_shapes(self, runs, variant):
        assert len(runs[variant]) == FRAMES - 1
        assert self.mean_bump(runs[variant]) < 1.3 * runs["true_bump"]

    @pytest.mark.parametrize("variant", ["smooth", "curvature"])
    def test_regularized_tracker_is_also_more_accurate(self, runs, variant):
        assert self.mean_error(runs[variant]) < self.mean_error(runs["naive"])
