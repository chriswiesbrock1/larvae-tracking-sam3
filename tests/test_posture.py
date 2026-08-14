"""Tests for the body-axis sorting, which the whole analysis depends on.

Run with ``pytest`` from the repository root.
"""

import numpy as np

from larvatracker.posture import interpolate_gaps, sort_points_along_axis, sort_series


def make_larva(n_frames: int = 600, n_points: int = 5) -> np.ndarray:
    """A synthetic larva that translates, rotates a full turn and bends."""
    truth = np.zeros((n_frames, n_points, 2))

    for t in range(n_frames):
        heading = 2 * np.pi * t / n_frames
        cx = 50 + 10 * np.cos(t / 30)
        cy = 50 + 10 * np.sin(t / 30)

        for k in range(n_points):
            offset = (k - n_points // 2) * 6.0
            bend = 1.2 * np.sin(t / 7.0 + k)
            truth[t, k, 0] = cx + offset * np.cos(heading) - bend * np.sin(heading)
            truth[t, k, 1] = cy + offset * np.sin(heading) + bend * np.cos(heading)

    return truth


def test_sorting_recovers_a_consistent_order():
    """Randomly permuted labels must be restored to one consistent ordering.

    Which end is index 0 is arbitrary — the axis direction has no intrinsic
    sign — but it must be the *same* end in every frame.
    """
    rng = np.random.default_rng(7)
    truth = make_larva()

    observed = np.stack([truth[t][rng.permutation(truth.shape[1])] for t in range(len(truth))])
    sorted_xy = sort_series(observed)

    forward = sum(np.allclose(sorted_xy[t], truth[t], atol=1e-6) for t in range(len(truth)))
    reversed_ = sum(np.allclose(sorted_xy[t], truth[t][::-1], atol=1e-6) for t in range(len(truth)))

    assert max(forward, reversed_) == len(truth)


def test_sorting_removes_spurious_displacement():
    """Label swaps inflate per-keypoint displacement; sorting must remove that."""
    rng = np.random.default_rng(7)
    truth = make_larva()

    observed = np.stack([truth[t][rng.permutation(truth.shape[1])] for t in range(len(truth))])
    sorted_xy = sort_series(observed)

    def mean_step(xy):
        return np.nanmean(np.hypot(*np.diff(xy[:, 0, :], axis=0).T))

    assert mean_step(sorted_xy) < mean_step(observed) / 3


def test_axis_direction_is_carried_over():
    """A reversed input must not flip the assignment when a previous axis exists."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])

    result, _ = sort_points_along_axis(points[::-1], prev_axis=np.array([1.0, 0.0]))

    assert result is not None
    assert result[0, 0] == 0.0
    assert result[-1, 0] == 4.0


def test_too_few_valid_points_returns_none():
    """Fewer than three finite keypoints cannot define an orientation."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])

    result, axis = sort_points_along_axis(points, prev_axis=None)

    assert result is None
    assert axis is None


def test_interpolate_gaps_keeps_long_runs_as_nan():
    """Short dropouts are bridged, long tracking failures are not invented."""
    values = np.arange(20.0)
    values[5:7] = np.nan       # short gap  -> filled
    values[10:18] = np.nan     # long gap   -> stays NaN

    filled = interpolate_gaps(values, max_gap=3)

    assert np.isfinite(filled[5:7]).all()
    assert np.isnan(filled[10:18]).all()
