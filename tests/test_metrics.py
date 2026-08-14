"""Tests for the movement metrics."""

import numpy as np

from larvatracker.config import AnalysisConfig
from larvatracker.metrics import (
    burst_frequency,
    detect_bursts,
    displacement_from_start,
    movement_onset,
    moving_average_nan,
    step_displacement,
    time_bin_metrics,
)


def test_step_displacement_keeps_length_and_starts_with_nan():
    xy = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])

    step = step_displacement(xy)

    assert len(step) == 3
    assert np.isnan(step[0])
    assert step[1] == 5.0
    assert step[2] == 0.0


def test_displacement_from_start_ignores_leading_nan():
    xy = np.array([[np.nan, np.nan], [0.0, 0.0], [6.0, 8.0]])

    out = displacement_from_start(xy)

    assert np.isnan(out[0])
    assert out[1] == 0.0
    assert out[2] == 10.0


def test_moving_average_tolerates_nan():
    values = np.array([1.0, np.nan, 1.0, 1.0, 1.0])

    assert np.isfinite(moving_average_nan(values, 3)).all()


def test_detect_bursts_finds_the_planted_peaks():
    n = 600
    signal = np.zeros(n)
    peaks_at = [100, 250, 400]

    for centre in peaks_at:
        signal[centre - 5 : centre + 5] += np.hanning(10) * 10

    found = detect_bursts(signal, prominence=1.2, distance=10)

    assert len(found) == len(peaks_at)
    assert all(min(abs(found - p)) <= 2 for p in peaks_at)


def test_detect_bursts_on_a_short_trace_returns_empty():
    assert detect_bursts(np.ones(5)).size == 0


def test_burst_frequency_matches_the_definition():
    # 30 bursts over 900 frames at 30 fps = 30 s -> 1 Hz
    assert burst_frequency(30, 900, 30.0) == 1.0


def test_movement_onset_is_nan_when_never_exceeded():
    assert np.isnan(movement_onset(np.zeros(100), threshold=4.0))
    assert movement_onset(np.array([0.0, 0.0, 9.0, 0.0]), threshold=4.0) == 2.0


def test_time_bins_drop_the_trailing_partial_bin():
    config = AnalysisConfig(bin_size_frames=100, fps=10.0)
    raw = np.ones(250)

    bins = time_bin_metrics(raw, raw, config)

    assert [b["bin"] for b in bins] == [1, 2]
    assert bins[0]["time_sec"] == 5.0
