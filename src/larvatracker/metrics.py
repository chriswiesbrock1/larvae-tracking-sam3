"""Movement metrics derived from sorted keypoint trajectories.

The central quantity is the frame-to-frame displacement of a keypoint in
pixels. Smoothed with a centred rolling mean it forms a burst-like signal:
larvae in droplets alternate between rest and short bouts of locomotion. Peaks
of that signal are counted as bursts, and their rate is reported in Hz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from larvatracker.config import AnalysisConfig


def moving_average_nan(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean that tolerates NaN."""
    if window is None or window <= 1:
        return np.asarray(values, dtype=float)

    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=max(1, window // 2))
        .mean()
        .to_numpy()
    )


def step_displacement(xy: np.ndarray) -> np.ndarray:
    """Euclidean shift between consecutive frames, in px/frame.

    The first element is NaN so the result keeps the length of the recording.
    """
    xy = np.asarray(xy, dtype=float)

    out = np.full(len(xy), np.nan)
    out[1:] = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    return out


def displacement_from_start(xy: np.ndarray) -> np.ndarray:
    """Distance to the first valid position, in pixels."""
    xy = np.asarray(xy, dtype=float)
    x, y = xy[:, 0], xy[:, 1]

    ok = np.isfinite(x) & np.isfinite(y)
    out = np.full(len(x), np.nan)

    if ok.any():
        first = np.where(ok)[0][0]
        out[ok] = np.hypot(x[ok] - x[first], y[ok] - y[first])

    return out


def detect_bursts(
    smoothed: np.ndarray,
    prominence: float = 1.2,
    distance: int = 10,
) -> np.ndarray:
    """Locate movement bursts in a smoothed displacement trace.

    NaN samples are removed before peak detection and the resulting indices are
    mapped back onto the original frame axis, so the returned indices can be
    used directly to index the input array.

    Returns
    -------
    np.ndarray
        Frame indices of the detected peaks (empty if the trace is too short).
    """
    valid = np.where(np.isfinite(smoothed))[0]
    if valid.size < 20:
        return np.array([], dtype=int)

    peaks, _ = find_peaks(smoothed[valid], prominence=prominence, distance=distance)
    return valid[peaks]


def burst_frequency(n_bursts: int, n_frames: int, fps: float) -> float:
    """Burst rate in Hz."""
    if n_frames <= 0:
        return float("nan")
    return n_bursts / (n_frames / fps)


def movement_onset(smoothed: np.ndarray, threshold: float) -> float:
    """First frame whose smoothed displacement exceeds ``threshold``.

    Returns NaN when the threshold is never crossed, i.e. the larva never
    showed heavy movement during the recording.
    """
    indices = np.where(smoothed > threshold)[0]
    return float(indices[0]) if indices.size else float("nan")


def bodypart_metrics(
    sorted_xy: np.ndarray,
    index: int,
    config: AnalysisConfig,
) -> dict:
    """Compute the standard metric set for one keypoint.

    Returns a dict with the raw and smoothed displacement traces, the detected
    burst frames, the burst rate in Hz, the onset latency in seconds and the
    mean velocity. The traces are included so callers can plot them without
    recomputing.
    """
    raw = step_displacement(sorted_xy[:, index, :])
    smoothed = moving_average_nan(raw, config.smoothing_window)

    peaks = detect_bursts(smoothed, config.peak_prominence, config.peak_distance)
    onset_frame = movement_onset(smoothed, config.onset_threshold)

    return {
        "raw_step": raw,
        "ma_step": smoothed,
        "peaks": peaks,
        "burst_count": int(len(peaks)),
        "freq_hz": burst_frequency(len(peaks), len(raw), config.fps),
        "onset_frame": onset_frame,
        "onset_sec": onset_frame / config.fps if np.isfinite(onset_frame) else np.nan,
        "mean_vel": float(np.nanmean(raw)) if np.isfinite(raw).any() else np.nan,
    }


def time_bin_metrics(
    raw: np.ndarray,
    smoothed: np.ndarray,
    config: AnalysisConfig,
) -> list[dict]:
    """Split a trace into fixed-length bins and score each bin separately.

    Bins let a drug effect that develops over the recording show up as a change
    across bins rather than being averaged away. A trailing partial bin is
    discarded so every bin covers the same duration.

    Returns
    -------
    list of dict
        One entry per bin with ``bin`` (1-based), ``time_sec`` (bin centre),
        ``freq_hz`` and ``mean_vel``.
    """
    size = config.bin_size_frames
    n_bins = len(raw) // size

    results = []
    for b in range(n_bins):
        start = b * size
        stop = start + size

        bin_smoothed = smoothed[start:stop]
        bin_raw = raw[start:stop]

        peaks = detect_bursts(bin_smoothed, config.peak_prominence, config.peak_distance)

        results.append(
            {
                "bin": b + 1,
                "time_sec": (start + size / 2) / config.fps,
                "freq_hz": burst_frequency(len(peaks), size, config.fps),
                "mean_vel": float(np.nanmean(bin_raw)) if np.isfinite(bin_raw).any() else np.nan,
            }
        )

    return results
