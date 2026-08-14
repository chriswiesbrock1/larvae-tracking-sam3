"""Reading DeepLabCut output and reconstructing a consistent body axis.

DeepLabCut labels five points (a-e) along the larva, but the identity of those
points is not reliable frame by frame: a symmetric, deforming larva regularly
causes head and tail labels to swap. Any metric computed per keypoint is
meaningless as long as those swaps are present.

The fix implemented here is geometric rather than model-based. For each frame
the five keypoints are projected onto their first principal component (the
body axis, obtained via SVD) and re-sorted along it. The axis direction is
carried over between frames and flipped whenever it would otherwise reverse,
which keeps "point 0" on the same end of the animal for the whole recording.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from larvatracker.config import DEFAULT_BODYPARTS


def load_dlc_csv(
    path: str,
    bodyparts: tuple[str, ...] = DEFAULT_BODYPARTS,
    likelihood_threshold: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a DeepLabCut CSV and mask out low-confidence keypoints.

    DeepLabCut writes three header rows (scorer, bodypart, coordinate) followed
    by one row per frame: ``frame, x0, y0, likelihood0, x1, y1, likelihood1, ...``
    The file is read positionally, so it works regardless of the network name
    embedded in the header.

    Parameters
    ----------
    path:
        Path to a ``*DLC*.csv`` file.
    bodyparts:
        Expected labels, in file order. Only the count matters here.
    likelihood_threshold:
        Coordinates with a lower likelihood are replaced by NaN.

    Returns
    -------
    (frames, points)
        ``frames`` has shape ``(T,)``, ``points`` has shape ``(T, n_bodyparts, 2)``.
    """
    df = pd.read_csv(path, header=None, skiprows=(0, 1, 2))

    frames = df[0].to_numpy()
    n_frames = len(frames)
    points = np.full((n_frames, len(bodyparts), 2), np.nan)

    for i in range(len(bodyparts)):
        col = 1 + i * 3
        x = df[col].to_numpy(dtype=float)
        y = df[col + 1].to_numpy(dtype=float)
        likelihood = df[col + 2].to_numpy(dtype=float)

        bad = likelihood < likelihood_threshold
        x[bad] = np.nan
        y[bad] = np.nan

        points[:, i, 0] = x
        points[:, i, 1] = y

    return frames, points


def sort_points_along_axis(
    points: np.ndarray,
    prev_axis: np.ndarray | None = None,
    min_valid: int = 3,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Order the keypoints of a single frame from one body end to the other.

    Parameters
    ----------
    points:
        Array of shape ``(n, 2)`` for one frame; may contain NaN.
    prev_axis:
        Body axis of the previous frame. If the new axis points in the opposite
        direction it is flipped, which prevents the head/tail assignment from
        alternating between frames.
    min_valid:
        Minimum number of finite keypoints required to fit an axis. Fewer than
        three points cannot define a body orientation reliably.

    Returns
    -------
    (sorted_points, axis)
        ``(None, prev_axis)`` if the frame has too few valid keypoints.
    """
    pts = np.asarray(points, dtype=float)
    n_points = pts.shape[0]

    ok = np.isfinite(pts).all(axis=1)
    if ok.sum() < min_valid:
        return None, prev_axis

    valid = pts[ok]
    center = valid.mean(axis=0)

    # First right-singular vector of the centred points = principal axis.
    _, _, vt = np.linalg.svd(valid - center, full_matrices=False)
    axis = vt[0]
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    if prev_axis is not None and np.dot(axis, prev_axis) < 0:
        axis = -axis

    # Project onto the axis; NaN keypoints are pushed to the end of the order
    # so that the array keeps its shape and missing points stay missing.
    projection = np.full(n_points, np.nan)
    projection[ok] = (pts[ok] - center) @ axis

    order = np.argsort(np.where(np.isfinite(projection), projection, np.inf))
    return pts[order], axis


def sort_series(points: np.ndarray, min_valid: int = 3) -> np.ndarray:
    """Apply :func:`sort_points_along_axis` to every frame of a recording.

    Parameters
    ----------
    points:
        Array of shape ``(T, n, 2)`` as returned by :func:`load_dlc_csv`.

    Returns
    -------
    np.ndarray
        Array of the same shape with keypoints consistently ordered along the
        body axis. Frames that could not be resolved stay NaN.
    """
    n_frames, n_points, _ = points.shape
    sorted_xy = np.full((n_frames, n_points, 2), np.nan)

    prev_axis = None
    for t in range(n_frames):
        result, axis = sort_points_along_axis(points[t], prev_axis, min_valid=min_valid)
        if result is not None:
            sorted_xy[t] = result
            prev_axis = axis

    return sorted_xy


def interpolate_gaps(values: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """Linearly interpolate short NaN runs, leaving long ones untouched.

    Bridging a few dropped frames is harmless; interpolating across a long
    tracking failure would invent movement that never happened, so runs longer
    than ``max_gap`` are kept as NaN.
    """
    series = pd.Series(values)
    filled = series.interpolate(limit_direction="both").to_numpy()

    missing = series.isna().to_numpy()
    idx = np.where(missing)[0]
    if idx.size == 0:
        return filled

    splits = np.where(np.diff(idx) != 1)[0] + 1
    for run in np.split(idx, splits):
        if len(run) > max_gap:
            filled[run] = np.nan

    return filled


def body_axis_angle(sorted_xy: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """Unwrapped orientation of the head-to-tail vector, in radians."""
    head = sorted_xy[:, 0, :]
    tail = sorted_xy[:, -1, :]
    vector = tail - head

    theta = np.arctan2(vector[:, 1], vector[:, 0])
    return np.unwrap(interpolate_gaps(theta, max_gap=max_gap))


def angular_velocity(theta: np.ndarray) -> np.ndarray:
    """Frame-to-frame change of the body axis angle (rad/frame)."""
    return np.gradient(theta)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    dot = np.sum(v1 * v2, axis=1)
    norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12
    return np.arccos(np.clip(dot / norm, -1.0, 1.0))


def joint_angles(sorted_xy: np.ndarray) -> np.ndarray:
    """Bending angle at each interior keypoint.

    Returns
    -------
    np.ndarray
        Shape ``(T, n - 2)``: for five keypoints these are the angles at b, c
        and d, i.e. the local bends of the body.
    """
    segments = np.diff(sorted_xy, axis=1)  # (T, n - 1, 2)
    return np.stack(
        [_angle_between(segments[:, i, :], segments[:, i + 1, :])
         for i in range(segments.shape[1] - 1)],
        axis=1,
    )


def curvature(sorted_xy: np.ndarray) -> np.ndarray:
    """Total body curvature: the sum of all joint angles per frame."""
    return joint_angles(sorted_xy).sum(axis=1)


def center_of_mass(sorted_xy: np.ndarray) -> np.ndarray:
    """Mean of all valid keypoints per frame, shape ``(T, 2)``.

    Frames in which no keypoint survived the likelihood filter yield NaN. That
    is the intended result, so numpy's "mean of empty slice" warning is
    suppressed rather than worked around.
    """
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(sorted_xy, axis=1)
