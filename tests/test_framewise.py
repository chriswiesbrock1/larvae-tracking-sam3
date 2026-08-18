"""Tests for the movement/temperature join."""

import os

import numpy as np
import pandas as pd
import pytest

from larvatracker.config import AnalysisConfig
from larvatracker.framewise import (
    FRAMEWISE_COLUMNS,
    build_framewise_batch,
    build_framewise_for_experiment,
    find_scheme_file,
    find_temperature_file,
    load_temperature,
    movement_per_frame,
    temperature_coverage,
)


# ---------------------------------------------------------------------------
# Synthetic experiment folder
# ---------------------------------------------------------------------------

def write_dlc_csv(path, n_frames, seed=0):
    """A DeepLabCut export with five keypoints along a moving larva."""
    rng = np.random.default_rng(seed)
    x0 = y0 = 30.0
    rows = []

    for t in range(n_frames):
        x0 += 0.4 * np.sin(t / 5.0)
        y0 += 0.4 * np.cos(t / 7.0)

        values = [t]
        for k in range(5):
            offset = (k - 2) * 5.0
            values += [
                round(x0 + offset + rng.normal(0, 0.2), 3),
                round(y0 + rng.normal(0, 0.2), 3),
                0.95,
            ]
        rows.append(values)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("scorer," + ",".join(["DLC_test"] * 15) + "\n")
        handle.write("bodyparts," + ",".join(sum([[c] * 3 for c in "abcde"], [])) + "\n")
        handle.write("coords," + ",".join(["x", "y", "likelihood"] * 5) + "\n")
        for row in rows:
            handle.write(",".join(map(str, row)) + "\n")


def write_temperature_csv(path, n_frames, missing_frames=(), fps=30.0):
    """A temperature.csv with an optional set of unreadable frames."""
    rows = []
    for t in range(n_frames):
        value = "" if t in set(missing_frames) else f"{22.0 + t * 0.01:.1f}"
        status = "missing" if t in set(missing_frames) else "ok"
        rows.append(f"{t},{t / fps:.6f},compact,{value},22.0,20.0,1,{status}")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "frame,time_s,geometry,temperature_c,raw_temperature_c,"
            "confidence,raw_valid,status\n"
        )
        handle.write("\n".join(rows) + "\n")


def make_experiment(root, name, n_droplets=3, n_frames=120, missing_frames=(), temperature=True):
    folder = os.path.join(root, name)
    videos = os.path.join(folder, "droplet_videos")
    os.makedirs(videos, exist_ok=True)

    for droplet in range(1, n_droplets + 1):
        write_dlc_csv(
            os.path.join(videos, f"droplet_{droplet:03d}DLC_Resnet50_test.csv"),
            n_frames,
            seed=droplet,
        )

    if temperature:
        write_temperature_csv(os.path.join(folder, "temperature.csv"), n_frames, missing_frames)

    with open(os.path.join(folder, "scheme.csv"), "w", encoding="utf-8") as handle:
        handle.write("Droplet,Group\n")
        for droplet in range(1, n_droplets + 1):
            handle.write(f"{droplet},{'conventional' if droplet % 2 else 'germfree'}\n")

    return folder


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_temperature_renames_and_keeps_gaps(tmp_path):
    path = str(tmp_path / "temperature.csv")
    write_temperature_csv(path, 50, missing_frames=(10, 11, 12))

    table = load_temperature(path)

    assert list(table.columns) == ["Frame", "Time_Sec", "Temperature_C"]
    assert len(table) == 50
    # Gaps must survive loading so coverage can be measured.
    assert table["Temperature_C"].isna().sum() == 3
    assert temperature_coverage(table) == pytest.approx(47 / 50)


def test_movement_matches_a_direct_computation(tmp_path):
    from larvatracker.metrics import moving_average_nan, step_displacement
    from larvatracker.posture import load_dlc_csv, sort_series

    path = str(tmp_path / "droplet_001DLC_Resnet50_test.csv")
    write_dlc_csv(path, 200, seed=3)

    config = AnalysisConfig()
    _, table = movement_per_frame(path, config)

    _, points = load_dlc_csv(path, config.bodyparts, config.likelihood_threshold)
    sorted_xy = sort_series(points)
    expected = step_displacement(sorted_xy[:, 2, :])   # keypoint 'c'

    got = table[table["BodyPart"] == "c"].sort_values("Frame")

    assert np.allclose(got["Movement_px_frame"], expected, equal_nan=True)
    assert np.allclose(
        got["Movement_MA_px_frame"],
        moving_average_nan(expected, config.smoothing_window),
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# Per-experiment join
# ---------------------------------------------------------------------------

def test_join_produces_one_row_per_droplet_keypoint_frame(tmp_path):
    folder = make_experiment(str(tmp_path), "V1", n_droplets=3, n_frames=100)

    table, info = build_framewise_for_experiment(
        input_folder=os.path.join(folder, "droplet_videos"),
        temperature_path=os.path.join(folder, "temperature.csv"),
        scheme_path=os.path.join(folder, "scheme.csv"),
    )

    assert list(table.columns) == FRAMEWISE_COLUMNS
    assert len(table) == 3 * 5 * 100
    assert info["droplets"] == 3
    assert info["temperature_coverage"] == 1.0
    assert info["frame_count_mismatch"] == 0
    assert sorted(info["groups"]) == ["conventional", "germfree"]


def test_temperature_is_matched_to_the_right_frame(tmp_path):
    folder = make_experiment(str(tmp_path), "V1", n_droplets=1, n_frames=60)

    table, _ = build_framewise_for_experiment(
        input_folder=os.path.join(folder, "droplet_videos"),
        temperature_path=os.path.join(folder, "temperature.csv"),
    )

    reference = load_temperature(os.path.join(folder, "temperature.csv"))
    merged = table.merge(reference, on="Frame", suffixes=("", "_ref"))

    assert np.allclose(merged["Temperature_C"], merged["Temperature_C_ref"])
    assert np.allclose(merged["Time_Sec"], merged["Time_Sec_ref"])


def test_frames_without_temperature_are_dropped_by_default(tmp_path):
    folder = make_experiment(
        str(tmp_path), "V1", n_droplets=2, n_frames=100, missing_frames=range(40, 60)
    )

    table, info = build_framewise_for_experiment(
        input_folder=os.path.join(folder, "droplet_videos"),
        temperature_path=os.path.join(folder, "temperature.csv"),
    )

    assert table["Temperature_C"].notna().all()
    assert len(table) == 2 * 5 * 80
    assert info["rows_dropped_no_temperature"] == 2 * 5 * 20
    assert info["temperature_coverage"] == pytest.approx(0.8)


def test_frames_without_temperature_can_be_kept(tmp_path):
    folder = make_experiment(
        str(tmp_path), "V1", n_droplets=1, n_frames=50, missing_frames=range(10, 20)
    )

    table, _ = build_framewise_for_experiment(
        input_folder=os.path.join(folder, "droplet_videos"),
        temperature_path=os.path.join(folder, "temperature.csv"),
        drop_missing_temperature=False,
    )

    assert len(table) == 1 * 5 * 50
    assert table["Temperature_C"].isna().sum() == 5 * 10


def test_frame_count_mismatch_is_reported_not_silently_misaligned(tmp_path):
    """Movement and temperature from different runs must not be zipped together."""
    folder = make_experiment(str(tmp_path), "V1", n_droplets=1, n_frames=100)

    # Temperature covers fewer frames than the tracking does.
    write_temperature_csv(os.path.join(folder, "temperature.csv"), 80)

    table, info = build_framewise_for_experiment(
        input_folder=os.path.join(folder, "droplet_videos"),
        temperature_path=os.path.join(folder, "temperature.csv"),
    )

    assert info["frame_count_mismatch"] == 20
    # Only the overlap survives, and frame numbers still line up.
    assert table["Frame"].max() == 79
    assert len(table) == 1 * 5 * 80


def test_missing_dlc_files_raise(tmp_path):
    folder = os.path.join(str(tmp_path), "empty")
    os.makedirs(os.path.join(folder, "droplet_videos"))
    write_temperature_csv(os.path.join(folder, "temperature.csv"), 10)

    with pytest.raises(ValueError):
        build_framewise_for_experiment(
            input_folder=os.path.join(folder, "droplet_videos"),
            temperature_path=os.path.join(folder, "temperature.csv"),
        )


# ---------------------------------------------------------------------------
# Batch behaviour
# ---------------------------------------------------------------------------

def test_experiment_without_temperature_is_skipped_not_fatal(tmp_path):
    """The whole point: one unread display must not cost the batch."""
    root = str(tmp_path)
    make_experiment(root, "V1", n_droplets=2, n_frames=60, temperature=False)
    make_experiment(root, "V2", n_droplets=2, n_frames=60)
    make_experiment(root, "V3", n_droplets=2, n_frames=60)

    combined_path, report = build_framewise_batch(root, pattern="V*")

    assert combined_path is not None and os.path.isfile(combined_path)

    status = dict(zip(report["folder"], report["status"]))
    assert status == {"V1": "skipped", "V2": "ok", "V3": "ok"}
    assert "temperature.csv" in report.loc[report.folder == "V1", "reason"].iloc[0]

    combined = pd.read_csv(combined_path)
    assert sorted(combined["Folder"].unique()) == ["V2", "V3"]
    assert len(combined) == 2 * (2 * 5 * 60)


def test_low_coverage_can_be_refused(tmp_path):
    root = str(tmp_path)
    make_experiment(root, "V1", n_droplets=2, n_frames=100, missing_frames=range(0, 70))
    make_experiment(root, "V2", n_droplets=2, n_frames=100)

    _, report = build_framewise_batch(root, pattern="V*", min_coverage=0.8)

    status = dict(zip(report["folder"], report["status"]))
    assert status["V1"] == "skipped"
    assert status["V2"] == "ok"
    assert "coverage" in report.loc[report.folder == "V1", "reason"].iloc[0]


def test_batch_without_any_temperature_writes_nothing(tmp_path):
    root = str(tmp_path)
    make_experiment(root, "V1", n_droplets=1, n_frames=40, temperature=False)

    combined_path, report = build_framewise_batch(root, pattern="V*")

    assert combined_path is None
    assert report["status"].tolist() == ["skipped"]


def test_helpers_find_files_in_either_location(tmp_path):
    folder = make_experiment(str(tmp_path), "V1", n_droplets=1, n_frames=20)
    videos = os.path.join(folder, "droplet_videos")

    assert find_temperature_file(folder, videos) == os.path.join(folder, "temperature.csv")
    assert find_scheme_file(folder, videos) == os.path.join(folder, "scheme.csv")

    empty = str(tmp_path / "nothing")
    os.makedirs(empty, exist_ok=True)
    assert find_temperature_file(empty, empty) is None
