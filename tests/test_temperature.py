"""Tests for the temperature-response analysis."""

import numpy as np
import pandas as pd
import pytest

from larvatracker.temperature import (
    add_temperature_bins,
    bin_coverage,
    canonical_group,
    compare_to_controls,
    control_for_group,
    filter_by_coverage,
    group_summary,
    normalise_to_baseline_seconds,
    per_larva_by_bin,
)


# --- label harmonisation --------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1 mM ASP", "1 mM Asp"),
        ("1mM Ibu", "1 mM Ibu"),
        ("5 mM  Asp", "5 mM Asp"),
        ("10 mM ibu", "10 mM Ibu"),
        ("etoh", "ETOH"),
        ("  DMSO ", "DMSO"),
        ("PBS", "PBS"),
    ],
)
def test_canonical_group(raw, expected):
    assert canonical_group(raw) == expected


def test_canonical_group_leaves_unknown_labels_alone():
    """An unrecognised label must survive intact so it can be spotted."""
    assert canonical_group("some new treatment") == "some new treatment"


@pytest.mark.parametrize(
    "group,expected",
    [
        ("5 mM Asp", "ETOH"),
        ("10 mM Ibu", "ETOH"),
        ("1 mM Dic", "DMSO"),
        ("5 mM Cis", "PBS"),
        ("ETOH", None),
        ("DMSO", None),
        ("mystery drug", None),
    ],
)
def test_control_for_group(group, expected):
    assert control_for_group(group) == expected


# --- normalisation --------------------------------------------------------

def make_frames(n_larvae=4, n_frames=600, fps=30.0, baseline=2.0, gain=3.0):
    """Larvae that move at `baseline` for 10 s, then at baseline * gain.

    600 frames at 30 fps is 20 s, so the recording extends past the 10 s
    baseline window — with exactly 300 frames there would be no post-baseline
    data to test against.
    """
    rows = []
    for larva in range(1, n_larvae + 1):
        for frame in range(n_frames):
            t = frame / fps
            rows.append(
                {
                    "Dataset": "D",
                    "Folder": "F1",
                    "Droplet": larva,
                    "Group": "ETOH",
                    "Frame": frame,
                    "Time_Sec": t,
                    "Temperature_C": 22.0 if t < 10 else 30.0,
                    "Movement": baseline * larva if t < 10 else baseline * larva * gain,
                }
            )
    return pd.DataFrame(rows)


def test_normalisation_makes_the_baseline_exactly_one():
    frames, excluded = normalise_to_baseline_seconds(make_frames(), baseline_seconds=10.0)

    early = frames[frames["Time_Sec"] < 10]
    assert np.allclose(early["Movement_norm"], 1.0)
    assert excluded.empty


def test_normalisation_removes_the_between_larva_scale():
    """Larvae moving at different absolute speeds must end up on one scale."""
    frames, _ = normalise_to_baseline_seconds(make_frames(gain=3.0), baseline_seconds=10.0)

    late = frames[frames["Time_Sec"] >= 10]
    assert np.allclose(late["Movement_norm"], 3.0)
    assert late.groupby("Droplet")["Movement_norm"].std().max() == pytest.approx(0.0)


def test_larva_without_baseline_movement_is_excluded_not_divided_by_zero():
    frames = make_frames(n_larvae=2)
    frames.loc[(frames["Droplet"] == 1) & (frames["Time_Sec"] < 10), "Movement"] = 0.0

    kept, excluded = normalise_to_baseline_seconds(frames, baseline_seconds=10.0)

    assert 1 not in set(kept["Droplet"])
    assert len(excluded) == 1
    assert np.isfinite(kept["Movement_norm"]).all()


def test_short_baseline_window_is_excluded():
    frames = make_frames(n_larvae=2)
    # Larva 2 is only tracked for a fraction of the baseline window.
    mask = (frames["Droplet"] == 2) & (frames["Time_Sec"] < 10) & (frames["Frame"] > 20)
    frames.loc[mask, "Movement"] = np.nan

    kept, excluded = normalise_to_baseline_seconds(
        frames, baseline_seconds=10.0, min_baseline_frames=30
    )

    assert 2 not in set(kept["Droplet"])
    assert "baseline frames" in excluded.iloc[0]["Reason"]


# --- binning --------------------------------------------------------------

def test_temperature_bins_are_labelled_by_their_centre():
    frames = pd.DataFrame({"Temperature_C": [22.0, 22.4, 22.5, 22.9, 23.0]})

    binned = add_temperature_bins(frames, bin_width=0.5)

    assert list(binned["Temp_Bin"]) == [22.25, 22.25, 22.75, 22.75, 23.25]


def test_bins_with_too_few_frames_are_dropped():
    frames, _ = normalise_to_baseline_seconds(make_frames(), baseline_seconds=10.0)
    frames = add_temperature_bins(frames)

    per_larva = per_larva_by_bin(frames, min_frames=1000)

    assert per_larva.empty


# --- coverage -------------------------------------------------------------

def make_per_larva():
    rows = []
    for folder in ["F1", "F2", "F3", "F4", "F5", "F6"]:
        for droplet in [1, 2, 3]:
            for temp_bin, only_f1 in [(25.25, False), (30.25, False), (38.75, True)]:
                if only_f1 and folder != "F1":
                    continue
                rows.append(
                    {
                        "Dataset": "D",
                        "Folder": folder,
                        "Droplet": droplet,
                        "Group": "ETOH" if droplet == 1 else "5 mM Asp",
                        "Temp_Bin": temp_bin,
                        "Movement_norm": 1.0 + droplet * 0.1,
                        "N_Frames": 100,
                    }
                )
    return pd.DataFrame(rows)


def test_coverage_counts_experiments_not_animals():
    coverage = bin_coverage(make_per_larva()).set_index("Temp_Bin")

    assert coverage.loc[25.25, "N_Folders"] == 6
    assert coverage.loc[38.75, "N_Folders"] == 1
    assert coverage.loc[25.25, "N_Observations"] == 18


def test_thinly_covered_bins_are_dropped_and_reported():
    kept, dropped = filter_by_coverage(make_per_larva(), min_folders=5)

    assert 38.75 not in set(kept["Temp_Bin"])
    assert list(dropped["Temp_Bin"]) == [38.75]


def test_group_summary_reports_animals_and_experiments():
    summary = group_summary(make_per_larva())
    row = summary[(summary["Group"] == "5 mM Asp") & (summary["Temp_Bin"] == 25.25)].iloc[0]

    assert row["N_Larvae"] == 12   # 6 folders x 2 droplets
    assert row["N_Folders"] == 6
    assert row["SEM"] == pytest.approx(row["SD"] / np.sqrt(row["N_Larvae"]))


# --- statistics -----------------------------------------------------------

def test_comparison_detects_a_planted_difference():
    rng = np.random.default_rng(0)
    rows = []

    for i in range(40):
        for group, centre in [("ETOH", 1.0), ("5 mM Asp", 2.0)]:
            rows.append(
                {
                    "Dataset": "D",
                    "Folder": f"F{i % 5}",
                    "Droplet": i,
                    "Group": group,
                    "Temp_Bin": 30.25,
                    "Movement_norm": rng.normal(centre, 0.15),
                }
            )

    stats = compare_to_controls(pd.DataFrame(rows))

    assert len(stats) == 1
    assert stats.iloc[0]["signif"]
    assert stats.iloc[0]["Log2_Ratio"] == pytest.approx(1.0, abs=0.2)


def test_controls_are_not_compared_against_themselves():
    rows = [
        {"Dataset": "D", "Folder": "F1", "Droplet": i, "Group": g,
         "Temp_Bin": 30.25, "Movement_norm": 1.0 + i * 0.01}
        for g in ["ETOH", "DMSO", "PBS"]
        for i in range(10)
    ]

    assert compare_to_controls(pd.DataFrame(rows)).empty


# ---------------------------------------------------------------------------
# Coverage threshold adapts to the dataset
# ---------------------------------------------------------------------------

def per_larva_with_folders(n_folders, n_bins=4):
    """One larva per folder, present in every bin."""
    rows = []
    for f in range(n_folders):
        for b in range(n_bins):
            rows.append(
                {
                    "Dataset": "D",
                    "Folder": f"F{f}",
                    "Droplet": 1,
                    "Group": "ETOH" if f % 2 else "5 mM Asp",
                    "Temp_Bin": 25.0 + b,
                    "Movement_norm": 1.0,
                    "N_Frames": 100,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "n_folders,expected",
    [(1, 1), (2, 1), (3, 2), (4, 2), (8, 4), (19, 5), (40, 5)],
)
def test_default_threshold_is_a_capped_majority(n_folders, expected):
    """A fixed threshold empties small datasets; a majority scales with them."""
    from larvatracker.temperature import default_min_folders

    assert default_min_folders(per_larva_with_folders(n_folders)) == expected


def test_small_dataset_survives_the_default_threshold():
    """Three experiments must not produce an empty analysis."""
    from larvatracker.temperature import filter_by_coverage

    kept, dropped = filter_by_coverage(per_larva_with_folders(3))

    assert not kept.empty
    assert dropped.empty
