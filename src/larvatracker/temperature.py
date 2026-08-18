"""Movement as a function of temperature, compared across treatment groups.

This module works on the framewise export (``Framewise_Movement_Temperature.csv``
and its combined variants), which holds one row per droplet, keypoint and frame
together with the chamber temperature at that frame.

The reduction happens in four steps:

1. **Pool the keypoints.** The question is how *much* the animal moved, not
   which part moved, so the five keypoints are averaged into one movement value
   per larva and frame.
2. **Normalise per larva.** Each larva is divided by its own mean movement
   during the first seconds of the recording, while the chamber is still at
   baseline temperature. Absolute movement depends on body size and on how
   well that particular droplet was tracked; the ratio does not.
3. **Bin by temperature.** Frames are grouped into fixed-width temperature
   bins, and each larva contributes one mean value per bin.
4. **Compare groups.** Per bin, every treatment is tested against its vehicle
   control.

Step 3 is what makes the per-larva table the right unit for statistics: a
recording contributes thousands of frames but only a handful of animals, and
frames within a larva are heavily autocorrelated. Aggregating to one value per
larva and bin before testing avoids treating frames as independent samples.
"""

from __future__ import annotations

import math
import os
import re

import numpy as np
import pandas as pd

# Rows whose Group matches one of these are dropped before anything else.
# "Temp" marks the bare temperature probe droplet, "invalid" a droplet that was
# flagged during manual inspection; neither is an experimental animal.
DEFAULT_EXCLUDED_GROUPS = ("invalid", "Temp")

# Column names of the framewise export.
KEY_COLUMNS = ["Folder", "Droplet", "Group", "BodyPart", "Frame"]
VALUE_COLUMNS = ["Time_Sec", "Temperature_C"]

# Vehicle control for each drug family. Aspirin and ibuprofen are dissolved in
# ethanol, diclofenac in DMSO, cisplatin in PBS, so each treatment is compared
# against the vehicle it was actually delivered in rather than against a single
# global control.
DEFAULT_CONTROL_MAP = {
    "asp": "ETOH",
    "ibu": "ETOH",
    "dic": "DMSO",
    "cis": "PBS",
}

CONTROL_GROUPS = ("ETOH", "DMSO", "PBS")


def canonical_group(label: str) -> str:
    """Fold hand-typed group labels onto one spelling.

    Labels drift across recording days: ``1 mM ASP``, ``1mM Ibu`` and
    ``5 mM  Asp`` all occur in the exports. Without this step each spelling
    would silently form its own group and split the sample.

    The concentration and the drug name are parsed out and re-emitted in a
    fixed format, so the function is robust to spacing and capitalisation:

    >>> canonical_group("1mM ASP")
    '1 mM Asp'
    >>> canonical_group("etoh")
    'ETOH'
    """
    text = re.sub(r"\s+", " ", str(label)).strip()

    # Controls have no concentration and are written in upper case.
    for control in CONTROL_GROUPS:
        if text.lower() == control.lower():
            return control

    match = re.match(r"^(\d+(?:\.\d+)?)\s*mM\s+(.+)$", text, flags=re.IGNORECASE)
    if not match:
        return text

    concentration, drug = match.groups()
    drug = drug.strip().lower()

    # Title case is the house style for drug abbreviations (Asp, Ibu, Dic, Cis).
    return f"{concentration} mM {drug[:1].upper()}{drug[1:]}"


def control_for_group(group: str, control_map: dict[str, str] | None = None) -> str | None:
    """Return the vehicle control a treatment group should be tested against.

    Returns None for the control groups themselves and for anything that does
    not match a known drug family, so unmapped groups are skipped rather than
    silently compared against the wrong vehicle.
    """
    control_map = control_map or DEFAULT_CONTROL_MAP

    if group in CONTROL_GROUPS:
        return None

    lowered = group.lower()
    for token, control in control_map.items():
        if token in lowered:
            return control

    return None


def load_framewise(
    path: str,
    signal: str = "Movement_MA_px_frame",
    excluded_groups: tuple[str, ...] = DEFAULT_EXCLUDED_GROUPS,
    dataset: str | None = None,
    chunksize: int = 1_000_000,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Read a framewise export and reduce it to one row per larva and frame.

    The file is read in chunks and each chunk is immediately collapsed over the
    keypoint axis, so peak memory stays proportional to the number of
    *frames* rather than the number of rows. A 700 MB export reduces to a few
    tens of MB this way.

    The keypoint mean is accumulated as a sum and a count rather than a mean,
    because a frame's five keypoint rows are far apart in the file and can land
    in different chunks. Summing first and dividing at the end makes the result
    independent of where the chunk boundaries fall.

    Parameters
    ----------
    path:
        The framewise CSV.
    signal:
        Movement column to use, raw or smoothed.
    excluded_groups:
        Group labels to drop entirely.
    dataset:
        Value for the ``Dataset`` column; defaults to the file's folder name.

    Returns
    -------
    (frames, info)
        ``frames`` has one row per (dataset, folder, droplet, frame) with the
        keypoint-averaged movement. ``info`` reports what was read and dropped.
    """
    dataset = dataset or os.path.basename(os.path.dirname(os.path.abspath(path)))
    excluded_lower = {g.lower() for g in excluded_groups}

    usecols = KEY_COLUMNS + VALUE_COLUMNS + [signal]
    partials: list[pd.DataFrame] = []

    n_total = 0
    n_excluded = 0
    excluded_counts: dict[str, int] = {}

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        n_total += len(chunk)

        mask = chunk["Group"].astype(str).str.strip().str.lower().isin(excluded_lower)
        if mask.any():
            n_excluded += int(mask.sum())
            for label, count in chunk.loc[mask, "Group"].value_counts().items():
                excluded_counts[label] = excluded_counts.get(label, 0) + int(count)
            chunk = chunk[~mask]

        if chunk.empty:
            continue

        chunk["Group"] = chunk["Group"].map(canonical_group)

        value = pd.to_numeric(chunk[signal], errors="coerce")
        chunk = chunk.assign(
            _sum=value.fillna(0.0),
            _count=value.notna().astype("int32"),
        )

        partials.append(
            chunk.groupby(["Folder", "Droplet", "Frame"], sort=False).agg(
                Group=("Group", "first"),
                Time_Sec=("Time_Sec", "first"),
                Temperature_C=("Temperature_C", "first"),
                _sum=("_sum", "sum"),
                _count=("_count", "sum"),
            )
        )

    if not partials:
        raise ValueError(f"{path}: no rows left after excluding {sorted(excluded_groups)}")

    # Merge the per-chunk partials; a frame split across chunks is summed here.
    frames = (
        pd.concat(partials)
        .groupby(level=["Folder", "Droplet", "Frame"], sort=False)
        .agg(
            Group=("Group", "first"),
            Time_Sec=("Time_Sec", "first"),
            Temperature_C=("Temperature_C", "first"),
            _sum=("_sum", "sum"),
            _count=("_count", "sum"),
        )
        .reset_index()
    )

    # Frames where no keypoint was tracked have count 0 and stay NaN.
    frames["Movement"] = np.where(
        frames["_count"] > 0, frames["_sum"] / frames["_count"].replace(0, np.nan), np.nan
    )
    frames = frames.drop(columns=["_sum", "_count"])
    frames.insert(0, "Dataset", dataset)

    info = {
        "path": path,
        "dataset": dataset,
        "rows_read": n_total,
        "rows_excluded": n_excluded,
        "excluded_counts": excluded_counts,
        "frames": len(frames),
        "larvae": int(frames.groupby(["Folder", "Droplet"]).ngroups),
        "groups": sorted(frames["Group"].unique()),
    }

    if verbose:
        print(
            f"{dataset}: {n_total:,} rows -> {len(frames):,} larva-frames "
            f"({info['larvae']} larvae, {n_excluded:,} rows excluded)"
        )

    return frames, info


def normalise_to_baseline_seconds(
    frames: pd.DataFrame,
    baseline_seconds: float = 10.0,
    subject_keys: tuple[str, ...] = ("Dataset", "Folder", "Droplet"),
    min_baseline_frames: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide each larva by its own mean movement in the opening seconds.

    Parameters
    ----------
    baseline_seconds:
        Length of the baseline window, measured from the start of the
        recording, while the chamber is still at ambient temperature.
    min_baseline_frames:
        Larvae with fewer usable frames in the window are excluded. A baseline
        computed from a handful of frames is too noisy to divide by.

    Returns
    -------
    (normalised, excluded)
        ``normalised`` gains a ``Movement_norm`` column. ``excluded`` lists the
        larvae that were dropped and why — a larva that never moved during the
        baseline window has no defined ratio, so it is removed rather than
        clipped to an arbitrary value.
    """
    keys = list(subject_keys)

    window = frames[frames["Time_Sec"] < baseline_seconds]
    baseline = window.groupby(keys)["Movement"].agg(
        Baseline="mean", Baseline_Frames="count"
    )

    usable = (
        baseline["Baseline"].notna()
        & (baseline["Baseline"] > 0)
        & (baseline["Baseline_Frames"] >= min_baseline_frames)
    )

    excluded = baseline[~usable].reset_index()
    if not excluded.empty:
        excluded["Reason"] = np.where(
            excluded["Baseline_Frames"] < min_baseline_frames,
            f"fewer than {min_baseline_frames} usable baseline frames",
            "baseline movement is zero or undefined",
        )

    merged = frames.merge(baseline[usable], left_on=keys, right_index=True, how="inner")
    merged["Movement_norm"] = merged["Movement"] / merged["Baseline"]

    return merged, excluded


def add_temperature_bins(
    frames: pd.DataFrame,
    bin_width: float = 0.5,
    column: str = "Temperature_C",
) -> pd.DataFrame:
    """Assign each frame to a fixed-width temperature bin.

    The bin is labelled by its centre, so a 0.5 °C bin covering [22.0, 22.5)
    is reported as 22.25 °C. Labelling by the centre keeps the x-axis of the
    resulting plot on the true temperature scale.
    """
    out = frames.copy()
    lower = np.floor(out[column] / bin_width) * bin_width

    out["Temp_Bin"] = lower + bin_width / 2.0
    out["Temp_Bin_Low"] = lower
    return out


def per_larva_by_bin(
    frames: pd.DataFrame,
    value_col: str = "Movement_norm",
    subject_keys: tuple[str, ...] = ("Dataset", "Folder", "Droplet"),
    min_frames: int = 10,
) -> pd.DataFrame:
    """Collapse to one value per larva and temperature bin.

    This is the unit of analysis for every statistic that follows. Bins in
    which a larva contributed fewer than ``min_frames`` frames are dropped,
    because a bin the chamber swept through in a fraction of a second gives an
    unreliable mean.
    """
    keys = list(subject_keys) + ["Group", "Temp_Bin"]

    out = (
        frames.groupby(keys, sort=True)
        .agg(
            Movement_norm=(value_col, "mean"),
            N_Frames=(value_col, "count"),
            Temperature_C=("Temperature_C", "mean"),
        )
        .reset_index()
    )

    return out[out["N_Frames"] >= min_frames].reset_index(drop=True)


def bin_coverage(per_larva: pd.DataFrame) -> pd.DataFrame:
    """How many experiments and animals contribute to each temperature bin.

    Recordings do not all start at the same ambient temperature and do not all
    reach the same maximum, so the extreme bins are populated by a small,
    non-random subset of experiments. A group difference in such a bin is
    confounded with which recordings happen to cover it. This table makes that
    visible; :func:`filter_by_coverage` acts on it.
    """
    return (
        per_larva.groupby("Temp_Bin")
        .agg(
            N_Folders=("Folder", "nunique"),
            N_Datasets=("Dataset", "nunique"),
            N_Groups=("Group", "nunique"),
            N_Observations=("Movement_norm", "size"),
        )
        .reset_index()
    )


def default_min_folders(per_larva: pd.DataFrame, cap: int = 5) -> int:
    """Pick a sensible coverage threshold for the dataset at hand.

    A fixed threshold does not survive contact with datasets of different
    sizes: five experiments is a reasonable bar when nineteen were recorded,
    but it silently empties the analysis when there are only three. The
    requirement is therefore a simple majority of the available experiments,
    capped so that a large dataset does not become needlessly strict.

    With a temperature ramp the chamber sweeps some bins quickly, so not every
    recording contributes to every bin even when all of them cover the range —
    which is why a majority, rather than all, is the right bar.
    """
    n_folders = int(per_larva["Folder"].nunique())
    if n_folders <= 1:
        return 1

    return min(cap, max(1, math.ceil(n_folders / 2)))


def filter_by_coverage(
    per_larva: pd.DataFrame,
    min_folders: int | None = None,
    min_groups: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop temperature bins that too few experiments reached.

    Parameters
    ----------
    min_folders:
        Minimum number of experiments a bin must appear in. ``None`` picks a
        value with :func:`default_min_folders` based on the dataset size.

    Returns
    -------
    (kept, dropped_bins)
        ``dropped_bins`` is the coverage table restricted to the removed bins,
        so the exclusion is auditable rather than silent.
    """
    if min_folders is None:
        min_folders = default_min_folders(per_larva)

    coverage = bin_coverage(per_larva)
    ok = (coverage["N_Folders"] >= min_folders) & (coverage["N_Groups"] >= min_groups)

    keep_bins = set(coverage.loc[ok, "Temp_Bin"])
    kept = per_larva[per_larva["Temp_Bin"].isin(keep_bins)].reset_index(drop=True)

    return kept, coverage[~ok].reset_index(drop=True)


def group_summary(per_larva: pd.DataFrame, value_col: str = "Movement_norm") -> pd.DataFrame:
    """Mean, SEM and sample size per treatment group and temperature bin.

    ``N_Larvae`` counts animals, not frames — that is the number the error bars
    are based on. ``N_Folders`` counts the experiments behind those animals; a
    bin where many animals come from one recording is weaker evidence than the
    animal count alone suggests.
    """
    summary = (
        per_larva.groupby(["Group", "Temp_Bin"], sort=True)
        .agg(
            Mean=(value_col, "mean"),
            SD=(value_col, "std"),
            N_Larvae=(value_col, "count"),
            N_Folders=("Folder", "nunique"),
        )
        .reset_index()
    )

    summary["SEM"] = summary["SD"] / np.sqrt(summary["N_Larvae"])
    return summary


def compare_controls(
    per_larva: pd.DataFrame,
    value_col: str = "Movement_norm",
    ranges: tuple[tuple[float, float, str], ...] = (
        (23.0, 28.0, "cold"),
        (28.0, 31.0, "rising"),
        (31.0, 39.0, "hot"),
    ),
) -> pd.DataFrame:
    """Compare the vehicle controls against each other.

    A quality check, not a result. If ETOH, DMSO and PBS larvae behave
    differently, then a curve from one dataset cannot be read against a curve
    from another: the difference may come from the vehicle or from the
    recording batch rather than from the drug. Only comparisons within a
    dataset, against that dataset's own vehicle, stay interpretable.

    Returns one row per control group and temperature range, with the number of
    animals behind each value.
    """
    controls = per_larva[per_larva["Group"].isin(CONTROL_GROUPS)]
    rows = []

    for low, high, label in ranges:
        window = controls[(controls["Temp_Bin"] >= low) & (controls["Temp_Bin"] < high)]
        if window.empty:
            continue

        per_animal = (
            window.groupby(["Dataset", "Group", "Folder", "Droplet"])[value_col]
            .mean()
            .reset_index()
        )

        for (dataset, group), data in per_animal.groupby(["Dataset", "Group"]):
            rows.append(
                {
                    "Range": label,
                    "Temp_Low": low,
                    "Temp_High": high,
                    "Dataset": dataset,
                    "Group": group,
                    "N_Larvae": len(data),
                    "Mean": data[value_col].mean(),
                    "Median": data[value_col].median(),
                    "SD": data[value_col].std(),
                }
            )

    return pd.DataFrame(rows)


def compare_to_controls(
    per_larva: pd.DataFrame,
    value_col: str = "Movement_norm",
    control_map: dict[str, str] | None = None,
    min_n: int = 3,
    fdr_method: str = "fdr_bh",
) -> pd.DataFrame:
    """Test every treatment against its vehicle control, per temperature bin.

    A two-sided Mann-Whitney U test is used because the normalised values are
    ratios and not normally distributed. All comparisons are corrected together
    with Benjamini-Hochberg FDR: with ~37 temperature bins times several
    groups, uncorrected p-values would produce false positives by construction.

    Returns
    -------
    pd.DataFrame
        One row per (group, bin) comparison, with group sizes in *animals*,
        medians, and raw plus corrected p-values.
    """
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    rows = []

    for group in sorted(per_larva["Group"].unique()):
        control = control_for_group(group, control_map)
        if control is None:
            continue

        treatment_data = per_larva[per_larva["Group"] == group]
        control_data = per_larva[per_larva["Group"] == control]

        for temp_bin in sorted(treatment_data["Temp_Bin"].unique()):
            treated = treatment_data.loc[treatment_data["Temp_Bin"] == temp_bin, value_col].dropna()
            untreated = control_data.loc[control_data["Temp_Bin"] == temp_bin, value_col].dropna()

            if len(treated) < min_n or len(untreated) < min_n:
                continue

            _, p_value = mannwhitneyu(treated, untreated, alternative="two-sided")

            rows.append(
                {
                    "Group": group,
                    "Control": control,
                    "Temp_Bin": temp_bin,
                    "N_Treatment": len(treated),
                    "N_Control": len(untreated),
                    "Median_Treatment": treated.median(),
                    "Median_Control": untreated.median(),
                    "Log2_Ratio": np.log2(treated.median() / untreated.median())
                    if untreated.median() > 0
                    else np.nan,
                    "p_raw": p_value,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["p_fdr"] = multipletests(result["p_raw"], method=fdr_method)[1]
    result["signif"] = result["p_fdr"] < 0.05

    return result.sort_values(["Group", "Temp_Bin"]).reset_index(drop=True)
