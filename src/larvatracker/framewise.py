"""Joining per-frame movement with the per-frame chamber temperature.

Step 3 reduces each larva to summary numbers per time bin. The temperature
analysis in :mod:`larvatracker.temperature` needs the opposite: one row per
larva, keypoint and *frame*, with the temperature that frame was recorded at.
This module produces that table.

The join itself is trivial — both sides are indexed by frame number, and both
come from the same decode pass in step 1, so no timestamp matching between two
devices is involved. What the module has to get right is the bookkeeping:

* a recording whose display could not be read has no ``temperature.csv`` and
  must be skipped without taking the whole batch down;
* a recording whose display was read only intermittently is more dangerous
  than one that failed outright, because the output still looks usable — so
  coverage is measured, reported, and can be enforced with a threshold;
* which recordings were skipped, and why, belongs in a file rather than in
  console output that scrolls past.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from larvatracker.config import AnalysisConfig
from larvatracker.metrics import moving_average_nan, step_displacement
from larvatracker.posture import load_dlc_csv, sort_series
from larvatracker.scheme import droplet_id_from_filename, group_for_droplet, load_scheme

# Column order of the framewise export. Kept identical to the historical
# exports so files from both can be fed to step 7 interchangeably.
FRAMEWISE_COLUMNS = [
    "Folder",
    "Droplet",
    "Group",
    "BodyPart",
    "Frame",
    "Time_Sec",
    "Temperature_C",
    "Movement_px_frame",
    "Movement_MA_px_frame",
]

TEMPERATURE_FILENAME = "temperature.csv"
PER_FOLDER_FILENAME = "Framewise_Movement_Temperature.csv"
COMBINED_FILENAME = "Combined_All_Folders_Framewise_Temperature.csv"

SCHEME_NAMES = ("Scheme.xlsx", "Scheme.csv", "scheme.xlsx", "scheme.csv")


def load_temperature(path: str) -> pd.DataFrame:
    """Read ``temperature.csv`` down to the three columns the join needs.

    ``temperature_c`` is the filtered value and is empty for frames where the
    display could not be read; those become NaN here rather than being dropped,
    so the caller can measure coverage.

    Returns
    -------
    pd.DataFrame
        Columns ``Frame``, ``Time_Sec``, ``Temperature_C``.
    """
    table = pd.read_csv(path, usecols=["frame", "time_s", "temperature_c"])

    return table.rename(
        columns={"frame": "Frame", "time_s": "Time_Sec", "temperature_c": "Temperature_C"}
    )


def temperature_coverage(temperature: pd.DataFrame) -> float:
    """Fraction of frames carrying a usable temperature, between 0 and 1."""
    if temperature.empty:
        return 0.0
    return float(temperature["Temperature_C"].notna().mean())


def movement_per_frame(csv_path: str, config: AnalysisConfig) -> tuple[np.ndarray, pd.DataFrame]:
    """Per-frame displacement of every keypoint of one larva.

    The keypoints are re-sorted along the body axis first, exactly as in step 3
    — without that, a head/tail label swap shows up as a large spurious
    displacement that would then be attributed to whatever temperature that
    frame happened to be at.

    Returns
    -------
    (frames, table)
        ``table`` is long format with ``BodyPart``, ``Frame``,
        ``Movement_px_frame`` and ``Movement_MA_px_frame``.
    """
    frames, points = load_dlc_csv(csv_path, config.bodyparts, config.likelihood_threshold)
    sorted_xy = sort_series(points)

    parts = []
    for index, part in enumerate(config.bodyparts):
        raw = step_displacement(sorted_xy[:, index, :])

        parts.append(
            pd.DataFrame(
                {
                    "BodyPart": part,
                    "Frame": frames,
                    "Movement_px_frame": raw,
                    "Movement_MA_px_frame": moving_average_nan(raw, config.smoothing_window),
                }
            )
        )

    return frames, pd.concat(parts, ignore_index=True)


def build_framewise_for_experiment(
    input_folder: str,
    temperature_path: str,
    scheme_path: str | None = None,
    config: AnalysisConfig | None = None,
    folder_label: str | None = None,
    drop_missing_temperature: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Build the framewise table for one experiment folder.

    Parameters
    ----------
    input_folder:
        Folder holding the ``droplet_XXXDLC_*.csv`` files.
    temperature_path:
        The recording's ``temperature.csv``.
    drop_missing_temperature:
        Drop rows whose frame has no temperature. They carry no information for
        the temperature analysis, and on a recording with a poorly read display
        they can be the majority of the file.

    Returns
    -------
    (table, info)
        ``info`` reports droplet count, frame count, temperature coverage and
        any frame-count mismatch between movement and temperature.
    """
    cfg = config or AnalysisConfig()
    folder_label = folder_label or os.path.basename(os.path.normpath(input_folder))

    temperature = load_temperature(temperature_path)
    coverage = temperature_coverage(temperature)

    scheme = load_scheme(scheme_path) if scheme_path else None

    csv_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
    tables: list[pd.DataFrame] = []

    n_droplets = 0
    frame_mismatch = 0

    for path in csv_files:
        droplet_id = droplet_id_from_filename(path)
        if droplet_id is None:
            # Not a per-droplet DeepLabCut export (a summary table, say).
            continue

        n_droplets += 1
        frames, movement = movement_per_frame(path, cfg)

        # Temperature was logged during the same decode pass that cut the ROI
        # videos, so the frame numbering is shared. A mismatch means the two
        # came from different runs and the join would silently misalign.
        if len(frames) != len(temperature):
            frame_mismatch = max(frame_mismatch, abs(len(frames) - len(temperature)))

        merged = movement.merge(temperature, on="Frame", how="inner")
        merged.insert(0, "Group", group_for_droplet(scheme, droplet_id))
        merged.insert(0, "Droplet", droplet_id)
        merged.insert(0, "Folder", folder_label)

        tables.append(merged)

    if not tables:
        raise ValueError(f"{input_folder}: no DeepLabCut CSV files found.")

    table = pd.concat(tables, ignore_index=True)[FRAMEWISE_COLUMNS]

    n_before = len(table)
    if drop_missing_temperature:
        table = table[table["Temperature_C"].notna()].reset_index(drop=True)

    info = {
        "folder": folder_label,
        "droplets": n_droplets,
        "frames": int(len(temperature)),
        "temperature_coverage": round(coverage, 4),
        "rows": int(len(table)),
        "rows_dropped_no_temperature": int(n_before - len(table)),
        "frame_count_mismatch": int(frame_mismatch),
        "groups": sorted(table["Group"].unique().tolist()),
        "status": "ok",
        "reason": "",
    }

    return table, info


def _find_in(folders: tuple[str, ...], names: tuple[str, ...]) -> str | None:
    for folder in folders:
        for name in names:
            candidate = os.path.join(folder, name)
            if os.path.isfile(candidate):
                return candidate
    return None


def find_temperature_file(experiment_folder: str, input_folder: str) -> str | None:
    """Locate ``temperature.csv`` for an experiment.

    Step 1 writes it next to ``droplet_videos``, but a hand-assembled dataset
    may place it inside. Both are accepted; None means the display was never
    read for this recording.
    """
    return _find_in((experiment_folder, input_folder), (TEMPERATURE_FILENAME,))


def find_scheme_file(experiment_folder: str, input_folder: str) -> str | None:
    """Locate the droplet-to-group scheme for an experiment."""
    return _find_in((experiment_folder, input_folder), SCHEME_NAMES)


def build_framewise_batch(
    project_root: str,
    pattern: str = "*",
    subfolder: str = "droplet_videos",
    config: AnalysisConfig | None = None,
    drop_missing_temperature: bool = True,
    min_coverage: float = 0.0,
    write_per_folder: bool = True,
    combined_name: str = COMBINED_FILENAME,
) -> tuple[str | None, pd.DataFrame]:
    """Build the framewise export for every experiment below ``project_root``.

    Recordings without a ``temperature.csv``, or whose coverage falls below
    ``min_coverage``, are skipped and recorded in the report rather than
    aborting the run — a single unread display should not cost the whole batch.

    The combined file is written incrementally, one experiment at a time, so
    memory stays flat regardless of how many recordings are processed.

    Returns
    -------
    (combined_path, report)
        ``combined_path`` is None if nothing could be processed. ``report`` has
        one row per experiment with its status and coverage.
    """
    cfg = config or AnalysisConfig()

    folders = sorted(
        f for f in glob.glob(os.path.join(project_root, pattern)) if os.path.isdir(f)
    )

    combined_path = os.path.join(project_root, combined_name)
    rows: list[dict] = []
    first_write = True

    for experiment_folder in folders:
        label = os.path.basename(os.path.normpath(experiment_folder))
        input_folder = os.path.join(experiment_folder, subfolder)

        if not os.path.isdir(input_folder):
            continue

        temperature_path = find_temperature_file(experiment_folder, input_folder)

        if temperature_path is None:
            print(f"{label}: SKIPPED — no {TEMPERATURE_FILENAME} (display was never read)")
            rows.append(
                {
                    "folder": label,
                    "status": "skipped",
                    "reason": f"no {TEMPERATURE_FILENAME}",
                    "temperature_coverage": 0.0,
                    "droplets": 0,
                    "frames": 0,
                    "rows": 0,
                }
            )
            continue

        try:
            table, info = build_framewise_for_experiment(
                input_folder=input_folder,
                temperature_path=temperature_path,
                scheme_path=find_scheme_file(experiment_folder, input_folder),
                config=cfg,
                folder_label=label,
                drop_missing_temperature=drop_missing_temperature,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            print(f"{label}: ERROR — {exc}")
            rows.append(
                {
                    "folder": label,
                    "status": "error",
                    "reason": str(exc),
                    "temperature_coverage": 0.0,
                    "droplets": 0,
                    "frames": 0,
                    "rows": 0,
                }
            )
            continue

        if info["temperature_coverage"] < min_coverage:
            print(
                f"{label}: SKIPPED — temperature coverage "
                f"{info['temperature_coverage']:.1%} below the required {min_coverage:.0%}"
            )
            info.update(
                status="skipped",
                reason=f"coverage {info['temperature_coverage']:.1%} < {min_coverage:.0%}",
                rows=0,
            )
            rows.append(info)
            continue

        if info["frame_count_mismatch"]:
            print(
                f"{label}: WARNING — movement and temperature differ by "
                f"{info['frame_count_mismatch']} frame(s); only the overlap was kept. "
                "This usually means the two came from different runs of step 1."
            )

        if info["temperature_coverage"] < 0.8:
            print(
                f"{label}: WARNING — only {info['temperature_coverage']:.1%} of frames "
                "carry a temperature. Check temperature_display_debug.png."
            )

        print(
            f"{label}: {info['droplets']} droplets, {info['rows']:,} rows, "
            f"coverage {info['temperature_coverage']:.1%}"
        )

        if write_per_folder:
            table.to_csv(os.path.join(experiment_folder, PER_FOLDER_FILENAME), index=False)

        table.to_csv(
            combined_path, mode="w" if first_write else "a", header=first_write, index=False
        )
        first_write = False

        rows.append(info)

    report = pd.DataFrame(rows)

    if first_write:
        print("\nNothing could be processed — no combined file written.")
        return None, report

    return combined_path, report
