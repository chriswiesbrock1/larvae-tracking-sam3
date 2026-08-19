"""End-to-end drivers that tie the individual modules together.

Two entry points are provided:

``segment_recording``
    raw video -> droplet schema -> per-droplet ROI videos (+ temperature)

``segment_batch``
    every video in a folder, with the SAM 3 weights loaded once

``analyse_experiment``
    folder of DeepLabCut CSVs -> per-droplet figures + long-format summary table

Everything the command line scripts do is a thin wrapper around these.
"""

from __future__ import annotations

import glob
import os
import traceback

import numpy as np
import pandas as pd

from larvatracker.config import AnalysisConfig, SegmentationConfig
from larvatracker.droplets import find_droplets, save_schema_outputs
from larvatracker.metrics import bodypart_metrics, time_bin_metrics
from larvatracker.plotting import plot_droplet_traces
from larvatracker.posture import load_dlc_csv, sort_series
from larvatracker.roi_videos import write_droplet_videos
from larvatracker.scheme import droplet_id_from_filename, group_for_droplet, load_scheme

# Column order of the long-format summary table written by analyse_experiment.
SUMMARY_COLUMNS = [
    "Folder",
    "Droplet",
    "Group",
    "BodyPart",
    "Time_Bin",
    "Time_Sec",
    "Freq_Hz",
    "Onset_Sec",
    "Mean_Vel",
    "Burst_Count",
]


def _label_for(input_folder: str) -> str:
    """Derive a readable experiment label from the input folder.

    The CSVs normally sit in ``<experiment>/droplet_videos``; that generic
    subfolder name would be useless in the ``Folder`` column, so the parent is
    used instead.
    """
    path = os.path.normpath(os.path.abspath(input_folder))
    name = os.path.basename(path)

    if name.lower() == "droplet_videos":
        return os.path.basename(os.path.dirname(path)) or name

    return name


def default_output_dir(video_path: str) -> str:
    """``/data/M4.mp4`` -> ``/data/M4`` (created if missing)."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(video_path)), base)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")

# Files that must exist for a recording to count as already processed.
COMPLETION_MARKERS = ("droplets.csv",)


def segment_recording(
    video_path: str,
    out_dir: str | None = None,
    config: SegmentationConfig | None = None,
    write_videos: bool = True,
    write_pixel_table: bool = False,
    require_cuda: bool = True,
    components: tuple | None = None,
    read_temperature: bool = True,
    temperature_config=None,
    require_temperature: bool = False,
    lcd_calibration: dict | None = None,
) -> dict:
    """Segment droplets, export ROI videos and read the temperature display.

    Parameters
    ----------
    components:
        A ``(device, model, processor)`` tuple from
        :func:`larvatracker.segmentation.load_sam3`, so a batch loads the
        weights once.
    read_temperature:
        Locate the LCD thermometer and log its reading for every frame. The
        display is searched once and then read from a small ROI, inside the
        same decode pass that writes the ROI videos.
    require_temperature:
        Treat a failed LCD search as a fatal error. By default the recording is
        still processed and only the temperature is missing — the ROI videos
        are the expensive part and are worth keeping even when the display was
        out of frame.
    lcd_calibration:
        A calibration from ``scripts/10_calibrate_lcd.py``. When given, the
        display is read at that known position instead of being searched for.
        Use it when the full-frame search missed the display or settled on the
        wrong spot.

    Returns
    -------
    dict
        Summary of what was produced, suitable as a row in the batch table.
    """
    from larvatracker.segmentation import segment_first_frame

    cfg = config or SegmentationConfig()
    out_dir = out_dir or default_output_dir(video_path)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Output directory: {out_dir}")

    frame_bgr, mask = segment_first_frame(
        video_path, cfg, require_cuda=require_cuda, components=components
    )
    droplets, id_mask = find_droplets(mask, cfg.min_area_px, cfg.padding_px)
    print(f"Droplets found: {len(droplets)}")

    save_schema_outputs(
        out_dir=out_dir,
        frame_bgr=frame_bgr,
        mask=mask,
        droplets=droplets,
        id_mask=id_mask,
        config=cfg,
        write_pixel_table=write_pixel_table,
    )

    result = {
        "video": video_path,
        "output_dir": out_dir,
        "droplets": len(droplets),
        "frames": 0,
        "temperature_geometry": "",
        "temperature_min_c": None,
        "temperature_max_c": None,
        "temperature_missing_frames": None,
        "lcd_locator_score": None,
        "status": "ok",
        "error": "",
    }

    # --- locate the temperature display before the decode pass ------------
    reader = None
    records: list[dict] = []

    if read_temperature:
        from larvatracker.lcd_temperature import (
            GEOMETRY_PROFILES,
            locate_display,
            reader_at,
        )

        try:
            if lcd_calibration is not None:
                # Position already established and verified against the value on
                # screen; searching again could only move it somewhere worse.
                reader = reader_at(
                    frame_bgr.shape,
                    GEOMETRY_PROFILES[lcd_calibration["geometry"]],
                    lcd_calibration["scale"],
                    tuple(lcd_calibration["anchor"]),
                    temperature_config,
                    name=lcd_calibration["geometry"],
                )
                reader.calibrate(frame_bgr)
                print(
                    f"LCD from calibration: {reader.geometry} at "
                    f"{tuple(lcd_calibration['anchor'])}, scale {reader.scale:.2f}"
                )
            else:
                reader, _ = locate_display(video_path, temperature_config)

            result["temperature_geometry"] = reader.geometry
            result["lcd_locator_score"] = reader.locator_score

            debug = reader.debug_image(frame_bgr, reader.read(frame_bgr))
            import cv2

            cv2.imwrite(os.path.join(out_dir, "temperature_display_debug.png"), debug)
        except Exception as exc:  # noqa: BLE001 - a missing LCD is not fatal
            if require_temperature:
                raise
            print(f"WARNING: no temperature read for {os.path.basename(video_path)}: {exc}")
            reader = None

    def on_frame(frame_idx: int, time_s: float, frame) -> None:
        record = reader.read(frame)
        record["frame"] = frame_idx
        record["time_s"] = time_s
        records.append(record)

    # --- single decode pass: ROI videos and temperature -------------------
    if write_videos:
        pass_result = write_droplet_videos(
            video_path=video_path,
            droplets=droplets,
            out_dir=os.path.join(out_dir, "droplet_videos"),
            codec=cfg.codec,
            mask_background=cfg.mask_background,
            frame_callback=on_frame if reader is not None else None,
        )
        result["frames"] = pass_result["frames"]

    elif reader is not None:
        # No ROI videos wanted, but the temperature still needs every frame.
        import cv2

        from larvatracker.roi_videos import frame_timestamp

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx, last_time = 0, -1.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                last_time = frame_timestamp(cap, frame_idx, fps, last_time)
                on_frame(frame_idx, last_time, frame)
                frame_idx += 1
        finally:
            cap.release()
        result["frames"] = frame_idx

    if records:
        from larvatracker.lcd_temperature import write_temperature_csv

        _, summary = write_temperature_csv(records, out_dir, temperature_config)
        result["temperature_min_c"] = summary["min_c"]
        result["temperature_max_c"] = summary["max_c"]
        result["temperature_missing_frames"] = summary["missing"]

    return result


def collect_videos(folder: str, recursive: bool = False) -> list[str]:
    """Every supported video in ``folder``, sorted by name.

    ``droplet_videos`` subfolders are skipped in recursive mode: they contain
    this pipeline's own output, and feeding a droplet ROI back in as a source
    recording would be nonsense.
    """
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)

    videos: list[str] = []

    if recursive:
        for root, directories, filenames in os.walk(folder):
            directories[:] = [d for d in directories if d.lower() != "droplet_videos"]
            for name in filenames:
                if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
                    videos.append(os.path.join(root, name))
    else:
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
                videos.append(path)

    return sorted(videos, key=str.lower)


def is_already_processed(video_path: str, read_temperature: bool = True) -> bool:
    """Whether a recording's output folder already holds a complete result."""
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(video_path)),
        os.path.splitext(os.path.basename(video_path))[0],
    )

    markers = list(COMPLETION_MARKERS)
    if read_temperature:
        markers.append("temperature.csv")

    return all(os.path.isfile(os.path.join(out_dir, m)) for m in markers)


def segment_batch(
    videos: list[str],
    config: SegmentationConfig | None = None,
    require_cuda: bool = True,
    continue_on_error: bool = True,
    skip_completed: bool = False,
    summary_path: str | None = None,
    lcd_calibrations: dict | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run :func:`segment_recording` over many videos, loading SAM 3 once.

    One failing recording does not abandon the rest by default; the error is
    recorded in the summary table so it can be dealt with afterwards. GPU
    memory is released between videos either way.

    Returns
    -------
    pd.DataFrame
        One row per video: droplet count, frame count, temperature range and
        status.
    """
    from larvatracker.segmentation import load_sam3

    cfg = config or SegmentationConfig()
    read_temperature = kwargs.get("read_temperature", True)

    pending = [
        v for v in videos
        if not (skip_completed and is_already_processed(v, read_temperature))
    ]
    skipped = [v for v in videos if v not in pending]

    print(f"Videos: {len(videos)} total, {len(pending)} to process, {len(skipped)} skipped")

    rows: list[dict] = []
    for video in skipped:
        rows.append({"video": video, "status": "skipped_existing", "error": ""})

    components = load_sam3(cfg, require_cuda=require_cuda) if pending else None

    for index, video in enumerate(pending, start=1):
        print("\n" + "=" * 70)
        print(f"VIDEO {index}/{len(pending)}: {video}")
        print("=" * 70)

        try:
            calibration = None
            if lcd_calibrations:
                stem = os.path.splitext(os.path.basename(video))[0]
                calibration = lcd_calibrations.get(stem)

            rows.append(
                segment_recording(
                    video_path=video,
                    config=cfg,
                    require_cuda=require_cuda,
                    components=components,
                    lcd_calibration=calibration,
                    **kwargs,
                )
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            print(f"ERROR on {video}: {exc}")
            traceback.print_exc()

            rows.append({"video": video, "status": "error", "error": str(exc)})

            if not continue_on_error:
                raise
        finally:
            _free_gpu_memory()

    summary = pd.DataFrame(rows)

    if summary_path:
        summary.to_csv(summary_path, index=False)
        print(f"\nBatch summary: {summary_path}")

    n_ok = int((summary.get("status") == "ok").sum())
    n_error = int((summary.get("status") == "error").sum())
    print(f"Done. {n_ok} succeeded, {n_error} failed, {len(skipped)} skipped.")

    return summary


def _free_gpu_memory() -> None:
    """Drop cached CUDA allocations between videos, if torch is present."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def analyse_experiment(
    input_folder: str,
    output_root: str | None = None,
    scheme_path: str | None = None,
    config: AnalysisConfig | None = None,
    folder_label: str | None = None,
    make_plots: bool = True,
) -> pd.DataFrame:
    """Analyse every DeepLabCut CSV in one experiment folder.

    Parameters
    ----------
    input_folder:
        Folder containing the ``droplet_XXXDLC_*.csv`` files.
    output_root:
        Where figures and the summary table go. Defaults to
        ``<input_folder>/Analysis_Results``.
    scheme_path:
        Optional scheme file assigning treatment groups to droplet IDs. Without
        it every droplet is labelled ``Unknown``.
    folder_label:
        Name used in the ``Folder`` column; defaults to the folder's own name.

    Returns
    -------
    pd.DataFrame
        Long-format summary: one row per droplet, keypoint and time bin, plus a
        row with ``Time_Bin == "full"`` covering the whole recording.
    """
    cfg = config or AnalysisConfig()

    output_root = output_root or os.path.join(input_folder, "Analysis_Results")
    os.makedirs(output_root, exist_ok=True)

    folder_label = folder_label or _label_for(input_folder)
    scheme = load_scheme(scheme_path) if scheme_path else None

    csv_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
    rows: list[dict] = []

    for path in csv_files:
        droplet_id = droplet_id_from_filename(path)
        if droplet_id is None:
            # Not a per-droplet DeepLabCut export (e.g. a summary table).
            continue

        group = group_for_droplet(scheme, droplet_id)
        print(f"{folder_label} | droplet {droplet_id:03d} | group {group}")

        frames, points = load_dlc_csv(path, cfg.bodyparts, cfg.likelihood_threshold)
        sorted_xy = sort_series(points)

        metrics_per_part = {}

        for i, part in enumerate(cfg.bodyparts):
            metrics = bodypart_metrics(sorted_xy, i, cfg)
            metrics_per_part[part] = metrics

            rows.append(
                {
                    "Folder": folder_label,
                    "Droplet": droplet_id,
                    "Group": group,
                    "BodyPart": part,
                    "Time_Bin": "full",
                    "Time_Sec": np.nan,
                    "Freq_Hz": round(metrics["freq_hz"], 3),
                    "Onset_Sec": round(metrics["onset_sec"], 2) if np.isfinite(metrics["onset_sec"]) else np.nan,
                    "Mean_Vel": round(metrics["mean_vel"], 3) if np.isfinite(metrics["mean_vel"]) else np.nan,
                    "Burst_Count": metrics["burst_count"],
                }
            )

            for entry in time_bin_metrics(metrics["raw_step"], metrics["ma_step"], cfg):
                rows.append(
                    {
                        "Folder": folder_label,
                        "Droplet": droplet_id,
                        "Group": group,
                        "BodyPart": part,
                        "Time_Bin": str(entry["bin"]),
                        "Time_Sec": entry["time_sec"],
                        "Freq_Hz": round(entry["freq_hz"], 3),
                        "Onset_Sec": np.nan,
                        "Mean_Vel": round(entry["mean_vel"], 3) if np.isfinite(entry["mean_vel"]) else np.nan,
                        "Burst_Count": np.nan,
                    }
                )

        if make_plots:
            save_dir = os.path.join(output_root, f"droplet_{droplet_id:03d}")
            os.makedirs(save_dir, exist_ok=True)
            plot_droplet_traces(
                frames=frames,
                metrics_per_part=metrics_per_part,
                out_path=os.path.join(save_dir, f"{folder_label}_droplet_{droplet_id:03d}_traces.png"),
                title=f"{folder_label} | droplet {droplet_id} | group {group}",
            )

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    if not summary.empty:
        summary.to_csv(
            os.path.join(output_root, f"{folder_label}_Summary_Results.csv"), index=False
        )

    return summary
