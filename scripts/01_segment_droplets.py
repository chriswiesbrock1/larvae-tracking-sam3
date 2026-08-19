#!/usr/bin/env python
"""Step 1 — segment droplets, cut ROI videos and read the temperature display.

Runs SAM 3 on the first frame of each recording, splits the resulting mask into
individual droplets, and writes the droplet schema plus one masked ROI video
per droplet. If the heating stage's LCD thermometer is in frame, it is located
automatically and read for every frame, in the same decode pass.

Videos and folders can be mixed on the command line. When more than one
recording is processed, the SAM 3 weights are loaded once and reused.

Examples
--------
A single recording::

    python scripts/01_segment_droplets.py data/M4.mp4

Every recording in a folder, resuming an interrupted run::

    python scripts/01_segment_droplets.py data/session/ --skip-completed

Fainter droplets, larger minimum area, no temperature::

    python scripts/01_segment_droplets.py data/M4.mp4 \
        --threshold 0.06 --min-area-px 400 --no-temperature
"""

from __future__ import annotations

import argparse
import os
import sys

from larvatracker.config import SegmentationConfig
from larvatracker.lcd_temperature import TemperatureConfig, load_calibration
from larvatracker.pipeline import collect_videos, segment_batch


def expand_inputs(inputs: list[str], recursive: bool) -> list[str]:
    """Turn a mix of files and folders into a flat list of video paths."""
    videos: list[str] = []

    for item in inputs:
        if os.path.isdir(item):
            found = collect_videos(item, recursive=recursive)
            if not found:
                print(f"warning: no supported video in {item}", file=sys.stderr)
            videos.extend(found)
        elif os.path.isfile(item):
            videos.append(os.path.abspath(item))
        else:
            raise FileNotFoundError(item)

    # A file named on the command line and also found inside a listed folder
    # should still only be processed once.
    seen, unique = set(), []
    for video in videos:
        key = os.path.normcase(os.path.abspath(video))
        if key not in seen:
            seen.add(key)
            unique.append(video)

    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAM 3 droplet segmentation, ROI video export and temperature readout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = SegmentationConfig()

    parser.add_argument("inputs", nargs="+", help="video files and/or folders of videos")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory; only valid for a single video "
        "(default: a folder named after each recording)",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="descend into subfolders when given a folder"
    )

    segmentation = parser.add_argument_group("segmentation")
    segmentation.add_argument("--prompt", default=defaults.prompt, help="SAM 3 text prompt")
    segmentation.add_argument(
        "--threshold",
        type=float,
        default=defaults.threshold,
        help="mask threshold; lower values recover fainter droplets (0.06-0.15)",
    )
    segmentation.add_argument(
        "--min-area-px",
        type=int,
        default=defaults.min_area_px,
        help="discard connected components smaller than this",
    )
    segmentation.add_argument(
        "--padding-px",
        type=int,
        default=defaults.padding_px,
        help="margin around each droplet bounding box",
    )
    segmentation.add_argument("--codec", default=defaults.codec, help="FourCC codec for ROI videos")
    segmentation.add_argument(
        "--model-id", default=defaults.model_id, help="Hugging Face model identifier"
    )
    segmentation.add_argument(
        "--no-videos", action="store_true", help="only write the schema, skip ROI videos"
    )
    segmentation.add_argument(
        "--keep-background",
        action="store_true",
        help="do not blank pixels outside the droplet in the ROI videos",
    )
    segmentation.add_argument(
        "--pixel-table",
        action="store_true",
        help="also write droplet_pixels.csv (one row per mask pixel; large)",
    )
    segmentation.add_argument(
        "--allow-cpu",
        action="store_true",
        help="run on CPU when no GPU is available (very slow)",
    )

    temperature = parser.add_argument_group("temperature display")
    temperature.add_argument(
        "--no-temperature",
        action="store_true",
        help="do not look for the LCD thermometer",
    )
    temperature.add_argument(
        "--require-temperature",
        action="store_true",
        help="abort a recording when the LCD cannot be located, instead of "
        "continuing without a temperature trace",
    )
    temperature.add_argument(
        "--lcd-calibration",
        nargs="*",
        default=None,
        help="calibration JSON files from scripts/10_calibrate_lcd.py; matched to "
        "recordings by file name. Use when the automatic search missed the display",
    )
    temperature.add_argument(
        "--lcd-min-score",
        type=float,
        default=TemperatureConfig().locator_min_score,
        help="reject the located display below this score; lower it only if a "
        "known-good display is being missed",
    )
    temperature.add_argument(
        "--lcd-expected-start",
        type=float,
        nargs=2,
        metavar=("MIN_C", "MAX_C"),
        default=[
            TemperatureConfig().expected_start_min_c,
            TemperatureConfig().expected_start_max_c,
        ],
        help="plausible temperature at the start of a recording; candidates "
        "decoding outside this range are penalised during the search",
    )

    batch = parser.add_argument_group("batch")
    batch.add_argument(
        "--skip-completed",
        action="store_true",
        help="skip recordings whose output folder already looks complete",
    )
    batch.add_argument(
        "--stop-on-error",
        action="store_true",
        help="abort the whole batch on the first failure instead of continuing",
    )
    batch.add_argument(
        "--summary",
        default=None,
        help="path for the batch summary CSV "
        "(default: _batch_summary.csv next to the first input)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    videos = expand_inputs(args.inputs, recursive=args.recursive)
    if not videos:
        print("error: no videos to process", file=sys.stderr)
        return 1

    if args.out_dir and len(videos) > 1:
        print(
            "error: --out-dir works only with a single video; with several, each "
            "recording gets its own folder",
            file=sys.stderr,
        )
        return 2

    config = SegmentationConfig(
        prompt=args.prompt,
        threshold=args.threshold,
        min_area_px=args.min_area_px,
        padding_px=args.padding_px,
        codec=args.codec,
        mask_background=not args.keep_background,
        model_id=args.model_id,
    )

    temperature_config = TemperatureConfig(
        locator_min_score=args.lcd_min_score,
        expected_start_min_c=args.lcd_expected_start[0],
        expected_start_max_c=args.lcd_expected_start[1],
    )

    calibrations = {}
    for path in args.lcd_calibration or []:
        calibration = load_calibration(path)
        stem = os.path.splitext(os.path.basename(calibration["video"]))[0]
        calibrations[stem] = calibration
        print(f"LCD calibration for {stem}: {calibration['geometry']} "
              f"at {tuple(calibration['anchor'])}")

    summary_path = args.summary
    if summary_path is None:
        base = args.inputs[0]
        folder = base if os.path.isdir(base) else os.path.dirname(os.path.abspath(base))
        summary_path = os.path.join(folder, "_batch_summary.csv")

    summary = segment_batch(
        videos=videos,
        config=config,
        require_cuda=not args.allow_cpu,
        continue_on_error=not args.stop_on_error,
        skip_completed=args.skip_completed,
        summary_path=summary_path,
        lcd_calibrations=calibrations or None,
        out_dir=args.out_dir,
        write_videos=not args.no_videos,
        write_pixel_table=args.pixel_table,
        read_temperature=not args.no_temperature,
        temperature_config=temperature_config,
        require_temperature=args.require_temperature,
    )

    failed = int((summary.get("status") == "error").sum())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
