#!/usr/bin/env python
"""Step 0b — calibrate the temperature display on a recording.

Step 1 finds the LCD by scanning the whole frame. That works when the display
is unambiguous, but it can settle on the wrong spot — and then it reports a
confident temperature that is not what the screen says. When a recording ends
up with no ``temperature.csv``, or with a low coverage in
``_framewise_report.csv``, this is the tool to reach for.

You supply the number visible on screen at the start of the recording. The
calibration is then chosen by the only criterion that cannot be faked: it has
to decode to exactly that value. The result is written as JSON and handed to
step 1 with ``--lcd-calibration``.

Examples
--------
Calibrate one recording and check the coverage::

    python scripts/10_calibrate_lcd.py data/V1.mp4 --known-temperature 23.7

Narrow the search when the display is small or the scene is busy::

    python scripts/10_calibrate_lcd.py data/V1.mp4 --known-temperature 23.7 \
        --roi 1580 0 1920 220

Recover the temperature without re-running the segmentation::

    python scripts/10_calibrate_lcd.py data/V1.mp4 --known-temperature 23.7 \
        --write-temperature

Or hand the calibration to step 1 for a fresh run::

    python scripts/01_segment_droplets.py data/V1.mp4 \
        --lcd-calibration data/V1_lcd.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from larvatracker.lcd_temperature import (
    TemperatureConfig,
    calibrate_display,
    calibration_debug_image,
    measure_coverage,
    read_temperature_track,
    save_calibration,
    write_temperature_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate the LCD thermometer on a recording.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("videos", nargs="+", help="one or more recordings")
    parser.add_argument(
        "--known-temperature",
        type=float,
        nargs="+",
        required=True,
        help="the value shown on screen at the start, read off by eye — one per "
        "video, or a single value used for all of them",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=None,
        help="search region around the display; found by colour when omitted",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="where to write the JSON and the debug image (default: next to the video)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.9,
        help="warn when the calibration reads fewer than this fraction of frames",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=5,
        help="check every Nth frame when measuring coverage",
    )
    parser.add_argument(
        "--write-temperature",
        action="store_true",
        help="also read the whole recording and write temperature.csv into its "
        "output folder — recovers a failed readout without re-running step 1",
    )
    parser.add_argument(
        "--no-coverage-check",
        action="store_true",
        help="skip the coverage measurement (faster, but only checks the first frame)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    known = args.known_temperature
    if len(known) == 1:
        known = known * len(args.videos)
    elif len(known) != len(args.videos):
        print(
            "error: give one --known-temperature per video, or a single value for all",
            file=sys.stderr,
        )
        return 2

    config = TemperatureConfig()
    failures = 0

    for video, expected in zip(args.videos, known):
        name = os.path.splitext(os.path.basename(video))[0]
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(video))
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== {name} (screen shows {expected} °C) ===")

        try:
            calibration = calibrate_display(
                video, known_temperature=expected, config=config,
                roi=tuple(args.roi) if args.roi else None,
            )
        except Exception as exc:  # noqa: BLE001 - reported per video
            print(f"  FAILED: {exc}", file=sys.stderr)
            failures += 1
            continue

        print(
            f"  geometry {calibration['geometry']} at scale {calibration['scale']:.2f}, "
            f"anchor {tuple(calibration['anchor'])}, confidence {calibration['confidence']}"
        )

        if not args.no_coverage_check:
            coverage = measure_coverage(
                video, calibration, config=config, every=args.sample_every
            )
            calibration["coverage"] = coverage

            print(
                f"  reads {coverage['frames_valid']}/{coverage['frames_sampled']} sampled "
                f"frames ({coverage['coverage']:.0%}), "
                f"{coverage['min_value']:.1f}–{coverage['max_value']:.1f} °C"
            )

            if coverage["coverage"] < args.min_coverage:
                print(
                    f"  WARNING: below the required {args.min_coverage:.0%}. Check the "
                    "debug image; the grid may be sitting slightly off the digits.",
                    file=sys.stderr,
                )

        if args.write_temperature:
            recording_dir = os.path.join(os.path.dirname(os.path.abspath(video)), name)
            os.makedirs(recording_dir, exist_ok=True)

            records = read_temperature_track(video, calibration, config=config)
            _, summary = write_temperature_csv(records, recording_dir, config)

            print(
                f"  temperature.csv: {summary['frames']} frames, "
                f"{summary['missing']} missing, "
                f"{summary['min_c']:.1f}–{summary['max_c']:.1f} °C"
            )

        json_path = save_calibration(calibration, os.path.join(out_dir, f"{name}_lcd.json"))

        try:
            import cv2

            debug = calibration_debug_image(video, calibration, config=config)
            cv2.imwrite(os.path.join(out_dir, f"{name}_lcd_debug.png"), debug)
        except Exception as exc:  # noqa: BLE001 - the JSON is what matters
            print(f"  (no debug image: {exc})")

        print(f"  written: {json_path}")

    if failures:
        print(f"\n{failures} recording(s) could not be calibrated.", file=sys.stderr)
        return 1

    print("\nHand these to step 1 with --lcd-calibration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
