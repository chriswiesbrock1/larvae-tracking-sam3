#!/usr/bin/env python
"""Step 8 — join per-frame movement with the per-frame chamber temperature.

Produces the framewise export that step 7 consumes: one row per larva,
keypoint and frame, carrying both the displacement and the temperature that
frame was recorded at.

Recordings whose LCD could not be read have no ``temperature.csv``. They are
skipped and listed in the report instead of aborting the batch — one unread
display should not cost the whole run.

Examples
--------
A whole project::

    python scripts/08_framewise_temperature.py data/Genotypes

Only experiments matching a pattern, and refuse recordings whose display was
read for less than 80 % of frames::

    python scripts/08_framewise_temperature.py data/Genotypes \
        --pattern "V*_*" --min-coverage 0.8

Expected layout::

    project_root/
      V1_09032026/
        droplet_videos/     # DeepLabCut CSVs
        temperature.csv     # from step 1; absent if the LCD was not found
        scheme.csv          # optional droplet -> group mapping
      V2_09032026/
        ...
"""

from __future__ import annotations

import argparse
import os
import sys

from larvatracker.cli import add_analysis_arguments, config_from_args
from larvatracker.framewise import COMBINED_FILENAME, build_framewise_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join per-frame movement with the per-frame chamber temperature.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("project_root", help="folder containing the experiment folders")
    parser.add_argument(
        "--pattern", default="*", help="glob pattern selecting the experiment folders"
    )
    parser.add_argument(
        "--subfolder",
        default="droplet_videos",
        help="subfolder holding the DeepLabCut CSV files",
    )
    parser.add_argument(
        "--combined-name", default=COMBINED_FILENAME, help="file name of the combined export"
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="skip recordings whose display was read for a smaller fraction of "
        "frames; 0 keeps everything and only warns",
    )
    parser.add_argument(
        "--keep-missing-temperature",
        action="store_true",
        help="keep rows whose frame has no temperature instead of dropping them",
    )
    parser.add_argument(
        "--no-per-folder",
        action="store_true",
        help="write only the combined file, not one per experiment",
    )
    return add_analysis_arguments(parser)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.isdir(args.project_root):
        print(f"error: {args.project_root} is not a folder", file=sys.stderr)
        return 2

    combined_path, report = build_framewise_batch(
        project_root=args.project_root,
        pattern=args.pattern,
        subfolder=args.subfolder,
        config=config_from_args(args),
        drop_missing_temperature=not args.keep_missing_temperature,
        min_coverage=args.min_coverage,
        write_per_folder=not args.no_per_folder,
        combined_name=args.combined_name,
    )

    if report.empty:
        print(
            f"error: no experiment folder matching {args.pattern!r} with a "
            f"{args.subfolder}/ subfolder",
            file=sys.stderr,
        )
        return 1

    report_path = os.path.join(args.project_root, "_framewise_report.csv")
    report.to_csv(report_path, index=False)

    ok = report[report["status"] == "ok"]
    skipped = report[report["status"] == "skipped"]
    errored = report[report["status"] == "error"]

    print(f"\n{'=' * 60}")
    print(f"Processed : {len(ok)} experiment(s), {int(ok['rows'].sum()):,} rows")

    if not skipped.empty:
        print(f"Skipped   : {len(skipped)} — {', '.join(skipped['folder'])}")
        for _, row in skipped.iterrows():
            print(f"            {row['folder']}: {row['reason']}")

    if not errored.empty:
        print(f"Failed    : {len(errored)} — {', '.join(errored['folder'])}")

    print(f"Report    : {report_path}")

    if combined_path:
        print(f"Combined  : {combined_path}")
    else:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
