#!/usr/bin/env python
"""Step 4 — run the analysis over every experiment folder of a project.

Walks a project root, analyses each experiment folder that contains a
``droplet_videos`` subfolder, and concatenates all summary tables into one
combined CSV. That combined table is the input for ``05_group_statistics.py``.

Expected layout::

    project_root/
      Q1_050526/
        droplet_videos/   <- DeepLabCut CSV files
        Scheme.xlsx       <- droplet -> group mapping (optional)
      Q2_060526/
        ...

Example
-------
::

    python scripts/04_batch_analysis.py data/my_project --pattern "Q*_*"
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

from larvatracker.cli import add_analysis_arguments, config_from_args
from larvatracker.pipeline import analyse_experiment
from larvatracker.plotting import (
    plot_frequency_over_time,
    plot_group_dashboard,
)

SCHEME_NAMES = ("Scheme.xlsx", "Scheme.csv", "scheme.xlsx", "scheme.csv")


def find_scheme(folder: str) -> str | None:
    """Return the first scheme file present in ``folder``, if any."""
    for name in SCHEME_NAMES:
        candidate = os.path.join(folder, name)
        if os.path.exists(candidate):
            return candidate
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-analyse all experiment folders below a project root.",
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
        "--combined-name",
        default="Combined_All_Folders_Summary.csv",
        help="file name of the pooled summary table",
    )
    parser.add_argument("--no-plots", action="store_true", help="skip all figures")
    return add_analysis_arguments(parser)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    folders = sorted(
        f for f in glob.glob(os.path.join(args.project_root, args.pattern)) if os.path.isdir(f)
    )
    if not folders:
        print(f"No folders matching {args.pattern!r} in {args.project_root}", file=sys.stderr)
        return 1

    summaries = []

    for folder in folders:
        label = os.path.basename(os.path.normpath(folder))
        input_folder = os.path.join(folder, args.subfolder)

        if not os.path.isdir(input_folder):
            print(f"Skipping {label}: no {args.subfolder}/ subfolder")
            continue

        scheme_path = find_scheme(folder)
        if scheme_path is None:
            print(f"{label}: no scheme file, groups will be 'Unknown'")

        print(f"\n=== {label} ===")
        output_root = os.path.join(folder, "Analysis_Results")

        summary = analyse_experiment(
            input_folder=input_folder,
            output_root=output_root,
            scheme_path=scheme_path,
            config=config,
            folder_label=label,
            make_plots=not args.no_plots,
        )

        if summary.empty:
            print(f"{label}: no DeepLabCut CSV files found")
            continue

        summaries.append(summary)

        if not args.no_plots:
            if summary["Group"].nunique() > 1:
                plot_group_dashboard(
                    summary, os.path.join(output_root, f"{label}_Group_Comparison.png")
                )
            plot_frequency_over_time(
                summary, os.path.join(output_root, f"{label}_Frequency_Over_Time.png")
            )

    if not summaries:
        print("Nothing analysed.", file=sys.stderr)
        return 1

    combined = pd.concat(summaries, ignore_index=True)
    combined_path = os.path.join(args.project_root, args.combined_name)
    combined.to_csv(combined_path, index=False)

    print(f"\nCombined summary ({len(combined)} rows) written to {combined_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
