#!/usr/bin/env python
"""Step 3 — analyse the DeepLabCut output of a single experiment.

Reads every ``droplet_XXXDLC_*.csv`` in a folder, reconstructs a consistent
body axis, computes burst frequency, onset latency and mean velocity per
keypoint (whole recording and per time bin), and writes per-droplet trace
figures plus a long-format summary table.

Examples
--------
Without group information::

    python scripts/03_analyze_experiment.py data/M4/droplet_videos

With a treatment scheme and a population overview::

    python scripts/03_analyze_experiment.py data/Q4/droplet_videos \
        --scheme data/Q4/Scheme.xlsx --dashboard
"""

from __future__ import annotations

import argparse
import os
import sys

from larvatracker.cli import add_analysis_arguments, config_from_args
from larvatracker.pipeline import analyse_experiment
from larvatracker.plotting import (
    plot_frequency_over_time,
    plot_group_dashboard,
    plot_population_overview,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse DeepLabCut pose estimates for one experiment folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_folder", help="folder containing the DeepLabCut CSV files")
    parser.add_argument(
        "--scheme", default=None, help="scheme file mapping droplet IDs to treatment groups"
    )
    parser.add_argument(
        "--output-root", default=None, help="default: <input_folder>/Analysis_Results"
    )
    parser.add_argument("--no-plots", action="store_true", help="skip the per-droplet figures")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="also write the group comparison and population overview figures",
    )
    return add_analysis_arguments(parser)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    output_root = args.output_root or os.path.join(args.input_folder, "Analysis_Results")

    summary = analyse_experiment(
        input_folder=args.input_folder,
        output_root=output_root,
        scheme_path=args.scheme,
        config=config_from_args(args),
        make_plots=not args.no_plots,
    )

    if summary.empty:
        print("No DeepLabCut CSV files found — nothing to analyse.", file=sys.stderr)
        return 1

    print(f"\nSummary table written to {output_root}")

    if args.dashboard:
        plot_population_overview(summary, os.path.join(output_root, "Population_Overview.png"))
        plot_frequency_over_time(summary, os.path.join(output_root, "Frequency_Over_Time.png"))

        if summary["Group"].nunique() > 1:
            plot_group_dashboard(
                summary, os.path.join(output_root, "Group_Comparison_Dashboard.png")
            )

        print("Figures written.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
