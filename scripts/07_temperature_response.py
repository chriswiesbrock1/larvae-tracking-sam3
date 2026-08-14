#!/usr/bin/env python
"""Step 7 — compare movement across temperature between treatment groups.

Takes one or more framewise exports, normalises every larva to its own opening
seconds, bins the frames by chamber temperature and compares each treatment
against its vehicle control.

Several files can be passed at once; each is tagged with a ``Dataset`` column
so they can be pooled into one figure while staying distinguishable in the
tables.

Examples
--------
One dataset::

    python scripts/07_temperature_response.py \
        results/Analgetics/Combined_All_Folders_Framewise_Temperature.csv \
        --out-dir results/temperature

Two datasets pooled, with explicit names::

    python scripts/07_temperature_response.py \
        Analgetics/Combined_All_Folders_Framewise_Temperature.csv \
        Cisplatin/Combined_All_Folders_Framewise_Temperature.csv \
        --dataset-names Analgetics Cisplatin_Diclo \
        --out-dir results/temperature
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from larvatracker.plotting import (
    plot_temperature_response,
    plot_temperature_response_by_family,
    plot_significance_heatmap,
)
from larvatracker.temperature import (
    DEFAULT_CONTROL_MAP,
    DEFAULT_EXCLUDED_GROUPS,
    add_temperature_bins,
    bin_coverage,
    compare_controls,
    compare_to_controls,
    filter_by_coverage,
    group_summary,
    load_framewise,
    normalise_to_baseline_seconds,
    per_larva_by_bin,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Movement versus temperature, compared across treatment groups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_files", nargs="+", help="one or more framewise exports")
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=None,
        help="label for each input file (default: the file's parent folder name)",
    )
    parser.add_argument("--out-dir", default="temperature_results", help="output directory")
    parser.add_argument(
        "--signal",
        default="Movement_MA_px_frame",
        choices=["Movement_MA_px_frame", "Movement_px_frame"],
        help="smoothed or raw movement column",
    )
    parser.add_argument(
        "--exclude-groups",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_GROUPS),
        help="group labels to drop (the temperature probe and flagged droplets)",
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=10.0,
        help="length of the per-larva baseline window at the start of the recording",
    )
    parser.add_argument(
        "--min-baseline-frames",
        type=int,
        default=30,
        help="minimum usable frames in the baseline window; below this a larva is dropped",
    )
    parser.add_argument("--temp-bin", type=float, default=0.5, help="temperature bin width in °C")
    parser.add_argument(
        "--min-frames-per-bin",
        type=int,
        default=10,
        help="minimum frames a larva must contribute to a bin to be counted",
    )
    parser.add_argument(
        "--min-folders-per-bin",
        type=int,
        default=5,
        help="drop temperature bins reached by fewer experiments; the extreme "
        "bins are covered by only a few recordings and would confound group "
        "differences with which recording got there",
    )
    parser.add_argument(
        "--control-map",
        default=None,
        help="JSON file mapping a drug-name substring to its vehicle control "
        f"(default: {json.dumps(DEFAULT_CONTROL_MAP)})",
    )
    parser.add_argument("--chunksize", type=int, default=1_000_000, help="CSV read chunk size")
    parser.add_argument("--no-plots", action="store_true", help="write tables only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset_names and len(args.dataset_names) != len(args.csv_files):
        print("error: --dataset-names must have one entry per input file", file=sys.stderr)
        return 2

    control_map = DEFAULT_CONTROL_MAP
    if args.control_map:
        with open(args.control_map, encoding="utf-8") as handle:
            control_map = json.load(handle)

    # --- 1. load and pool -------------------------------------------------
    all_frames = []
    infos = []

    for i, path in enumerate(args.csv_files):
        name = args.dataset_names[i] if args.dataset_names else None
        frames, info = load_framewise(
            path,
            signal=args.signal,
            excluded_groups=tuple(args.exclude_groups),
            dataset=name,
            chunksize=args.chunksize,
        )
        all_frames.append(frames)
        infos.append(info)

    frames = pd.concat(all_frames, ignore_index=True)
    del all_frames

    # --- 2. normalise per larva ------------------------------------------
    frames, excluded = normalise_to_baseline_seconds(
        frames,
        baseline_seconds=args.baseline_seconds,
        min_baseline_frames=args.min_baseline_frames,
    )

    if not excluded.empty:
        excluded.to_csv(os.path.join(args.out_dir, "excluded_larvae.csv"), index=False)
        print(f"\n{len(excluded)} larva(e) excluded — see excluded_larvae.csv")
        print(excluded["Reason"].value_counts().to_string())

    # --- 3. bin by temperature -------------------------------------------
    frames = add_temperature_bins(frames, bin_width=args.temp_bin)
    per_larva_all = per_larva_by_bin(frames, min_frames=args.min_frames_per_bin)

    coverage = bin_coverage(per_larva_all)
    coverage.to_csv(os.path.join(args.out_dir, "bin_coverage.csv"), index=False)

    per_larva, dropped_bins = filter_by_coverage(
        per_larva_all, min_folders=args.min_folders_per_bin
    )

    if not dropped_bins.empty:
        print(
            f"\n{len(dropped_bins)} temperature bin(s) dropped for thin coverage "
            f"(< {args.min_folders_per_bin} experiments):"
        )
        print(
            dropped_bins[["Temp_Bin", "N_Folders", "N_Groups", "N_Observations"]]
            .to_string(index=False)
        )

    per_larva_all.to_csv(
        os.path.join(args.out_dir, "per_larva_by_temperature_all_bins.csv"), index=False
    )
    per_larva.to_csv(os.path.join(args.out_dir, "per_larva_by_temperature.csv"), index=False)

    summary = group_summary(per_larva)
    summary.to_csv(os.path.join(args.out_dir, "group_summary_by_temperature.csv"), index=False)

    # --- 4. quality check: do the vehicle controls agree? -----------------
    controls = compare_controls(per_larva)
    if not controls.empty:
        controls.to_csv(os.path.join(args.out_dir, "controls_comparison.csv"), index=False)

        wide = controls.pivot_table(index="Range", columns="Group", values="Median")
        spread = (wide.max(axis=1) / wide.min(axis=1)).max()

        if spread > 1.25:
            print(
                "\nWARNING: the vehicle controls differ from each other by up to "
                f"{spread:.0%} in at least one temperature range. Curves from "
                "different datasets are therefore not directly comparable; rely "
                "on the per-drug figure and stats_vs_control.csv, which compare "
                "each treatment against its own vehicle."
            )
            print(wide.round(2).to_string())

    # --- 5. statistics ----------------------------------------------------
    stats = compare_to_controls(per_larva, control_map=control_map)
    if not stats.empty:
        stats.to_csv(os.path.join(args.out_dir, "stats_vs_control.csv"), index=False)

    # --- 6. figures -------------------------------------------------------
    if not args.no_plots:
        plot_temperature_response(
            summary, os.path.join(args.out_dir, "movement_vs_temperature.png")
        )
        plot_temperature_response_by_family(
            summary, os.path.join(args.out_dir, "movement_vs_temperature_by_drug.png")
        )
        if not stats.empty:
            plot_significance_heatmap(
                stats, os.path.join(args.out_dir, "significance_heatmap.png")
            )

    # --- 7. report --------------------------------------------------------
    report = {
        "inputs": infos,
        "signal": args.signal,
        "baseline_seconds": args.baseline_seconds,
        "temperature_bin_width": args.temp_bin,
        "control_map": control_map,
        "min_folders_per_bin": args.min_folders_per_bin,
        "bins_dropped_for_coverage": [float(b) for b in dropped_bins["Temp_Bin"]],
        "larvae_after_exclusion": int(
            per_larva.groupby(["Dataset", "Folder", "Droplet"]).ngroups
        ),
        "larvae_excluded": int(len(excluded)),
        "groups": sorted(per_larva["Group"].unique()),
        "temperature_range": [
            float(per_larva["Temp_Bin"].min()),
            float(per_larva["Temp_Bin"].max()),
        ],
        "significant_comparisons": int(stats["signif"].sum()) if not stats.empty else 0,
    }

    with open(os.path.join(args.out_dir, "run_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nLarvae analysed : {report['larvae_after_exclusion']}")
    print(f"Groups          : {', '.join(report['groups'])}")
    print(
        f"Temperature     : {report['temperature_range'][0]:.2f} – "
        f"{report['temperature_range'][1]:.2f} °C"
    )
    if not stats.empty:
        print(f"Significant bins: {report['significant_comparisons']} of {len(stats)}")
    print(f"\nWritten to {args.out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
