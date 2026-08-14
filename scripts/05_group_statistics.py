#!/usr/bin/env python
"""Step 5 — baseline normalisation and group statistics on the pooled table.

Takes the combined summary written by ``04_batch_analysis.py``, expresses every
larva relative to its own baseline time bin, and tests whether the treatment
groups differ from the control over time.

What it does, in order:

1. keep only the requested time bins (the ``full`` rows are dropped);
2. optionally fold inconsistent group labels onto canonical names;
3. divide each measurement by its own baseline-bin value;
4. fit ``Freq_Hz_norm ~ Group * Time`` with a random intercept per larva;
5. compare every treatment against the control per bin (Mann-Whitney + FDR).

Examples
--------
::

    python scripts/05_group_statistics.py data/Combined_All_Folders_Summary.csv \
        --control ETOH --bins 1 2 3 4 --out-dir results/

With a label mapping for hand-typed group names::

    python scripts/05_group_statistics.py combined.csv --control ETOH \
        --group-map examples/group_map.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from larvatracker.scheme import normalise_group_labels
from larvatracker.stats import mixed_model, normalise_to_baseline, posthoc_vs_control


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baseline normalisation and group statistics on a pooled summary table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("summary_csv", help="combined summary table")
    parser.add_argument("--control", required=True, help="name of the control group")
    parser.add_argument(
        "--bins",
        nargs="+",
        default=["1", "2", "3", "4"],
        help="time bins to keep; the first one is used as the baseline",
    )
    parser.add_argument(
        "--bodypart",
        default=None,
        help="restrict the analysis to a single keypoint (default: all)",
    )
    parser.add_argument("--value-col", default="Freq_Hz", help="column to normalise and test")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter of the input table")
    parser.add_argument(
        "--group-map",
        default=None,
        help="JSON file mapping lower-cased raw labels to canonical group names",
    )
    parser.add_argument("--out-dir", default=".", help="where to write the result tables")
    parser.add_argument(
        "--no-model", action="store_true", help="skip the mixed model, run only the post-hoc tests"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.summary_csv, delimiter=args.delimiter)
    df["Time_Bin"] = df["Time_Bin"].astype(str)

    df = df[df["Time_Bin"].isin([str(b) for b in args.bins])]
    if args.bodypart:
        df = df[df["BodyPart"] == args.bodypart]

    if df.empty:
        print("No rows left after filtering — check --bins and --bodypart.", file=sys.stderr)
        return 1

    # --- optional label harmonisation -----------------------------------
    if args.group_map:
        with open(args.group_map, encoding="utf-8") as handle:
            mapping = json.load(handle)

        df["Group_raw"] = df["Group"]
        df["Group"] = normalise_group_labels(df["Group_raw"], mapping)

        unmapped = df["Group"].isna()
        if unmapped.any():
            print("Unmapped group labels (excluded):")
            print(df.loc[unmapped, "Group_raw"].value_counts().to_string())
            df = df[~unmapped].copy()

    print("Groups:", sorted(df["Group"].unique()))

    # --- normalisation ---------------------------------------------------
    baseline_bin = str(args.bins[0])
    normalised = normalise_to_baseline(
        df, value_col=args.value_col, baseline_bin=baseline_bin
    )
    norm_col = f"{args.value_col}_norm"

    norm_path = os.path.join(args.out_dir, "normalised_data.csv")
    normalised.to_csv(norm_path, index=False)
    print(f"\nNormalised data ({len(normalised)} rows) -> {norm_path}")

    # The baseline bin is 1.0 by construction and carries no information.
    response = normalised[normalised["Time_Bin"] != baseline_bin].copy()
    if response.empty:
        print("Only the baseline bin is present — nothing to test.", file=sys.stderr)
        return 1

    # --- mixed model -----------------------------------------------------
    if not args.no_model:
        try:
            result = mixed_model(response, value_col=norm_col)
            print("\n=== Mixed model ===")
            print(result.summary())
            print("\n=== Omnibus tests ===")
            print(result.wald_test_terms(skip_single=False).table)
        except Exception as exc:  # noqa: BLE001 - model failures should not kill the run
            print(f"\nMixed model could not be fitted: {exc}", file=sys.stderr)

    # --- post-hoc --------------------------------------------------------
    posthoc = posthoc_vs_control(response, control=args.control, value_col=norm_col)

    if posthoc.empty:
        print("\nNo post-hoc comparison had enough data.", file=sys.stderr)
        return 0

    posthoc_path = os.path.join(args.out_dir, "posthoc_vs_control.csv")
    posthoc.to_csv(posthoc_path, index=False)

    print(f"\n=== Post-hoc vs {args.control} (Mann-Whitney, BH-FDR) ===")
    print(posthoc.round(4).to_string(index=False))
    print(f"\nWritten to {posthoc_path}")

    significant = posthoc[posthoc["signif"]]
    print(f"\nSignificant after FDR correction (alpha = 0.05): {len(significant)} comparison(s)")
    if not significant.empty:
        print(
            significant[["Time_Bin", "Group", "median_treatment", "median_control", "p_fdr"]]
            .round(4)
            .to_string(index=False)
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
