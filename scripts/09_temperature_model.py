#!/usr/bin/env python
"""Step 9 — mixed model for the temperature response.

Step 7 tests every temperature bin separately. That is fine for describing
where a difference sits, but it spends its power on thirty-odd tests: an effect
that is consistent yet modest at each single temperature does not survive the
correction. This step asks the question once per group instead —
*does this group's response curve differ from the control's at all?*

Model::

    log(Movement_norm) ~ Group * spline(Temperature) + Folder
                         + (1 + Temperature | larva)

Run it on ``per_larva_by_temperature.csv``, the table step 7 writes.

Examples
--------
::

    python scripts/09_temperature_model.py \
        results/temperature/per_larva_by_temperature.csv \
        --control conventional --out-dir results/model

Check the specification against alternatives before trusting it::

    python scripts/09_temperature_model.py per_larva_by_temperature.csv \
        --control ETOH --compare-specifications
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from larvatracker.model import (
    compare_specifications,
    contrast_curve,
    fit_temperature_model,
    group_omnibus_tests,
    model_diagnostics,
    model_terms,
    predicted_curves,
    sample_sizes,
    sample_sizes_by_folder,
)
from larvatracker.plotting import plot_model_diagnostics, plot_model_fit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mixed model for movement across temperature.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("per_larva_csv", help="per_larva_by_temperature.csv from step 7")
    parser.add_argument(
        "--control", required=True, help="group every other group is compared against"
    )
    parser.add_argument("--out-dir", default="model_results", help="output directory")
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="restrict the model to these groups (the control is always kept)",
    )
    parser.add_argument(
        "--spline-df",
        type=int,
        default=4,
        help="degrees of freedom of the temperature spline; 0 fits a linear term",
    )
    parser.add_argument(
        "--no-random-slope",
        action="store_true",
        help="random intercept only; larvae then differ in level but not in steepness",
    )
    parser.add_argument(
        "--no-folder",
        action="store_true",
        help="leave the recording out of the model",
    )
    parser.add_argument(
        "--contrast-step",
        type=float,
        default=0.5,
        help="temperature spacing of the contrast table, in °C",
    )
    parser.add_argument(
        "--correction",
        default="holm",
        help="multiple-testing correction across the per-group omnibus tests",
    )
    parser.add_argument(
        "--compare-specifications",
        action="store_true",
        help="also fit competing spline and random-effect specifications and rank by AIC",
    )
    parser.add_argument("--no-plots", action="store_true", help="write tables only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    per_larva = pd.read_csv(args.per_larva_csv)

    if args.groups:
        keep = set(args.groups) | {args.control}
        per_larva = per_larva[per_larva["Group"].isin(keep)]

    if args.control not in set(per_larva["Group"]):
        print(
            f"error: control {args.control!r} not in the data; available: "
            f"{sorted(per_larva['Group'].unique())}",
            file=sys.stderr,
        )
        return 2

    if per_larva["Group"].nunique() < 2:
        print("error: need at least one group besides the control", file=sys.stderr)
        return 2

    # --- optional specification search -----------------------------------
    if args.compare_specifications:
        print("Comparing specifications (ML, lower AIC is better)\n")
        table = compare_specifications(per_larva, control=args.control)
        table.to_csv(os.path.join(args.out_dir, "specification_comparison.csv"), index=False)
        print(table.to_string(index=False))
        print()

    # --- fit --------------------------------------------------------------
    model = fit_temperature_model(
        per_larva,
        control=args.control,
        spline_df=args.spline_df,
        random_slope=not args.no_random_slope,
        include_folder=not args.no_folder,
    )

    print(f"Model : {model.formula}")
    if model.random_slope:
        print("        + random intercept and slope per larva")
    else:
        print("        + random intercept per larva")

    meta = model.metadata
    print(
        f"Data  : {meta['n_observations']} observations, {meta['n_larvae']} larvae, "
        f"{meta['n_folders']} recording(s), {meta['temperature_range'][0]:.2f}"
        f"–{meta['temperature_range'][1]:.2f} °C"
    )

    if not meta["converged"]:
        print(
            "\nWARNING: the fit did not converge. Treat every number below as "
            "unreliable — try --spline-df 3 or --no-random-slope.",
            file=sys.stderr,
        )

    if meta["non_positive_dropped"]:
        print(f"        {meta['non_positive_dropped']} non-positive value(s) dropped for the log")

    with open(os.path.join(args.out_dir, "model_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"{model.formula}\n\n{model.fit.summary()}\n")

    # --- sample sizes -----------------------------------------------------
    sizes = sample_sizes(model)
    sizes.to_csv(os.path.join(args.out_dir, "sample_sizes.csv"), index=False)

    print("\n=== Sample sizes ===")
    print(sizes.to_string(index=False))
    print(
        "\nN_Larvae is the unit of analysis — one random intercept per animal. "
        "N_Observations counts larva x temperature bin and is not a sample size."
    )

    by_folder = sample_sizes_by_folder(model)
    if not by_folder.empty:
        by_folder.to_csv(os.path.join(args.out_dir, "sample_sizes_by_folder.csv"), index=False)

        print("\n--- animals per recording ---")
        print(by_folder.to_string(index=False))

        per_group = by_folder.drop(columns=["Folder"])
        if (per_group == 0).any().any():
            missing = [
                f"{column} in {by_folder.loc[per_group[column] == 0, 'Folder'].tolist()}"
                for column in per_group.columns
                if (per_group[column] == 0).any()
            ]
            print(f"  WARNING: group absent from some recordings — {'; '.join(missing)}")

        smallest, largest = int(per_group.values.min()), int(per_group.values.max())
        if smallest and largest / smallest >= 3:
            print(
                f"  WARNING: group sizes per recording range from {smallest} to "
                f"{largest}; the recording term will carry part of the group effect."
            )

    # --- omnibus tests ----------------------------------------------------
    terms = model_terms(model)
    terms.to_csv(os.path.join(args.out_dir, "model_terms.csv"), index=False)

    print("\n=== Model terms ===")
    print(terms.to_string(index=False))
    print(
        "\nThe Group x spline row is the one that matters: it tests whether the "
        "*shape* of the temperature response depends on the group."
    )

    omnibus = group_omnibus_tests(model, correction=args.correction)
    omnibus.to_csv(os.path.join(args.out_dir, "group_tests.csv"), index=False)

    print(f"\n=== Each group versus {args.control}, whole range ({args.correction}-corrected) ===")
    print(
        omnibus[
            [
                "Group",
                "Control",
                "N_Larvae",
                "N_Larvae_Control",
                "N_Folders",
                "df",
                "Chi2",
                "p_raw",
                "p_adjusted",
                "signif",
            ]
        ].to_string(index=False)
    )

    # --- contrasts and predictions ---------------------------------------
    contrasts = contrast_curve(model, step=args.contrast_step)
    contrasts.to_csv(os.path.join(args.out_dir, "contrast_curve.csv"), index=False)

    predicted = predicted_curves(model)
    predicted.to_csv(os.path.join(args.out_dir, "predicted_curves.csv"), index=False)

    print("\n=== Strongest difference per group ===")
    for group, data in contrasts.groupby("Group"):
        row = data.loc[data["Log_Difference"].abs().idxmax()]
        significant = bool(omnibus.loc[omnibus["Group"] == group, "signif"].iloc[0])
        verdict = "group differs overall" if significant else "no overall difference"

        n_group = int(row["N_Larvae"])
        n_control = int(row["N_Larvae_Control"])

        print(
            f"  {group:<14} {row['Ratio']:.2f}x control at {row['Temperature_C']:.1f} °C "
            f"[{row['CI_low']:.2f}–{row['CI_high']:.2f}], n = {n_group} vs "
            f"{n_control} — {verdict}"
        )

    # --- diagnostics ------------------------------------------------------
    diagnostics = model_diagnostics(model)

    print("\n=== Residual diagnostics ===")
    print(f"  skew {diagnostics['residual_skew']:+.3f}, excess kurtosis "
          f"{diagnostics['residual_excess_kurtosis']:+.3f}")
    print(f"  heteroscedasticity corr(|resid|, fitted) = "
          f"{diagnostics['heteroscedasticity_corr']:+.3f}")

    if abs(diagnostics["heteroscedasticity_corr"]) > 0.3:
        print("  WARNING: residual spread depends clearly on the fitted value.")
    if abs(diagnostics["residual_skew"]) > 1.0:
        print("  WARNING: residuals are strongly skewed; the log scale may not be enough.")

    # --- figures ----------------------------------------------------------
    if not args.no_plots:
        plot_model_fit(
            predicted=predicted,
            contrasts=contrasts,
            out_path=os.path.join(args.out_dir, "model_fit.png"),
            control=args.control,
            observed=model.data,
        )
        plot_model_diagnostics(
            model.fit, os.path.join(args.out_dir, "model_diagnostics.png")
        )

    report = {
        "formula": model.formula,
        "random_effects": "intercept + slope per larva"
        if model.random_slope
        else "intercept per larva",
        "control": args.control,
        "groups": model.groups,
        "sample_sizes": sizes.to_dict(orient="records"),
        "correction": args.correction,
        **meta,
        "diagnostics": diagnostics,
    }

    with open(os.path.join(args.out_dir, "model_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nWritten to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
