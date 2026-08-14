"""Baseline normalisation and group statistics on the pooled summary table.

Absolute burst frequencies vary strongly between droplets and between
recording days, so raw group comparisons are dominated by that variability.
Each larva is therefore expressed relative to its own first time bin, which
turns the analysis into a within-subject comparison of the drug response.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SUBJECT_KEYS = ["Folder", "Droplet", "Group"]


def add_repetition_index(df: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    """Number repeated measurements that share the same subject key.

    A single droplet contributes one row per keypoint per time bin. Within a
    bin those rows are matched across bins by their running index, so keypoint
    ``a`` in bin 1 is paired with keypoint ``a`` in bin 2.
    """
    keys = keys or SUBJECT_KEYS
    out = df.copy()
    out["rep"] = out.groupby(keys + ["Time_Bin"]).cumcount()
    return out


def normalise_to_baseline(
    df: pd.DataFrame,
    value_col: str = "Freq_Hz",
    baseline_bin: str = "1",
    keys: list[str] | None = None,
    drop_zero_baseline: bool = True,
) -> pd.DataFrame:
    """Divide each measurement by its own baseline-bin value.

    Parameters
    ----------
    df:
        Long-format summary with ``Time_Bin`` and ``value_col`` columns.
    baseline_bin:
        Time bin used as the reference, normally the first one.
    drop_zero_baseline:
        Larvae that were completely immobile during the baseline bin have a
        baseline of 0 and would produce infinite ratios. They are dropped
        rather than clipped, because a ratio is undefined for them.

    Returns
    -------
    pd.DataFrame
        Input rows that had a usable baseline, with an extra column
        ``<value_col>_norm``.
    """
    keys = keys or SUBJECT_KEYS
    work = add_repetition_index(df, keys)

    index_cols = keys + ["rep"]
    baseline = (
        work[work["Time_Bin"].astype(str) == str(baseline_bin)]
        .set_index(index_cols)[value_col]
    )
    baseline = baseline[~baseline.index.duplicated(keep="first")]

    if drop_zero_baseline:
        n_zero = int((baseline == 0).sum())
        if n_zero:
            print(f"Excluding {n_zero} subject(s) with a zero baseline in bin {baseline_bin}.")
        baseline = baseline[baseline != 0]

    idx = pd.MultiIndex.from_frame(work[index_cols])
    keep = idx.isin(baseline.index)

    out = work[keep].copy()
    out[f"{value_col}_norm"] = (
        out[value_col].to_numpy() / pd.MultiIndex.from_frame(out[index_cols]).map(baseline).to_numpy()
    )
    return out


def mixed_model(
    df: pd.DataFrame,
    value_col: str = "Freq_Hz_norm",
    group_col: str = "Group",
    time_col: str = "Time_Bin",
    subject_cols: list[str] | None = None,
):
    """Fit ``value ~ Group * Time`` with a random intercept per subject.

    Repeated measurements of the same larva are not independent; the random
    intercept accounts for that. Returns the fitted statsmodels result, whose
    ``wald_test_terms()`` gives one omnibus p-value per factor.

    Note
    ----
    The baseline bin has zero variance after normalisation and must be excluded
    by the caller before fitting.
    """
    import statsmodels.formula.api as smf

    subject_cols = subject_cols or (SUBJECT_KEYS + ["rep"])

    data = df.dropna(subset=[value_col]).copy()
    data = data[np.isfinite(data[value_col])]
    data["subject"] = data[subject_cols].astype(str).agg("_".join, axis=1)

    model = smf.mixedlm(
        f"{value_col} ~ C({group_col}) * C({time_col})",
        data=data,
        groups=data["subject"],
    )
    return model.fit()


def posthoc_vs_control(
    df: pd.DataFrame,
    control: str,
    value_col: str = "Freq_Hz_norm",
    group_col: str = "Group",
    time_col: str = "Time_Bin",
    min_n: int = 3,
    fdr_method: str = "fdr_bh",
) -> pd.DataFrame:
    """Compare every treatment against the control, separately per time bin.

    Uses a two-sided Mann-Whitney U test — the normalised frequencies are
    ratios and not normally distributed — and corrects all comparisons together
    with Benjamini-Hochberg FDR.

    Returns
    -------
    pd.DataFrame
        One row per (time bin, group) with group sizes, medians, raw and
        FDR-corrected p-values, and a ``signif`` flag at alpha = 0.05.
    """
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    rows = []
    treatments = [g for g in df[group_col].dropna().unique() if g != control]

    for time_bin in sorted(df[time_col].dropna().unique(), key=str):
        subset = df[df[time_col] == time_bin]
        ctrl = subset.loc[subset[group_col] == control, value_col].dropna()

        if len(ctrl) < min_n:
            continue

        for treatment in treatments:
            values = subset.loc[subset[group_col] == treatment, value_col].dropna()
            if len(values) < min_n:
                continue

            _, p_value = mannwhitneyu(values, ctrl, alternative="two-sided")
            rows.append(
                {
                    "Time_Bin": time_bin,
                    "Group": treatment,
                    "n_treatment": len(values),
                    "n_control": len(ctrl),
                    "median_treatment": values.median(),
                    "median_control": ctrl.median(),
                    "p_raw": p_value,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["p_fdr"] = multipletests(result["p_raw"], method=fdr_method)[1]
    result["signif"] = result["p_fdr"] < 0.05

    return result.sort_values(["Time_Bin", "Group"]).reset_index(drop=True)
