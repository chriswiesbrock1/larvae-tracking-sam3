"""A mixed model for the temperature response.

The per-bin tests in :mod:`larvatracker.temperature` ask a separate question at
every temperature and then pay for it: with thirty-odd bins, an effect that is
consistent but modest at each individual temperature does not survive multiple
testing, even though the pattern across bins is obvious. This module asks the
question once instead — *does this group's response curve differ from the
control's at all?* — which is one test per group rather than one per bin.

Model
-----
::

    log(Movement_norm) ~ Group * spline(Temperature) + Folder
                         + (1 + Temperature | larva)

**Log response.** The normalised movement is a ratio, bounded below by zero and
right-skewed; on the log scale the residuals are near-symmetric and group
effects become multiplicative, which is how a ratio should behave. Model
contrasts are therefore reported as ratios.

**Spline in temperature.** The response is flat, then rises steeply, then
plateaus. A linear term is badly misspecified — on the data this was built for
it costs over 200 AIC compared with a four-degree-of-freedom spline. The
``Group x spline`` interaction is what carries the biological question: it asks
whether the *shape* of the response differs, not just its overall level.

**Folder as a fixed effect.** Recording day matters, but with a handful of
recordings there are too few levels to estimate a variance component reliably,
so it enters as a fixed effect. It cancels out of any within-temperature group
contrast anyway.

**Random intercept and slope per larva.** Each larva contributes one value per
temperature bin, and those are not independent. Animals differ both in overall
activity and in how steeply they respond, so both terms are needed — dropping
the slope costs about 150 AIC here.

Temperature is centred and scaled internally. That is not cosmetic: with a raw
20-40 scale alongside a spline basis, the random-slope model is badly
conditioned and the fit fails outright with a singular matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SUBJECT_KEYS = ("Dataset", "Folder", "Droplet")


@dataclass
class TemperatureModel:
    """A fitted temperature-response model and everything needed to query it.

    ``design_info`` is kept deliberately: predicting at a new temperature
    requires the *original* spline basis. Rebuilding the basis from the new
    rows silently places the knots somewhere else, which produces contrasts
    that look plausible but are constant in temperature.
    """

    fit: object
    formula: str
    design_info: object
    data: pd.DataFrame
    control: str
    groups: list[str]
    temp_mean: float
    temp_std: float
    spline_df: int
    random_slope: bool
    metadata: dict = field(default_factory=dict)

    @property
    def param_names(self) -> list[str]:
        return list(self.fit.fe_params.index)

    @property
    def beta(self) -> np.ndarray:
        return self.fit.fe_params.values

    @property
    def vcov(self) -> np.ndarray:
        """Covariance of the fixed effects only."""
        n = len(self.beta)
        return np.asarray(self.fit.cov_params())[:n, :n]

    def scale_temperature(self, celsius) -> np.ndarray:
        return (np.asarray(celsius, dtype=float) - self.temp_mean) / self.temp_std


def prepare_model_frame(
    per_larva: pd.DataFrame,
    control: str,
    value_col: str = "Movement_norm",
    subject_keys: tuple[str, ...] = SUBJECT_KEYS,
) -> pd.DataFrame:
    """Build the modelling frame: log response, scaled temperature, subject id.

    The control group is made the first category so that every model
    coefficient is read against it.
    """
    keys = [k for k in subject_keys if k in per_larva.columns]

    df = per_larva.dropna(subset=[value_col, "Temp_Bin"]).copy()

    non_positive = int((df[value_col] <= 0).sum())
    if non_positive:
        # A zero would make the log undefined. Dropping is honest; adding a
        # constant would invent a value and shift every ratio.
        df = df[df[value_col] > 0]

    if control not in set(df["Group"]):
        raise ValueError(
            f"control group {control!r} not present; available: {sorted(df['Group'].unique())}"
        )

    others = sorted(g for g in df["Group"].unique() if g != control)
    df["Group"] = pd.Categorical(df["Group"], categories=[control] + others)

    df["larva"] = df[keys].astype(str).agg("_".join, axis=1)
    df["response"] = np.log(df[value_col])

    df.attrs["non_positive_dropped"] = non_positive
    return df


def fit_temperature_model(
    per_larva: pd.DataFrame,
    control: str,
    spline_df: int = 4,
    random_slope: bool = True,
    include_folder: bool = True,
    value_col: str = "Movement_norm",
    reml: bool = True,
) -> TemperatureModel:
    """Fit the mixed model.

    Parameters
    ----------
    spline_df:
        Degrees of freedom of the B-spline in temperature. Four is enough for a
        flat-rise-plateau shape; more gives the curve licence to chase noise.
    random_slope:
        Allow larvae to differ in how steeply they respond, not only in level.
    include_folder:
        Add the recording as a fixed effect. Only meaningful with more than one.
    """
    import statsmodels.formula.api as smf

    df = prepare_model_frame(per_larva, control, value_col=value_col)

    temp_mean = float(df["Temp_Bin"].mean())
    temp_std = float(df["Temp_Bin"].std())
    df["temp_z"] = (df["Temp_Bin"] - temp_mean) / temp_std

    n_folders = df["Folder"].nunique() if "Folder" in df.columns else 1
    use_folder = include_folder and n_folders > 1

    terms = [f"Group * bs(temp_z, df={spline_df})"]
    if use_folder:
        terms.append("Folder")

    formula = "response ~ " + " + ".join(terms)
    re_formula = "~temp_z" if random_slope else None

    model = smf.mixedlm(formula, df, groups=df["larva"], re_formula=re_formula)
    fit = model.fit(reml=reml)

    groups = [g for g in df["Group"].cat.categories if g != control]

    return TemperatureModel(
        fit=fit,
        formula=formula,
        design_info=model.data.design_info,
        data=df,
        control=control,
        groups=groups,
        temp_mean=temp_mean,
        temp_std=temp_std,
        spline_df=spline_df,
        random_slope=random_slope,
        metadata={
            "n_observations": int(fit.nobs),
            "n_larvae": int(df["larva"].nunique()),
            "n_folders": int(n_folders),
            "folder_in_model": bool(use_folder),
            "converged": bool(fit.converged),
            "reml": reml,
            "log_likelihood": float(fit.llf),
            "non_positive_dropped": int(df.attrs.get("non_positive_dropped", 0)),
            "temperature_range": [float(df["Temp_Bin"].min()), float(df["Temp_Bin"].max())],
        },
    )


def _design_row(model: TemperatureModel, group: str, celsius: float, folder=None) -> np.ndarray:
    """One row of the fixed-effects design matrix, using the fitted basis."""
    import patsy

    row = {
        "Group": pd.Categorical([group], categories=model.data["Group"].cat.categories),
        "temp_z": [float(model.scale_temperature(celsius))],
    }
    if "Folder" in model.data.columns:
        row["Folder"] = [folder if folder is not None else model.data["Folder"].iloc[0]]

    matrix = patsy.build_design_matrices([model.design_info], pd.DataFrame(row))[0]
    return np.asarray(matrix).ravel()


def _marginal_row(model: TemperatureModel, group: str, celsius: float) -> np.ndarray:
    """Design row averaged over the recordings.

    Predicting with one arbitrary recording would report that recording's
    offset as if it were the population value. Averaging the folder columns
    gives a marginal prediction instead. Group *contrasts* are unaffected
    either way, because the folder terms cancel.
    """
    if "Folder" not in model.data.columns:
        return _design_row(model, group, celsius)

    rows = [_design_row(model, group, celsius, folder=f) for f in model.data["Folder"].unique()]
    return np.mean(rows, axis=0)


def model_terms(model: TemperatureModel) -> pd.DataFrame:
    """Omnibus Wald test for every term in the model.

    The row that answers the biological question is the ``Group:bs(...)``
    interaction: it tests whether the shape of the temperature response depends
    on the group.
    """
    table = model.fit.wald_test_terms(skip_single=False).table.copy()
    table.index.name = "Term"

    table = table.reset_index()
    table.columns = [str(c) for c in table.columns]

    # statsmodels returns 1x1 arrays here; flatten so the CSV is readable.
    for column in table.columns:
        if column != "Term":
            table[column] = [float(np.asarray(v).ravel()[0]) for v in table[column]]

    return table


def group_omnibus_tests(
    model: TemperatureModel,
    correction: str = "holm",
) -> pd.DataFrame:
    """Test each group against the control across the whole temperature range.

    Every coefficient involving the group — its main effect and all its spline
    interactions — is tested jointly. That is one test per group instead of one
    per temperature bin, which is where the per-bin analysis loses its power:
    the pattern is consistent, but no single bin carries it alone.

    This is the inferential result. The contrast curve that follows localises a
    difference the omnibus has already established; used on its own it would
    reintroduce the multiplicity problem.
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    names = model.param_names
    beta, vcov = model.beta, model.vcov

    rows = []
    for group in model.groups:
        token = f"Group[T.{group}]"
        idx = [i for i, name in enumerate(names) if token in name]

        if not idx:
            continue

        contrast = np.zeros((len(idx), len(beta)))
        for row_index, column_index in enumerate(idx):
            contrast[row_index, column_index] = 1.0

        difference = contrast @ beta
        covariance = contrast @ vcov @ contrast.T

        statistic = float(difference @ np.linalg.solve(covariance, difference))
        p_value = float(stats.chi2.sf(statistic, len(idx)))

        rows.append(
            {
                "Group": group,
                "Control": model.control,
                "df": len(idx),
                "Chi2": statistic,
                "p_raw": p_value,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["p_adjusted"] = multipletests(result["p_raw"], method=correction)[1]
    result["signif"] = result["p_adjusted"] < 0.05
    result.attrs["correction"] = correction

    return result


def contrast_curve(
    model: TemperatureModel,
    temperatures=None,
    step: float = 0.5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Group-versus-control ratio at each temperature, with confidence interval.

    Because the response is on the log scale, the contrast exponentiates to a
    ratio: 0.75 means the group moved three quarters as much as the control at
    that temperature.

    The intervals are **pointwise**, not simultaneous, and neighbouring
    temperatures are strongly correlated — the curve says *where* a difference
    sits, it is not a set of independent tests. Treat
    :func:`group_omnibus_tests` as the inferential result.
    """
    from scipy import stats

    if temperatures is None:
        low, high = model.metadata["temperature_range"]
        temperatures = np.arange(low, high + step / 2, step)

    beta, vcov = model.beta, model.vcov
    z_critical = stats.norm.ppf(1 - alpha / 2)

    rows = []
    for group in model.groups:
        for celsius in np.asarray(temperatures, dtype=float):
            contrast = _design_row(model, group, celsius) - _design_row(
                model, model.control, celsius
            )

            difference = float(contrast @ beta)
            standard_error = float(np.sqrt(contrast @ vcov @ contrast))

            rows.append(
                {
                    "Group": group,
                    "Control": model.control,
                    "Temperature_C": celsius,
                    "Ratio": np.exp(difference),
                    "CI_low": np.exp(difference - z_critical * standard_error),
                    "CI_high": np.exp(difference + z_critical * standard_error),
                    "Log_Difference": difference,
                    "SE": standard_error,
                    "p_pointwise": 2 * float(stats.norm.sf(abs(difference / standard_error))),
                }
            )

    return pd.DataFrame(rows)


def predicted_curves(
    model: TemperatureModel,
    temperatures=None,
    step: float = 0.25,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Model-predicted movement per group across temperature, back on the ratio scale.

    Averaged over recordings, so the curve is a population prediction rather
    than one recording's.
    """
    from scipy import stats

    if temperatures is None:
        low, high = model.metadata["temperature_range"]
        temperatures = np.arange(low, high + step / 2, step)

    beta, vcov = model.beta, model.vcov
    z_critical = stats.norm.ppf(1 - alpha / 2)

    rows = []
    for group in [model.control] + model.groups:
        for celsius in np.asarray(temperatures, dtype=float):
            design = _marginal_row(model, group, celsius)

            prediction = float(design @ beta)
            standard_error = float(np.sqrt(design @ vcov @ design))

            rows.append(
                {
                    "Group": group,
                    "Temperature_C": celsius,
                    "Predicted": np.exp(prediction),
                    "CI_low": np.exp(prediction - z_critical * standard_error),
                    "CI_high": np.exp(prediction + z_critical * standard_error),
                }
            )

    return pd.DataFrame(rows)


def model_diagnostics(model: TemperatureModel) -> dict:
    """Residual checks, reported rather than tested into a verdict.

    A normality test on thousands of residuals rejects on deviations far too
    small to matter, so the skew and excess kurtosis are given directly. The
    heteroscedasticity number is the correlation between absolute residual and
    fitted value; near zero is what one hopes for.
    """
    from scipy import stats

    residuals = pd.Series(np.asarray(model.fit.resid))
    fitted = np.asarray(model.fit.fittedvalues)

    sample = residuals.sample(min(len(residuals), 1500), random_state=0)

    return {
        "converged": bool(model.fit.converged),
        "residual_skew": float(residuals.skew()),
        "residual_excess_kurtosis": float(residuals.kurt()),
        "shapiro_p_on_sample": float(stats.shapiro(sample)[1]),
        "heteroscedasticity_corr": float(np.corrcoef(np.abs(residuals), fitted)[0, 1]),
        "residual_sd": float(residuals.std()),
    }


def compare_specifications(
    per_larva: pd.DataFrame,
    control: str,
    spline_dfs=(0, 3, 4, 5, 6),
    value_col: str = "Movement_norm",
) -> pd.DataFrame:
    """Fit competing specifications and rank them by AIC.

    ``spline_df=0`` stands for a plain linear term in temperature, so the table
    shows what the spline actually buys. Fitted with ML, not REML, because REML
    likelihoods of models with different fixed effects are not comparable.
    """
    import statsmodels.formula.api as smf

    df = prepare_model_frame(per_larva, control, value_col=value_col)
    df["temp_z"] = (df["Temp_Bin"] - df["Temp_Bin"].mean()) / df["Temp_Bin"].std()

    use_folder = "Folder" in df.columns and df["Folder"].nunique() > 1
    folder_term = " + Folder" if use_folder else ""

    rows = []
    for spline_df in spline_dfs:
        basis = "temp_z" if spline_df == 0 else f"bs(temp_z, df={spline_df})"

        for random_slope in (False, True):
            formula = f"response ~ Group * {basis}{folder_term}"

            try:
                fit = smf.mixedlm(
                    formula,
                    df,
                    groups=df["larva"],
                    re_formula="~temp_z" if random_slope else None,
                ).fit(reml=False)
            except Exception as exc:  # noqa: BLE001 - a failed spec is a result
                rows.append(
                    {
                        "spline_df": spline_df,
                        "random_slope": random_slope,
                        "AIC": np.nan,
                        "BIC": np.nan,
                        "converged": False,
                        "note": type(exc).__name__,
                    }
                )
                continue

            rows.append(
                {
                    "spline_df": spline_df,
                    "random_slope": random_slope,
                    "AIC": float(fit.aic),
                    "BIC": float(fit.bic),
                    "converged": bool(fit.converged),
                    "note": "",
                }
            )

    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
