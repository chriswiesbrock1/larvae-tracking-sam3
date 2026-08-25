"""Tests for the temperature-response mixed model.

The model is checked against synthetic data with a known effect: if it cannot
recover an effect that was put in on purpose, or if it invents one that is not
there, nothing it says about real data is worth reading.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from larvatracker.model import (  # noqa: E402
    compare_specifications,
    contrast_curve,
    fit_temperature_model,
    group_omnibus_tests,
    model_diagnostics,
    model_terms,
    predicted_curves,
    prepare_model_frame,
)


def make_dataset(
    n_per_group=20,
    n_folders=3,
    effect=0.75,
    effect_above=31.0,
    noise=0.20,
    seed=0,
) -> pd.DataFrame:
    """Larvae following a sigmoid temperature response, with a known effect.

    ``effect`` is a multiplicative change applied to the treated group above
    ``effect_above`` — exactly the shape the model is meant to detect, and
    exactly what the contrast curve should report back.
    """
    rng = np.random.default_rng(seed)
    temperatures = np.arange(24.0, 39.5, 0.5)
    rows = []

    for folder in range(n_folders):
        folder_offset = rng.normal(0, 0.10)

        for group in ("control", "treated"):
            for animal in range(n_per_group):
                level = rng.normal(0, 0.25)
                slope = rng.normal(0, 0.05)

                for temperature in temperatures:
                    base = np.log(1.0 + 3.0 / (1 + np.exp(-(temperature - 32.0))))
                    treated = group == "treated" and temperature > effect_above

                    value = (
                        base
                        + (np.log(effect) if treated else 0.0)
                        + level
                        + slope * (temperature - 31.0)
                        + folder_offset
                        + rng.normal(0, noise)
                    )

                    rows.append(
                        {
                            "Dataset": "D",
                            "Folder": f"F{folder}",
                            "Droplet": f"{group}{animal}",
                            "Group": group,
                            "Temp_Bin": temperature,
                            "Movement_norm": float(np.exp(value)),
                        }
                    )

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_temperature_model(make_dataset(seed=1), control="control")


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------

def test_control_becomes_the_reference_category():
    """Every coefficient has to be read against the control, so it must be first."""
    frame = prepare_model_frame(make_dataset(n_per_group=3, seed=0), control="treated")

    assert list(frame["Group"].cat.categories)[0] == "treated"


def test_response_is_logged_and_subject_id_is_built():
    frame = prepare_model_frame(make_dataset(n_per_group=3, seed=0), control="control")

    assert np.allclose(frame["response"], np.log(frame["Movement_norm"]))
    assert frame["larva"].nunique() == 3 * 2 * 3   # folders x groups x animals


def test_non_positive_values_are_dropped_not_shifted():
    """Adding a constant to rescue a zero would shift every ratio in the model."""
    data = make_dataset(n_per_group=3, seed=0)
    data.loc[data.index[:5], "Movement_norm"] = 0.0

    frame = prepare_model_frame(data, control="control")

    assert (frame["Movement_norm"] > 0).all()
    assert frame.attrs["non_positive_dropped"] == 5
    assert np.isfinite(frame["response"]).all()


def test_unknown_control_raises():
    with pytest.raises(ValueError):
        prepare_model_frame(make_dataset(n_per_group=3), control="nonexistent")


# ---------------------------------------------------------------------------
# Recovering a known effect
# ---------------------------------------------------------------------------

def test_model_fits_and_reports_its_structure(fitted):
    assert fitted.metadata["converged"]
    assert fitted.metadata["n_larvae"] == 3 * 2 * 20
    assert fitted.control == "control"
    assert fitted.groups == ["treated"]


def test_known_effect_is_detected(fitted):
    tests = group_omnibus_tests(fitted)

    assert len(tests) == 1
    assert tests.iloc[0]["Group"] == "treated"
    assert tests.iloc[0]["signif"]


def test_contrast_recovers_the_true_ratio(fitted):
    """0.75 was put in above 31 °C and 1.0 below; both must come back out."""
    curve = contrast_curve(fitted, step=1.0)

    warm = curve[(curve["Temperature_C"] >= 33) & (curve["Temperature_C"] <= 37)]
    cold = curve[curve["Temperature_C"] <= 29]

    assert warm["Ratio"].mean() == pytest.approx(0.75, abs=0.06)
    assert cold["Ratio"].mean() == pytest.approx(1.00, abs=0.10)


def test_contrast_varies_with_temperature(fitted):
    """A constant contrast means the spline basis was rebuilt with wrong knots."""
    curve = contrast_curve(fitted, step=1.0)
    ratios = curve[curve["Group"] == "treated"]["Ratio"]

    assert ratios.std() > 0.02


def test_confidence_interval_brackets_the_estimate(fitted):
    curve = contrast_curve(fitted, step=2.0)

    assert (curve["CI_low"] < curve["Ratio"]).all()
    assert (curve["Ratio"] < curve["CI_high"]).all()


def test_predicted_curves_cover_every_group(fitted):
    predicted = predicted_curves(fitted, step=1.0)

    assert set(predicted["Group"]) == {"control", "treated"}
    assert (predicted["Predicted"] > 0).all()
    # The response rises with temperature, so the warm end must sit higher.
    control = predicted[predicted["Group"] == "control"].sort_values("Temperature_C")
    assert control["Predicted"].iloc[-1] > control["Predicted"].iloc[0]


# ---------------------------------------------------------------------------
# Not inventing effects
# ---------------------------------------------------------------------------

def test_no_effect_is_not_reported():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_temperature_model(make_dataset(effect=1.0, seed=2), control="control")

    assert not group_omnibus_tests(model).iloc[0]["signif"]


@pytest.mark.slow
def test_false_positive_rate_is_calibrated():
    """Across null datasets, significance should appear at roughly alpha."""
    positives = 0

    for seed in range(12):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fit_temperature_model(
                make_dataset(n_per_group=12, effect=1.0, seed=200 + seed), control="control"
            )
        positives += int(group_omnibus_tests(model).iloc[0]["signif"])

    # 12 draws at alpha = 0.05: seeing more than two would be a bad sign.
    assert positives <= 2


# ---------------------------------------------------------------------------
# Model terms, diagnostics, specification search
# ---------------------------------------------------------------------------

def test_interaction_term_is_present_and_significant(fitted):
    """The Group x spline term carries the question the model exists to answer."""
    terms = model_terms(fitted)
    interaction = terms[terms["Term"].str.contains("Group:")]

    assert len(interaction) == 1
    assert float(interaction.iloc[0]["pvalue"]) < 0.05


def test_diagnostics_report_plain_numbers(fitted):
    diagnostics = model_diagnostics(fitted)

    assert diagnostics["converged"]
    assert abs(diagnostics["residual_skew"]) < 1.0
    assert abs(diagnostics["heteroscedasticity_corr"]) < 0.5


@pytest.mark.slow
def test_spline_beats_a_linear_term_on_a_curved_response():
    """The response is sigmoid; a linear term in temperature must lose."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = compare_specifications(
            make_dataset(n_per_group=10, seed=5), control="control", spline_dfs=(0, 4)
        )

    usable = table[table["converged"]]
    best = usable.iloc[0]

    assert best["spline_df"] == 4


# ---------------------------------------------------------------------------
# Sample sizes
# ---------------------------------------------------------------------------

def test_sample_sizes_count_animals_not_rows(fitted):
    """The row count is larva x temperature bin, which is not a sample size."""
    from larvatracker.model import sample_sizes

    sizes = sample_sizes(fitted).set_index("Group")

    assert sizes.loc["control", "N_Larvae"] == 3 * 20     # folders x animals
    assert sizes.loc["treated", "N_Larvae"] == 3 * 20
    assert sizes.loc["control", "N_Folders"] == 3

    # Observations are far more numerous — that is exactly the trap.
    assert sizes.loc["control", "N_Observations"] > 10 * sizes.loc["control", "N_Larvae"]


def test_sample_sizes_mark_the_control(fitted):
    from larvatracker.model import sample_sizes

    sizes = sample_sizes(fitted).set_index("Group")

    assert sizes.loc["control", "Role"] == "control"
    assert sizes.loc["treated", "Role"] == "treatment"


def test_sample_sizes_by_folder_expose_imbalance():
    """A group missing from one recording must be visible, not averaged away."""
    import warnings

    from larvatracker.model import sample_sizes_by_folder

    data = make_dataset(n_per_group=8, n_folders=3, seed=11)
    # Drop the treated animals from one recording entirely.
    data = data[~((data["Folder"] == "F1") & (data["Group"] == "treated"))]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_temperature_model(data, control="control")

    table = sample_sizes_by_folder(model).set_index("Folder")

    assert table.loc["F1", "treated"] == 0
    assert table.loc["F0", "treated"] == 8
    assert table.loc["F1", "control"] == 8


def test_group_tests_carry_the_sample_sizes(fitted):
    """A p-value without an n is not reportable."""
    tests = group_omnibus_tests(fitted)
    row = tests.iloc[0]

    assert row["N_Larvae"] == 60
    assert row["N_Larvae_Control"] == 60
    assert row["N_Folders"] == 3
    assert row["N_Observations"] > 0


def test_contrast_curve_carries_the_sample_sizes(fitted):
    curve = contrast_curve(fitted, step=2.0)

    assert (curve["N_Larvae"] == 60).all()
    assert (curve["N_Larvae_Control"] == 60).all()
