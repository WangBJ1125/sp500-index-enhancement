import pandas as pd
import pytest

from src.alpha_model import build_alpha_score
from src.signal_processing import combine_factors, zscore_cross_section


def _factor_scores():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    momentum = pd.DataFrame(
        {
            "A": [1.0, 3.0],
            "B": [2.0, 4.0],
            "C": [3.0, 5.0],
        },
        index=dates,
    )
    lowvol = pd.DataFrame(
        {
            "A": [3.0, 1.0],
            "B": [2.0, 2.0],
            "C": [1.0, 3.0],
        },
        index=dates,
    )
    reversal = pd.DataFrame(
        {
            "A": [0.0, 2.0],
            "B": [1.0, 1.0],
            "C": [2.0, 0.0],
        },
        index=dates,
    )
    return {
        "momentum": momentum,
        "lowvol": lowvol,
        "reversal": reversal,
    }


def test_momentum_only_returns_standardized_momentum_like_scores():
    factors = _factor_scores()

    result = build_alpha_score(factors, model="momentum_only")

    expected = zscore_cross_section(factors["momentum"])
    pd.testing.assert_frame_equal(result, expected)


def test_risk_adjusted_momentum_uses_momentum_and_lowvol():
    factors = _factor_scores()

    result = build_alpha_score(
        factors,
        model="risk_adjusted_momentum",
        lowvol_penalty_weight=0.5,
        standardize=False,
    )

    expected = (
        zscore_cross_section(factors["momentum"])
        + 0.5 * zscore_cross_section(factors["lowvol"])
    )
    pd.testing.assert_frame_equal(result, expected)


def test_momentum_reversal_uses_momentum_and_reversal():
    factors = _factor_scores()

    result = build_alpha_score(
        factors,
        model="momentum_reversal",
        reversal_weight=0.25,
        standardize=False,
    )

    expected = (
        zscore_cross_section(factors["momentum"])
        + 0.25 * zscore_cross_section(factors["reversal"])
    )
    pd.testing.assert_frame_equal(result, expected)


def test_custom_weighted_respects_weights():
    factors = _factor_scores()
    weights = {"momentum": 0.75, "reversal": 0.25}

    result = build_alpha_score(
        factors,
        model="custom_weighted",
        weights=weights,
        standardize=False,
    )

    expected = combine_factors(
        {"momentum": factors["momentum"], "reversal": factors["reversal"]},
        weights=weights,
        standardize_composite=False,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_output_preserves_date_index_and_ticker_columns():
    factors = _factor_scores()

    result = build_alpha_score(factors, model="equal_weight_composite")

    pd.testing.assert_index_equal(result.index, factors["momentum"].index)
    pd.testing.assert_index_equal(result.columns, factors["momentum"].columns)


def test_missing_required_factor_raises_clear_value_error():
    factors = _factor_scores()
    factors.pop("lowvol")

    with pytest.raises(ValueError, match="Missing required factor.*lowvol"):
        build_alpha_score(factors, model="risk_adjusted_momentum")


def test_unknown_alpha_model_raises_clear_value_error():
    factors = _factor_scores()

    with pytest.raises(ValueError, match="Unknown alpha model"):
        build_alpha_score(factors, model="not_a_model")
