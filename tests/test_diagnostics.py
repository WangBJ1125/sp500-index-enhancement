import numpy as np
import pandas as pd

from src.diagnostics import (
    calculate_ic,
    compute_active_exposure_diagnostics,
    compute_forward_returns,
    ic_summary,
    run_factor_ic_analysis,
    run_factor_ic_subperiod_analysis,
    summarize_active_exposure,
)


def test_compute_forward_returns_uses_next_rebalance_prices():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0],
            "B": [50.0, 45.0, 54.0],
        },
        index=dates,
    )

    result = compute_forward_returns(prices, dates)

    expected = pd.DataFrame(
        {
            "A": [0.10, 0.10],
            "B": [-0.10, 0.20],
        },
        index=dates[:-1],
    )
    pd.testing.assert_frame_equal(result, expected)


def test_calculate_ic_positive_for_aligned_scores_and_forward_returns():
    dates = pd.to_datetime(["2024-01-31"])
    scores = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]}, index=dates)
    forward_returns = pd.DataFrame(
        {"A": [0.01], "B": [0.02], "C": [0.03]},
        index=dates,
    )

    result = calculate_ic(scores, forward_returns)

    assert np.isclose(result.iloc[0], 1.0)


def test_calculate_ic_negative_for_inverse_scores():
    dates = pd.to_datetime(["2024-01-31"])
    scores = pd.DataFrame({"A": [3.0], "B": [2.0], "C": [1.0]}, index=dates)
    forward_returns = pd.DataFrame(
        {"A": [0.01], "B": [0.02], "C": [0.03]},
        index=dates,
    )

    result = calculate_ic(scores, forward_returns)

    assert np.isclose(result.iloc[0], -1.0)


def test_calculate_ic_spearman_rank_correlation_works():
    dates = pd.to_datetime(["2024-01-31"])
    scores = pd.DataFrame({"A": [10.0], "B": [20.0], "C": [30.0]}, index=dates)
    forward_returns = pd.DataFrame(
        {"A": [0.03], "B": [0.01], "C": [0.02]},
        index=dates,
    )

    result = calculate_ic(scores, forward_returns, method="spearman")

    assert np.isclose(result.iloc[0], -0.5)


def test_calculate_ic_returns_nan_with_fewer_than_three_valid_observations():
    dates = pd.to_datetime(["2024-01-31"])
    scores = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [np.nan]}, index=dates)
    forward_returns = pd.DataFrame(
        {"A": [0.01], "B": [0.02], "C": [0.03]},
        index=dates,
    )

    result = calculate_ic(scores, forward_returns)

    assert pd.isna(result.iloc[0])


def test_ic_summary_handles_nan_values():
    ic_series = pd.Series([0.10, np.nan, -0.20, 0.30])

    result = ic_summary(ic_series)

    assert np.isclose(result["mean_ic"], (0.10 - 0.20 + 0.30) / 3)
    assert np.isclose(result["hit_rate"], 2 / 3)
    assert result["count"] == 3


def test_run_factor_ic_analysis_includes_factor_summary_columns():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0],
            "B": [100.0, 105.0, 103.0],
            "C": [100.0, 101.0, 111.0],
        },
        index=dates,
    )
    factor_dict = {
        "momentum": pd.DataFrame(
            {
                "A": [3.0, 3.0],
                "B": [2.0, 1.0],
                "C": [1.0, 2.0],
            },
            index=dates[:-1],
        )
    }

    result = run_factor_ic_analysis(factor_dict, prices, dates)

    assert result["factor"].tolist() == ["momentum"]
    assert {"mean_ic", "icir", "mean_rank_ic", "rank_icir"}.issubset(
        set(result.columns)
    )


def test_run_factor_ic_subperiod_analysis_returns_factor_period_rows():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0],
            "B": [100.0, 105.0, 103.0],
            "C": [100.0, 101.0, 111.0],
        },
        index=dates,
    )
    factor_dict = {
        "momentum": pd.DataFrame(
            {
                "A": [3.0, 3.0],
                "B": [2.0, 1.0],
                "C": [1.0, 2.0],
            },
            index=dates[:-1],
        )
    }
    periods = {
        "first": ("2024-01-01", "2024-01-31"),
        "empty": ("2025-01-01", "2025-12-31"),
    }

    result = run_factor_ic_subperiod_analysis(factor_dict, prices, dates, periods)

    assert result["period"].tolist() == ["first", "empty"]
    assert result.loc[result["period"] == "empty", "count"].iloc[0] == 0


def test_active_exposure_is_zero_when_weights_equal_benchmark():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    weights = pd.DataFrame(
        {
            "A": [0.60, 0.60],
            "B": [0.40, 0.40],
        },
        index=dates,
    )
    benchmark_weights = pd.Series({"A": 0.60, "B": 0.40})

    result = compute_active_exposure_diagnostics(weights, benchmark_weights)

    assert np.isclose(result["gross_active_exposure"].max(), 0.0)
    assert np.isclose(result["active_share"].max(), 0.0)
    assert np.isclose(result["net_active_weight"].abs().max(), 0.0)


def test_active_share_equals_half_gross_active_exposure():
    weights = pd.DataFrame(
        {"A": [0.70], "B": [0.30]},
        index=pd.to_datetime(["2024-01-31"]),
    )
    benchmark_weights = pd.Series({"A": 0.60, "B": 0.40})

    result = compute_active_exposure_diagnostics(weights, benchmark_weights)

    assert np.isclose(result.loc[weights.index[0], "gross_active_exposure"], 0.20)
    assert np.isclose(result.loc[weights.index[0], "active_share"], 0.10)


def test_active_exposure_max_absolute_active_weight_is_computed_correctly():
    weights = pd.DataFrame(
        {"A": [0.75], "B": [0.20], "C": [0.05]},
        index=pd.to_datetime(["2024-01-31"]),
    )
    benchmark_weights = pd.Series({"A": 0.50, "B": 0.30, "C": 0.20})

    result = compute_active_exposure_diagnostics(weights, benchmark_weights)

    row = result.loc[weights.index[0]]
    assert np.isclose(row["max_absolute_active_weight"], 0.25)
    assert row["top_overweight"] == "A"
    assert row["top_underweight"] == "C"


def test_active_exposure_aligns_and_normalizes_benchmark_weights():
    weights = pd.DataFrame(
        {"A": [0.60], "B": [0.40]},
        index=pd.to_datetime(["2024-01-31"]),
    )
    benchmark_weights = pd.Series({"A": 60.0, "B": 40.0, "C": 100.0})

    result = compute_active_exposure_diagnostics(weights, benchmark_weights)

    assert np.isclose(result.loc[weights.index[0], "gross_active_exposure"], 0.0)
    assert np.isclose(result.loc[weights.index[0], "active_share"], 0.0)


def test_summarize_active_exposure_returns_expected_summary_values():
    diagnostics = pd.DataFrame(
        {
            "gross_active_exposure": [0.20, 0.40],
            "active_share": [0.10, 0.20],
            "net_active_weight": [0.00, 0.02],
            "max_absolute_active_weight": [0.05, 0.15],
            "top_overweight": ["A", "B"],
            "top_underweight": ["C", "D"],
        }
    )

    result = summarize_active_exposure(diagnostics)

    assert np.isclose(result["average_gross_active_exposure"], 0.30)
    assert np.isclose(result["max_active_share"], 0.20)
    assert np.isclose(result["average_abs_net_active_weight"], 0.01)
