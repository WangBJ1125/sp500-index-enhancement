import numpy as np
import pandas as pd

from src.performance import (
    active_returns,
    annualized_return,
    annualized_volatility,
    hit_ratio,
    information_ratio,
    max_drawdown,
    performance_summary,
    sharpe_ratio,
    total_return,
    tracking_error,
)


def test_total_return_compounds_known_values():
    returns = pd.Series([0.10, -0.10, 0.20, None])

    result = total_return(returns)

    assert abs(result - 0.188) < 1e-12


def test_annualized_return_with_constant_returns():
    returns = pd.Series([0.01, 0.01])

    result = annualized_return(returns, periods_per_year=2)

    assert abs(result - 0.0201) < 1e-12


def test_annualized_return_empty_input_returns_nan():
    result = annualized_return(pd.Series(dtype=float))

    assert np.isnan(result)


def test_annualized_volatility_matches_pandas_std_scaling():
    returns = pd.Series([0.01, 0.03, -0.02])

    result = annualized_volatility(returns, periods_per_year=4)
    expected = returns.std() * np.sqrt(4)

    assert abs(result - expected) < 1e-12


def test_sharpe_ratio_zero_volatility_returns_nan():
    returns = pd.Series([0.01, 0.01, 0.01])

    result = sharpe_ratio(returns)

    assert np.isnan(result)


def test_active_returns_aligns_on_common_dates():
    portfolio = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    benchmark = pd.Series(
        [0.005, 0.010, 0.015],
        index=pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
    )

    result = active_returns(portfolio, benchmark)

    expected = pd.Series(
        [0.015, 0.020],
        index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
    )
    pd.testing.assert_series_equal(result, expected)


def test_tracking_error_and_information_ratio_use_active_returns():
    active = pd.Series([0.01, 0.03, -0.02])

    te = tracking_error(active, periods_per_year=4)
    ir = information_ratio(active, periods_per_year=4)
    expected_te = active.std() * np.sqrt(4)
    expected_ir = 4 * active.mean() / expected_te

    assert abs(te - expected_te) < 1e-12
    assert abs(ir - expected_ir) < 1e-12


def test_information_ratio_zero_tracking_error_returns_nan():
    active = pd.Series([0.01, 0.01, 0.01])

    result = information_ratio(active)

    assert np.isnan(result)


def test_max_drawdown_known_return_path():
    returns = pd.Series([0.10, -0.20, 0.05, -0.10])

    result = max_drawdown(returns)

    assert abs(result - (-0.244)) < 1e-12


def test_hit_ratio_drops_nan_values():
    returns = pd.Series([0.01, 0.00, -0.01, None])

    result = hit_ratio(returns)

    assert result == 1 / 3


def test_performance_summary_includes_active_metrics_with_benchmark():
    portfolio = pd.Series(
        [0.02, 0.01, -0.01],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    benchmark = pd.Series(
        [0.01, 0.00, -0.02],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    summary = performance_summary(portfolio, benchmark, periods_per_year=4)

    expected_metrics = {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "hit_ratio",
        "annualized_active_return",
        "tracking_error",
        "information_ratio",
    }
    assert expected_metrics.issubset(set(summary.index))
    assert abs(summary["annualized_active_return"] - 4 * 0.01) < 1e-12
