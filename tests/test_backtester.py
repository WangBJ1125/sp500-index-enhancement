import pandas as pd

from src.backtester import BacktestResult, run_backtest


def _synthetic_inputs():
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )
    stock_returns = pd.DataFrame(
        {
            "A": [1.00, 0.10, 0.00, 0.00],
            "B": [0.00, 0.00, 0.10, 0.10],
        },
        index=dates,
    )
    benchmark_returns = pd.Series(
        [0.01, 0.02, 0.03, 0.04],
        index=dates,
        name="SPY",
    )
    scores = pd.DataFrame(
        {
            "A": [1.0, -1.0],
            "B": [-1.0, 1.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )
    rebalance_dates = pd.to_datetime(["2024-01-02", "2024-01-04"])

    return stock_returns, benchmark_returns, scores, rebalance_dates


def test_run_backtest_returns_result_with_non_empty_series():
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()

    result = run_backtest(
        stock_returns,
        benchmark_returns,
        scores,
        rebalance_dates,
        transaction_cost_bps=0.0,
    )

    assert isinstance(result, BacktestResult)
    assert not result.portfolio_returns_gross.empty
    assert not result.portfolio_returns_net.empty
    assert not result.benchmark_returns.empty


def test_weights_are_not_applied_on_same_rebalance_date_with_one_day_lag():
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()

    result = run_backtest(
        stock_returns,
        benchmark_returns,
        scores,
        rebalance_dates,
        transaction_cost_bps=0.0,
        trade_lag_days=1,
    )

    assert pd.Timestamp("2024-01-02") not in result.portfolio_returns_gross.index
    assert pd.Timestamp("2024-01-03") in result.portfolio_returns_gross.index


def test_weights_are_applied_from_next_trading_day_with_one_day_lag():
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()

    result = run_backtest(
        stock_returns,
        benchmark_returns,
        scores,
        rebalance_dates,
        transaction_cost_bps=0.0,
        trade_lag_days=1,
    )

    first_active_weights = result.weights.loc[pd.Timestamp("2024-01-03")]

    assert abs(first_active_weights.loc["A"] - 0.6) < 1e-12
    assert abs(first_active_weights.loc["B"] - 0.4) < 1e-12
    assert abs(result.portfolio_returns_gross.loc["2024-01-03"] - 0.06) < 1e-12


def test_transaction_cost_reduces_net_return_only_on_cost_date():
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()

    result = run_backtest(
        stock_returns,
        benchmark_returns,
        scores,
        rebalance_dates,
        transaction_cost_bps=10.0,
        trade_lag_days=1,
    )

    first_cost_date = pd.Timestamp("2024-01-03")
    no_cost_date = pd.Timestamp("2024-01-04")

    assert result.transaction_costs.loc[first_cost_date] == 0.001
    assert (
        result.portfolio_returns_net.loc[first_cost_date]
        == result.portfolio_returns_gross.loc[first_cost_date] - 0.001
    )
    assert (
        result.portfolio_returns_net.loc[no_cost_date]
        == result.portfolio_returns_gross.loc[no_cost_date]
    )


def test_turnover_is_computed_against_previous_holdings():
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()

    result = run_backtest(
        stock_returns,
        benchmark_returns,
        scores,
        rebalance_dates,
        transaction_cost_bps=0.0,
    )

    assert abs(result.turnover.loc["2024-01-03"] - 0.5) < 1e-12
    assert abs(result.turnover.loc["2024-01-05"] - 0.2) < 1e-12


def test_portfolio_returns_are_weighted_sums_of_future_simple_returns():
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()

    result = run_backtest(
        stock_returns,
        benchmark_returns,
        scores,
        rebalance_dates,
        transaction_cost_bps=0.0,
    )

    expected = pd.Series(
        [0.06, 0.04, 0.06],
        index=pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    pd.testing.assert_series_equal(result.portfolio_returns_gross, expected)


def test_benchmark_returns_are_aligned_to_portfolio_return_dates():
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()

    result = run_backtest(
        stock_returns,
        benchmark_returns,
        scores,
        rebalance_dates,
        transaction_cost_bps=0.0,
    )

    expected = benchmark_returns.loc[result.portfolio_returns_gross.index]
    pd.testing.assert_series_equal(result.benchmark_returns, expected)


def test_external_benchmark_weights_are_used_as_starting_weights():
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()
    benchmark_weights = pd.Series({"A": 0.8, "B": 0.2})

    result = run_backtest(
        stock_returns,
        benchmark_returns,
        scores,
        rebalance_dates,
        benchmark_weights=benchmark_weights,
        active_budget=0.0,
        transaction_cost_bps=0.0,
    )

    first_active_weights = result.weights.loc[pd.Timestamp("2024-01-03")]

    assert abs(first_active_weights.loc["A"] - 0.8) < 1e-12
    assert abs(first_active_weights.loc["B"] - 0.2) < 1e-12


def test_external_benchmark_weights_are_renormalized_over_valid_score_tickers():
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()
    benchmark_weights = pd.Series({"A": 0.2, "B": 0.3, "C": 0.5})

    result = run_backtest(
        stock_returns,
        benchmark_returns,
        scores,
        rebalance_dates,
        benchmark_weights=benchmark_weights,
        active_budget=0.0,
        transaction_cost_bps=0.0,
    )

    first_active_weights = result.weights.loc[pd.Timestamp("2024-01-03")]

    assert abs(first_active_weights.loc["A"] - 0.4) < 1e-12
    assert abs(first_active_weights.loc["B"] - 0.6) < 1e-12
    assert "C" not in first_active_weights.index


def test_missing_external_benchmark_weights_are_handled_safely():
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()
    benchmark_weights = pd.Series({"A": 0.8, "C": 0.2})

    result = run_backtest(
        stock_returns,
        benchmark_returns,
        scores,
        rebalance_dates,
        benchmark_weights=benchmark_weights,
        active_budget=0.0,
        transaction_cost_bps=0.0,
    )

    first_active_weights = result.weights.loc[pd.Timestamp("2024-01-03")]

    assert abs(first_active_weights.loc["A"] - 1.0) < 1e-12
    assert abs(first_active_weights.loc["B"]) < 1e-12


def test_none_benchmark_weights_keeps_equal_weight_behavior():
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()

    result = run_backtest(
        stock_returns,
        benchmark_returns,
        scores,
        rebalance_dates,
        benchmark_weights=None,
        transaction_cost_bps=0.0,
    )

    first_active_weights = result.weights.loc[pd.Timestamp("2024-01-03")]

    assert abs(first_active_weights.loc["A"] - 0.6) < 1e-12
    assert abs(first_active_weights.loc["B"] - 0.4) < 1e-12
