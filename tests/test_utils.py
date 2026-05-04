import pandas as pd

from src import utils


def test_compute_simple_returns_for_price_dataframe():
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 121.0],
            "MSFT": [50.0, 45.0, 54.0],
        },
        index=pd.date_range("2024-01-02", periods=3),
    )

    returns = utils.compute_simple_returns(prices)

    assert pd.isna(returns.iloc[0, 0])
    assert round(returns.loc["2024-01-03", "AAPL"], 10) == 0.10
    assert round(returns.loc["2024-01-04", "MSFT"], 10) == 0.20


def test_get_month_end_rebalance_dates_uses_last_available_trading_date():
    trading_index = pd.to_datetime(
        [
            "2024-01-29",
            "2024-01-30",
            "2024-02-27",
            "2024-02-29",
            "2024-03-28",
        ]
    )

    rebalance_dates = utils.get_month_end_rebalance_dates(trading_index)

    expected = pd.DatetimeIndex(["2024-01-30", "2024-02-29", "2024-03-28"])
    pd.testing.assert_index_equal(rebalance_dates, expected)


def test_get_month_end_rebalance_dates_sorts_and_deduplicates_input():
    trading_index = pd.to_datetime(
        [
            "2024-02-29",
            "2024-01-31",
            "2024-01-31",
            "2024-02-28",
        ]
    )

    rebalance_dates = utils.get_month_end_rebalance_dates(trading_index)

    expected = pd.DatetimeIndex(["2024-01-31", "2024-02-29"])
    pd.testing.assert_index_equal(rebalance_dates, expected)


def test_align_to_common_dates_preserves_series_and_dataframe_types():
    prices = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    benchmark = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
        name="SPY",
    )

    aligned_prices, aligned_benchmark = utils.align_to_common_dates(prices, benchmark)

    expected_index = pd.DatetimeIndex(["2024-01-03", "2024-01-04"])
    assert isinstance(aligned_prices, pd.DataFrame)
    assert isinstance(aligned_benchmark, pd.Series)
    pd.testing.assert_index_equal(aligned_prices.index, expected_index)
    pd.testing.assert_index_equal(aligned_benchmark.index, expected_index)
