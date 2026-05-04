import numpy as np
import pandas as pd

from src.factors import (
    compute_low_volatility,
    compute_momentum_12_1,
    compute_short_term_reversal,
)


def test_compute_momentum_12_1_uses_log_shifted_prices_by_default():
    prices = pd.DataFrame(
        {
            "AAA": [10.0, 20.0, 30.0, 40.0, 50.0],
            "BBB": [100.0, 80.0, 60.0, 40.0, 20.0],
        },
        index=pd.date_range("2024-01-01", periods=5),
    )

    momentum = compute_momentum_12_1(prices, lookback_days=3, skip_days=1)

    expected = pd.DataFrame(
        {
            "AAA": [None, None, None, np.log(3.0), np.log(2.0)],
            "BBB": [None, None, None, np.log(0.6), np.log(0.5)],
        },
        index=prices.index,
    )
    pd.testing.assert_frame_equal(momentum, expected)


def test_compute_momentum_12_1_preserves_index_and_columns():
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"])
    prices = pd.DataFrame(
        {
            "MSFT": [100.0, 105.0, 110.0],
            "AAPL": [200.0, 210.0, 220.0],
        },
        index=index,
    )

    momentum = compute_momentum_12_1(prices, lookback_days=2, skip_days=1)

    pd.testing.assert_index_equal(momentum.index, prices.index)
    pd.testing.assert_index_equal(momentum.columns, prices.columns)


def test_compute_momentum_12_1_matches_required_formula():
    prices = pd.DataFrame(
        {"AAA": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]},
        index=pd.date_range("2024-01-01", periods=6),
    )

    momentum = compute_momentum_12_1(prices, lookback_days=4, skip_days=2)
    expected = np.log(prices.shift(2) / prices.shift(4))

    pd.testing.assert_frame_equal(momentum, expected)


def test_compute_momentum_12_1_simple_formula_when_use_log_false():
    prices = pd.DataFrame(
        {"AAA": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]},
        index=pd.date_range("2024-01-01", periods=6),
    )

    momentum = compute_momentum_12_1(
        prices,
        lookback_days=4,
        skip_days=2,
        use_log=False,
    )
    expected = prices.shift(2) / prices.shift(4) - 1

    pd.testing.assert_frame_equal(momentum, expected)


def test_compute_low_volatility_uses_prior_returns_window():
    returns = pd.DataFrame(
        {"AAA": [0.01, 0.02, 0.04, 0.08]},
        index=pd.date_range("2024-01-01", periods=4),
    )

    low_volatility = compute_low_volatility(returns, window=2, annualization=1)
    expected = -returns.shift(1).rolling(2).std()

    pd.testing.assert_frame_equal(low_volatility, expected)


def test_compute_short_term_reversal_uses_log_return_by_default():
    prices = pd.DataFrame(
        {
            "AAA": [10.0, 12.0, 15.0, 18.0],
            "BBB": [20.0, 10.0, 5.0, 2.5],
        },
        index=pd.date_range("2024-01-01", periods=4),
    )

    reversal = compute_short_term_reversal(prices, window=2)
    expected = -np.log(prices / prices.shift(2))

    pd.testing.assert_frame_equal(reversal, expected)
