import sys
import types

import pandas as pd
import pytest

from src.benchmark import (
    get_current_market_caps,
    load_market_caps,
    market_cap_weights,
    save_market_caps,
)


def test_market_cap_weights_with_synthetic_market_caps():
    market_caps = pd.Series({"A": 100.0, "B": 300.0, "C": 600.0})

    result = market_cap_weights(market_caps)

    expected = pd.Series({"A": 0.1, "B": 0.3, "C": 0.6})
    pd.testing.assert_series_equal(result, expected)


def test_market_cap_weights_drops_nan_and_zero_market_caps():
    market_caps = pd.Series({"A": 100.0, "B": None, "C": 0.0, "D": 300.0})

    result = market_cap_weights(market_caps)

    expected = pd.Series({"A": 0.25, "D": 0.75})
    pd.testing.assert_series_equal(result, expected)


def test_market_cap_weights_sum_to_one():
    market_caps = pd.Series({"A": 10.0, "B": 20.0, "C": 30.0})

    result = market_cap_weights(market_caps)

    assert abs(result.sum() - 1.0) < 1e-12


def test_market_cap_weights_raises_when_no_valid_caps_exist():
    market_caps = pd.Series({"A": None, "B": 0.0, "C": -10.0})

    with pytest.raises(ValueError):
        market_cap_weights(market_caps)


def test_save_and_load_market_caps_round_trip(tmp_path):
    market_caps = pd.Series({"A": 100.0, "B": None, "C": 300.0})
    path = tmp_path / "market_caps.csv"

    save_market_caps(market_caps, path)
    result = load_market_caps(path)

    expected = pd.Series({"A": 100.0, "B": None, "C": 300.0}, name="market_cap")
    expected.index.name = "ticker"
    pd.testing.assert_series_equal(result, expected)


def test_get_current_market_caps_uses_fast_info_without_live_call(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.fast_info = {"market_cap": 123.0 if ticker == "A" else None}
            self.info = {"marketCap": 456.0}

    fake_yfinance = types.SimpleNamespace(Ticker=FakeTicker)
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)

    result = get_current_market_caps(["A", "B"])

    expected = pd.Series({"A": 123.0, "B": 456.0}, dtype=float, name="market_cap")
    pd.testing.assert_series_equal(result, expected)
