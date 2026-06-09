"""Benchmark proxy utilities for S&P 500 index-enhancement research."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def get_current_market_caps(
    tickers: list[str],
    sleep_seconds: float = 0.0,
) -> pd.Series:
    """Retrieve current market capitalizations from yfinance.

    Warning:
        These are current market caps, not point-in-time historical market
        caps. Using current market caps to create historical benchmark weights
        introduces look-ahead bias. This function is intended only for a
        prototype benchmark proxy, not production-grade historical index
        replication.
    """
    import yfinance as yf

    market_caps: dict[str, float] = {}
    for position, ticker in enumerate(tickers):
        market_cap = np.nan
        try:
            yf_ticker = yf.Ticker(ticker)
            market_cap = _coerce_market_cap(
                _get_mapping_value(yf_ticker.fast_info, "market_cap")
            )
            if pd.isna(market_cap):
                market_cap = _coerce_market_cap(
                    _get_mapping_value(yf_ticker.info, "marketCap")
                )
        except Exception:
            market_cap = np.nan

        market_caps[ticker] = market_cap

        if sleep_seconds > 0 and position < len(tickers) - 1:
            time.sleep(sleep_seconds)

    return pd.Series(market_caps, dtype=float, name="market_cap")


def market_cap_weights(
    market_caps: pd.Series,
    min_market_cap: float = 0.0,
) -> pd.Series:
    """Convert market caps into normalized benchmark weights.

    Warning:
        If `market_caps` came from current yfinance data, these weights are not
        point-in-time. Using them for historical backtests introduces
        look-ahead bias. Treat them as a prototype benchmark proxy only.
    """
    valid_market_caps = market_caps.dropna().astype(float)
    valid_market_caps = valid_market_caps[valid_market_caps > min_market_cap]

    if valid_market_caps.empty:
        raise ValueError("No valid market caps remain after filtering.")

    total_market_cap = valid_market_caps.sum()
    if total_market_cap <= 0:
        raise ValueError("Total valid market cap must be positive.")

    return valid_market_caps / total_market_cap


def save_market_caps(market_caps: pd.Series, path: str | Path) -> None:
    """Save market caps to CSV while preserving ticker index values."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    market_caps.rename("market_cap").to_csv(output_path, index_label="ticker")


def load_market_caps(path: str | Path) -> pd.Series:
    """Load market caps from CSV while preserving ticker index values."""
    data = pd.read_csv(path, index_col=0)
    if "market_cap" in data.columns:
        market_caps = data["market_cap"]
    else:
        market_caps = data.iloc[:, 0]

    market_caps.index = market_caps.index.astype(str)
    return pd.to_numeric(market_caps, errors="coerce").rename("market_cap")


def _get_mapping_value(source: Any, key: str) -> Any:
    getter = getattr(source, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            return None

    return getattr(source, key, None)


def _coerce_market_cap(value: Any) -> float:
    if value is None:
        return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
