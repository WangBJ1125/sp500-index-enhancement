"""Common date and return utilities for the research framework."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Union

import numpy as np
import pandas as pd


PandasObject = Union[pd.DataFrame, pd.Series]


def compute_simple_returns(prices: PandasObject) -> PandasObject:
    """Compute simple period-over-period returns from prices."""
    return prices.pct_change()


def compute_log_returns(prices: PandasObject) -> PandasObject:
    """Compute log returns, leaving non-positive price observations as NaN."""
    previous_prices = prices.shift(1)
    ratio = prices / previous_prices
    valid_prices = (prices > 0) & (previous_prices > 0)

    return np.log(ratio.where(valid_prices))


def get_month_end_rebalance_dates(trading_index: Iterable) -> pd.DatetimeIndex:
    """Return the last available trading date in each calendar month."""
    index = pd.DatetimeIndex(trading_index).sort_values().drop_duplicates()
    if index.empty:
        return index

    month_periods = index.to_period("M")
    month_end_mask = ~month_periods.duplicated(keep="last")

    return pd.DatetimeIndex(index[month_end_mask], name=index.name)


def align_to_common_dates(*dfs: PandasObject) -> tuple[PandasObject, ...]:
    """Align Series/DataFrames to their shared DatetimeIndex dates."""
    if not dfs:
        raise ValueError("At least one DataFrame or Series is required.")

    normalized = tuple(_with_datetime_index(df) for df in dfs)
    common_index = normalized[0].index
    for df in normalized[1:]:
        common_index = common_index.intersection(df.index)

    common_index = pd.DatetimeIndex(common_index).sort_values()

    return tuple(df.loc[common_index].copy() for df in normalized)


def _with_datetime_index(df: PandasObject) -> PandasObject:
    aligned = df.copy()
    aligned.index = pd.to_datetime(aligned.index)
    return aligned
