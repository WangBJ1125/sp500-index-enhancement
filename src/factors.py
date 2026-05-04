"""Price-based factor construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_momentum_12_1(
    prices: pd.DataFrame,
    lookback_days: int = 252,
    skip_days: int = 21,
    use_log: bool = True,
) -> pd.DataFrame:
    """Compute 12-1 momentum using only historical adjusted close prices.
    12-1: use returns from 12 months ago to 1 month ago
    1 month: approximately 21 trading days

    At date t, the signal compares the price from t - skip_days with the price
    from t - lookback_days. The most recent skip window is excluded to avoid
    using short-term reversal noise in the momentum estimate.
    """
    recent_prices = prices.shift(skip_days)
    lookback_prices = prices.shift(lookback_days)
    ratio = recent_prices / lookback_prices

    if use_log:
        valid_prices = (recent_prices > 0) & (lookback_prices > 0)
        return np.log(ratio.where(valid_prices))

    return ratio - 1


def compute_low_volatility(
    returns: pd.DataFrame,
    window: int = 126,
    annualization: int = 252,
) -> pd.DataFrame:
    """Compute low-volatility score from caller-supplied simple or log returns."""
    volatility = returns.shift(1).rolling(window).std() * np.sqrt(annualization)
    return -volatility


def compute_short_term_reversal(
    prices: pd.DataFrame,
    window: int = 21,
    use_log: bool = True,
) -> pd.DataFrame:
    """Compute short-term reversal from historical adjusted close prices."""
    prior_prices = prices.shift(window)
    ratio = prices / prior_prices

    if use_log:
        valid_prices = (prices > 0) & (prior_prices > 0)
        return -np.log(ratio.where(valid_prices))

    return -(ratio - 1)
