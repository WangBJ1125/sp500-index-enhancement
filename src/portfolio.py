"""Portfolio construction helpers for index-enhancement research."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def equal_weight_benchmark(scores_row: pd.Series) -> pd.Series:
    """Create equal benchmark weights for tickers with valid scores."""
    valid_scores = scores_row.dropna()
    if valid_scores.empty:
        return pd.Series(dtype=float, index=valid_scores.index)

    weight = 1.0 / len(valid_scores)
    return pd.Series(weight, index=valid_scores.index, dtype=float)


def score_to_active_weights(
    scores_row: pd.Series,
    active_budget: float = 0.20,
) -> pd.Series:
    """Convert centered cross-sectional scores into active weights."""
    if active_budget < 0:
        raise ValueError("active_budget must be non-negative.")

    valid_scores = scores_row.dropna().astype(float)
    if valid_scores.empty:
        return pd.Series(dtype=float, index=valid_scores.index)

    centered_scores = valid_scores - valid_scores.mean()
    absolute_score_sum = centered_scores.abs().sum()
    if absolute_score_sum == 0 or pd.isna(absolute_score_sum):
        return pd.Series(0.0, index=valid_scores.index, dtype=float)

    return active_budget * centered_scores / absolute_score_sum


def apply_active_weight_caps(
    active_weights: pd.Series,
    max_active_weight: Optional[float] = None,
) -> pd.Series:
    """Clip active weights and re-center them to keep total active weight near zero."""
    if max_active_weight is None:
        return active_weights.copy()
    if max_active_weight < 0:
        raise ValueError("max_active_weight must be non-negative.")

    clipped = active_weights.astype(float).clip(
        lower=-max_active_weight,
        upper=max_active_weight,
    )
    for _ in range(max(len(clipped), 1) * 20):
        recentered = clipped - clipped.mean()
        clipped = recentered.clip(lower=-max_active_weight, upper=max_active_weight)
        if abs(clipped.sum()) < 1e-12:
            break

    return clipped


def construct_long_only_portfolio(
    benchmark_weights: pd.Series,
    active_weights: pd.Series,
    max_weight: Optional[float] = None,
) -> pd.Series:
    """Construct long-only portfolio weights from benchmark plus active weights."""
    tickers = benchmark_weights.index.union(active_weights.index)
    benchmark = benchmark_weights.reindex(tickers, fill_value=0.0).astype(float)
    active = active_weights.reindex(tickers, fill_value=0.0).astype(float)

    raw_weights = (benchmark + active).clip(lower=0.0)
    normalized = _normalize_nonnegative_weights(raw_weights)

    if max_weight is not None:
        normalized = _apply_max_weight_cap(normalized, max_weight)

    return _normalize_nonnegative_weights(normalized).clip(lower=0.0)


def calculate_turnover(old_weights: pd.Series, new_weights: pd.Series) -> float:
    """Calculate one-way portfolio turnover between two weight vectors."""
    tickers = old_weights.index.union(new_weights.index)
    old_aligned = old_weights.reindex(tickers, fill_value=0.0).astype(float)
    new_aligned = new_weights.reindex(tickers, fill_value=0.0).astype(float)

    return 0.5 * (new_aligned - old_aligned).abs().sum()


def apply_turnover_limit(
    current_weights: pd.Series,
    target_weights: pd.Series,
    max_turnover: Optional[float] = None,
) -> pd.Series:
    """Scale trades if moving fully to target weights would exceed turnover."""
    if max_turnover is None:
        return target_weights.copy()
    if max_turnover < 0:
        raise ValueError("max_turnover must be non-negative.")

    tickers = current_weights.index.union(target_weights.index)
    current = current_weights.reindex(tickers, fill_value=0.0).astype(float)
    target = target_weights.reindex(tickers, fill_value=0.0).astype(float)
    raw_turnover = calculate_turnover(current, target)

    if raw_turnover <= max_turnover or raw_turnover == 0:
        return _normalize_nonnegative_weights(target)

    delta = target - current
    scaled_delta = delta * (max_turnover / raw_turnover)
    limited_weights = current + scaled_delta

    return _normalize_nonnegative_weights(limited_weights.clip(lower=0.0))


def _normalize_nonnegative_weights(weights: pd.Series) -> pd.Series:
    nonnegative = weights.astype(float).clip(lower=0.0)
    total_weight = nonnegative.sum()

    if total_weight > 0:
        return nonnegative / total_weight

    if nonnegative.empty:
        return pd.Series(dtype=float, index=nonnegative.index)

    return pd.Series(1.0 / len(nonnegative), index=nonnegative.index, dtype=float)


def _apply_max_weight_cap(weights: pd.Series, max_weight: float) -> pd.Series:
    if max_weight <= 0:
        raise ValueError("max_weight must be positive.")

    weights = _normalize_nonnegative_weights(weights)
    if weights.empty or max_weight * len(weights) < 1.0 - 1e-12:
        return weights

    capped = weights.copy()
    for _ in range(len(capped)):
        over_cap = capped > max_weight
        if not over_cap.any():
            break

        capped.loc[over_cap] = max_weight
        remaining_weight = 1.0 - capped.loc[over_cap].sum()
        under_cap = ~over_cap

        if remaining_weight <= 0 or not under_cap.any():
            break

        redistributable = capped.loc[under_cap].sum()
        if redistributable <= 0:
            capped.loc[under_cap] = remaining_weight / under_cap.sum()
        else:
            capped.loc[under_cap] = (
                capped.loc[under_cap] / redistributable * remaining_weight
            )

    if (capped > max_weight + 1e-12).any():
        capped = capped.clip(upper=max_weight)

    final_total = capped.sum()
    if not np.isclose(final_total, 1.0) and final_total > 0:
        uncapped = capped < max_weight - 1e-12
        if uncapped.any():
            remaining_weight = 1.0 - capped.loc[~uncapped].sum()
            uncapped_total = capped.loc[uncapped].sum()
            capped.loc[uncapped] = (
                capped.loc[uncapped] / uncapped_total * remaining_weight
            )

    return capped
