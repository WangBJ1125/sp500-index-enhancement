"""Factor diagnostic utilities for cross-sectional research analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_forward_returns(
    stock_prices: pd.DataFrame,
    rebalance_dates,
) -> pd.DataFrame:
    """Compute next-rebalance simple returns for diagnostics only.

    The returned forward returns use prices at rebalance date ``t`` and the
    next rebalance date ``t_next``. These values are intended for factor IC
    diagnostics and must not be used in portfolio construction.
    """
    prices = stock_prices.copy()
    prices.index = pd.DatetimeIndex(prices.index)
    rebalance_index = pd.DatetimeIndex(rebalance_dates)

    if len(rebalance_index) < 2:
        return pd.DataFrame(index=rebalance_index[:0], columns=prices.columns)

    forward_rows = []
    forward_dates = []
    for date, next_date in zip(rebalance_index[:-1], rebalance_index[1:]):
        forward_return = prices.loc[next_date] / prices.loc[date] - 1
        forward_rows.append(forward_return)
        forward_dates.append(date)

    return pd.DataFrame(forward_rows, index=pd.DatetimeIndex(forward_dates))


def calculate_ic(
    scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "pearson",
) -> pd.Series:
    """Calculate date-by-date cross-sectional information coefficients."""
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'.")

    common_dates = scores.index.intersection(forward_returns.index)
    common_tickers = scores.columns.intersection(forward_returns.columns)
    aligned_scores = scores.loc[common_dates, common_tickers]
    aligned_forward_returns = forward_returns.loc[common_dates, common_tickers]

    ic_values = []
    for date in common_dates:
        cross_section = pd.concat(
            [
                aligned_scores.loc[date].rename("score"),
                aligned_forward_returns.loc[date].rename("forward_return"),
            ],
            axis=1,
        ).dropna()

        if len(cross_section) < 3:
            ic_values.append(np.nan)
            continue

        ic_values.append(
            cross_section["score"].corr(
                cross_section["forward_return"],
                method=method,
            )
        )

    return pd.Series(ic_values, index=common_dates, name=f"{method}_ic")


def ic_summary(ic_series: pd.Series) -> pd.Series:
    """Summarize an IC time series, ignoring missing values."""
    clean_ic = ic_series.dropna()
    count = len(clean_ic)
    mean_ic = clean_ic.mean() if count > 0 else np.nan
    std_ic = clean_ic.std() if count > 0 else np.nan
    icir = mean_ic / std_ic if pd.notna(std_ic) and std_ic != 0 else np.nan
    hit_rate = (clean_ic > 0).mean() if count > 0 else np.nan

    return pd.Series(
        {
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "icir": icir,
            "hit_rate": hit_rate,
            "count": count,
        }
    )


def run_factor_ic_analysis(
    factor_dict: dict[str, pd.DataFrame],
    stock_prices: pd.DataFrame,
    rebalance_dates,
) -> pd.DataFrame:
    """Run Pearson and Spearman IC analysis for each supplied factor."""
    forward_returns = compute_forward_returns(stock_prices, rebalance_dates)
    rows = []

    for factor_name, factor_scores in factor_dict.items():
        pearson_ic = calculate_ic(factor_scores, forward_returns, method="pearson")
        rank_ic = calculate_ic(factor_scores, forward_returns, method="spearman")
        pearson_summary = ic_summary(pearson_ic)
        rank_summary = ic_summary(rank_ic)

        rows.append(
            {
                "factor": factor_name,
                "mean_ic": pearson_summary["mean_ic"],
                "std_ic": pearson_summary["std_ic"],
                "icir": pearson_summary["icir"],
                "hit_rate": pearson_summary["hit_rate"],
                "count": pearson_summary["count"],
                "mean_rank_ic": rank_summary["mean_ic"],
                "rank_icir": rank_summary["icir"],
                "rank_hit_rate": rank_summary["hit_rate"],
            }
        )

    return pd.DataFrame(rows)


def run_factor_ic_subperiod_analysis(
    factor_dict: dict[str, pd.DataFrame],
    stock_prices: pd.DataFrame,
    rebalance_dates,
    periods: dict[str, tuple[str, str | None]],
) -> pd.DataFrame:
    """Run IC diagnostics for each factor across named date subperiods."""
    forward_returns = compute_forward_returns(stock_prices, rebalance_dates)
    rows = []

    for factor_name, factor_scores in factor_dict.items():
        pearson_ic = calculate_ic(factor_scores, forward_returns, method="pearson")
        rank_ic = calculate_ic(factor_scores, forward_returns, method="spearman")

        for period_name, (start_date, end_date) in periods.items():
            period_pearson_ic = _slice_period(pearson_ic, start_date, end_date)
            period_rank_ic = _slice_period(rank_ic, start_date, end_date)
            pearson_summary = ic_summary(period_pearson_ic)
            rank_summary = ic_summary(period_rank_ic)

            rows.append(
                {
                    "factor": factor_name,
                    "period": period_name,
                    "mean_ic": pearson_summary["mean_ic"],
                    "icir": pearson_summary["icir"],
                    "hit_rate": pearson_summary["hit_rate"],
                    "mean_rank_ic": rank_summary["mean_ic"],
                    "rank_icir": rank_summary["icir"],
                    "rank_hit_rate": rank_summary["hit_rate"],
                    "count": pearson_summary["count"],
                }
            )

    return pd.DataFrame(rows)


def _slice_period(
    series: pd.Series,
    start_date: str,
    end_date: str | None,
) -> pd.Series:
    start = pd.Timestamp(start_date)
    if end_date is None:
        return series.loc[series.index >= start]

    end = pd.Timestamp(end_date)
    return series.loc[(series.index >= start) & (series.index <= end)]
