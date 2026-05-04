"""Minimal Version 0 backtest loop for index-enhancement research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.portfolio import (
    apply_active_weight_caps,
    apply_turnover_limit,
    calculate_turnover,
    construct_long_only_portfolio,
    equal_weight_benchmark,
    score_to_active_weights,
)


@dataclass(frozen=True)
class BacktestResult:
    """Container for daily backtest outputs."""

    portfolio_returns_gross: pd.Series
    portfolio_returns_net: pd.Series
    benchmark_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series


def run_backtest(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    scores: pd.DataFrame,
    rebalance_dates,
    active_budget: float = 0.20,
    transaction_cost_bps: float = 10.0,
    max_active_weight: Optional[float] = None,
    max_weight: Optional[float] = None,
    max_turnover: Optional[float] = None,
    trade_lag_days: int = 1,
) -> BacktestResult:
    """Run a monthly-style long-only index-enhancement backtest.

    The backtester uses simple stock returns for portfolio aggregation. Weights
    formed on a rebalance date become active only after `trade_lag_days` trading
    days. Missing stock returns for held tickers are treated as zero during the
    weighted daily return calculation so sparse return panels do not break the
    loop; data-quality filtering should happen before calling this function.
    """
    if trade_lag_days < 0:
        raise ValueError("trade_lag_days must be non-negative.")

    stock_returns = _with_datetime_index(stock_returns).sort_index()
    benchmark_returns = _with_datetime_index(benchmark_returns).sort_index()
    scores = _with_datetime_index(scores).sort_index()
    rebalance_index = pd.DatetimeIndex(rebalance_dates).sort_values()

    active_weights: dict[pd.Timestamp, pd.Series] = {}
    turnover_by_active_date: dict[pd.Timestamp, float] = {}
    costs_by_active_date: dict[pd.Timestamp, float] = {}
    previous_weights = pd.Series(dtype=float)

    for rebalance_date in rebalance_index:
        score_row = _latest_score_row(scores, rebalance_date)
        if score_row is None:
            continue

        valid_scores = score_row.dropna()
        if valid_scores.empty:
            continue

        active_date = _active_date_for_rebalance(
            stock_returns.index,
            rebalance_date,
            trade_lag_days,
        )
        if active_date is None:
            continue

        benchmark_weights = equal_weight_benchmark(valid_scores)
        raw_active_weights = score_to_active_weights(
            valid_scores,
            active_budget=active_budget,
        )
        capped_active_weights = apply_active_weight_caps(
            raw_active_weights,
            max_active_weight=max_active_weight,
        )
        target_weights = construct_long_only_portfolio(
            benchmark_weights,
            capped_active_weights,
            max_weight=max_weight,
        )
        target_weights = apply_turnover_limit(
            previous_weights,
            target_weights,
            max_turnover=max_turnover,
        )

        turnover = calculate_turnover(previous_weights, target_weights)
        one_way_cost = transaction_cost_bps / 10000.0
        transaction_cost = 2.0 * one_way_cost * turnover

        active_weights[active_date] = target_weights
        turnover_by_active_date[active_date] = turnover
        costs_by_active_date[active_date] = transaction_cost
        previous_weights = target_weights

    return _build_result(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        active_weights=active_weights,
        turnover_by_active_date=turnover_by_active_date,
        costs_by_active_date=costs_by_active_date,
    )


def _build_result(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    active_weights: dict[pd.Timestamp, pd.Series],
    turnover_by_active_date: dict[pd.Timestamp, float],
    costs_by_active_date: dict[pd.Timestamp, float],
) -> BacktestResult:
    if not active_weights:
        empty_index = pd.DatetimeIndex([], name=stock_returns.index.name)
        return BacktestResult(
            portfolio_returns_gross=pd.Series(dtype=float, index=empty_index),
            portfolio_returns_net=pd.Series(dtype=float, index=empty_index),
            benchmark_returns=pd.Series(dtype=float, index=empty_index),
            weights=pd.DataFrame(dtype=float),
            turnover=pd.Series(dtype=float),
            transaction_costs=pd.Series(dtype=float),
        )

    active_dates = sorted(active_weights)
    weights = pd.DataFrame.from_dict(active_weights, orient="index").sort_index()
    weights.index = pd.DatetimeIndex(weights.index)
    turnover = pd.Series(turnover_by_active_date, dtype=float).sort_index()
    transaction_costs = pd.Series(costs_by_active_date, dtype=float).sort_index()

    gross_returns: dict[pd.Timestamp, float] = {}
    current_weights: Optional[pd.Series] = None
    active_date_position = 0

    for date, daily_returns in stock_returns.iterrows():
        while (
            active_date_position < len(active_dates)
            and date >= active_dates[active_date_position]
        ):
            current_weights = active_weights[active_dates[active_date_position]]
            active_date_position += 1

        if current_weights is None:
            continue

        held_returns = daily_returns.reindex(current_weights.index).fillna(0.0)
        gross_returns[date] = float((current_weights * held_returns).sum())

    gross = pd.Series(gross_returns, dtype=float).sort_index()
    costs_aligned = transaction_costs.reindex(gross.index, fill_value=0.0)
    net = gross - costs_aligned
    aligned_benchmark = benchmark_returns.reindex(gross.index)

    return BacktestResult(
        portfolio_returns_gross=gross,
        portfolio_returns_net=net,
        benchmark_returns=aligned_benchmark,
        weights=weights,
        turnover=turnover,
        transaction_costs=transaction_costs,
    )


def _latest_score_row(
    scores: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> Optional[pd.Series]:
    position = scores.index.searchsorted(rebalance_date, side="right") - 1
    if position < 0:
        return None

    return scores.iloc[position]


def _active_date_for_rebalance(
    trading_index: pd.DatetimeIndex,
    rebalance_date: pd.Timestamp,
    trade_lag_days: int,
) -> Optional[pd.Timestamp]:
    if rebalance_date in trading_index:
        rebalance_position = trading_index.get_loc(rebalance_date)
        if isinstance(rebalance_position, slice):
            rebalance_position = rebalance_position.stop - 1
        active_position = int(rebalance_position) + trade_lag_days
    else:
        insertion_position = trading_index.searchsorted(rebalance_date, side="right")
        active_position = insertion_position + max(trade_lag_days - 1, 0)

    if active_position >= len(trading_index):
        return None

    return trading_index[active_position]


def _with_datetime_index(data):
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    return data
