"""Experiment helpers for Version 0 robustness checks."""

from __future__ import annotations

import pandas as pd

from src.backtester import run_backtest
from src.diagnostics import calculate_ic, compute_forward_returns, ic_summary
from src.factors import compute_momentum_12_1
from src.performance import performance_summary
from src.signal_processing import combine_factors


def run_cost_sensitivity(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    scores: pd.DataFrame,
    rebalance_dates,
    cost_bps_list,
    active_budget: float = 0.20,
    max_active_weight: float | None = None,
    max_weight: float | None = None,
    max_turnover: float | None = None,
    trade_lag_days: int = 1,
    benchmark_weights: pd.Series | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Run the same backtest across transaction-cost assumptions.

    This experiment changes only `transaction_cost_bps`; factor scores,
    portfolio construction settings, and backtest timing are otherwise held
    fixed. Turnover is annualized as average rebalance turnover times 12,
    matching the monthly Version 0 convention.
    """
    rows: list[dict[str, object]] = []

    for cost_bps in cost_bps_list:
        result = run_backtest(
            stock_returns=stock_returns,
            benchmark_returns=benchmark_returns,
            scores=scores,
            rebalance_dates=rebalance_dates,
            active_budget=active_budget,
            transaction_cost_bps=cost_bps,
            max_active_weight=max_active_weight,
            max_weight=max_weight,
            max_turnover=max_turnover,
            trade_lag_days=trade_lag_days,
            benchmark_weights=benchmark_weights,
        )
        summary = performance_summary(
            result.portfolio_returns_net,
            benchmark_returns=result.benchmark_returns,
            periods_per_year=periods_per_year,
        )
        average_turnover = result.turnover.mean()

        row = {
            "cost_bps": float(cost_bps),
            "average_turnover": average_turnover,
            "annualized_turnover": average_turnover * 12,
        }
        row.update(
            {
                metric: summary.get(metric, pd.NA)
                for metric in [
                    "total_return",
                    "annualized_return",
                    "annualized_volatility",
                    "sharpe_ratio",
                    "max_drawdown",
                    "annualized_active_return",
                    "tracking_error",
                    "information_ratio",
                ]
            }
        )
        rows.append(row)

    columns = [
        "cost_bps",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "annualized_active_return",
        "tracking_error",
        "information_ratio",
        "average_turnover",
        "annualized_turnover",
    ]
    return pd.DataFrame(rows, columns=columns)


def run_active_budget_sensitivity(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    scores: pd.DataFrame,
    rebalance_dates,
    active_budget_list,
    transaction_cost_bps: float = 10.0,
    max_active_weight: float | None = None,
    max_weight: float | None = None,
    max_turnover: float | None = None,
    trade_lag_days: int = 1,
    benchmark_weights: pd.Series | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Run the same backtest across active-budget assumptions.

    This experiment changes only `active_budget`; transaction costs,
    constraints, scores, and timing assumptions are otherwise held fixed.
    Turnover is annualized as average rebalance turnover times 12, matching the
    monthly Version 0 convention.
    """
    rows: list[dict[str, object]] = []

    for active_budget in active_budget_list:
        result = run_backtest(
            stock_returns=stock_returns,
            benchmark_returns=benchmark_returns,
            scores=scores,
            rebalance_dates=rebalance_dates,
            active_budget=active_budget,
            transaction_cost_bps=transaction_cost_bps,
            max_active_weight=max_active_weight,
            max_weight=max_weight,
            max_turnover=max_turnover,
            trade_lag_days=trade_lag_days,
            benchmark_weights=benchmark_weights,
        )
        summary = performance_summary(
            result.portfolio_returns_net,
            benchmark_returns=result.benchmark_returns,
            periods_per_year=periods_per_year,
        )
        average_turnover = result.turnover.mean()

        row = {
            "active_budget": float(active_budget),
            "average_turnover": average_turnover,
            "annualized_turnover": average_turnover * 12,
        }
        row.update(
            {
                metric: summary.get(metric, pd.NA)
                for metric in [
                    "total_return",
                    "annualized_return",
                    "annualized_volatility",
                    "sharpe_ratio",
                    "max_drawdown",
                    "annualized_active_return",
                    "tracking_error",
                    "information_ratio",
                ]
            }
        )
        rows.append(row)

    columns = [
        "active_budget",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "annualized_active_return",
        "tracking_error",
        "information_ratio",
        "average_turnover",
        "annualized_turnover",
    ]
    return pd.DataFrame(rows, columns=columns)


def run_factor_variant_sensitivity(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    factor_dict: dict[str, pd.DataFrame],
    rebalance_dates,
    variants: dict[str, list[str]],
    transaction_cost_bps: float = 10.0,
    active_budget: float = 0.20,
    max_active_weight: float | None = None,
    max_weight: float | None = None,
    max_turnover: float | None = None,
    trade_lag_days: int = 1,
    benchmark_weights: pd.Series | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Run the same backtest across factor-combination variants."""
    rows: list[dict[str, object]] = []

    for variant_name, factor_names in variants.items():
        missing_factors = [name for name in factor_names if name not in factor_dict]
        if missing_factors:
            raise ValueError(
                f"Variant {variant_name!r} references missing factors: "
                f"{missing_factors}"
            )

        selected_factors = {name: factor_dict[name] for name in factor_names}
        composite_scores = combine_factors(
            selected_factors,
            weights=None,
            standardize_composite=True,
        )
        result = run_backtest(
            stock_returns=stock_returns,
            benchmark_returns=benchmark_returns,
            scores=composite_scores,
            rebalance_dates=rebalance_dates,
            active_budget=active_budget,
            transaction_cost_bps=transaction_cost_bps,
            max_active_weight=max_active_weight,
            max_weight=max_weight,
            max_turnover=max_turnover,
            trade_lag_days=trade_lag_days,
            benchmark_weights=benchmark_weights,
        )
        summary = performance_summary(
            result.portfolio_returns_net,
            benchmark_returns=result.benchmark_returns,
            periods_per_year=periods_per_year,
        )
        average_turnover = result.turnover.mean()

        row = {
            "variant": variant_name,
            "factors": ",".join(factor_names),
            "average_turnover": average_turnover,
            "annualized_turnover": average_turnover * 12,
        }
        row.update(
            {
                metric: summary.get(metric, pd.NA)
                for metric in [
                    "total_return",
                    "annualized_return",
                    "annualized_volatility",
                    "sharpe_ratio",
                    "max_drawdown",
                    "annualized_active_return",
                    "tracking_error",
                    "information_ratio",
                ]
            }
        )
        rows.append(row)

    columns = [
        "variant",
        "factors",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "annualized_active_return",
        "tracking_error",
        "information_ratio",
        "average_turnover",
        "annualized_turnover",
    ]
    return pd.DataFrame(rows, columns=columns)


def run_subperiod_analysis(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods: dict[str, tuple[str | None, str | None]],
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Summarize strategy performance over named calendar subperiods."""
    portfolio_returns = portfolio_returns.copy()
    benchmark_returns = benchmark_returns.copy()
    portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index)
    portfolio_returns = portfolio_returns.sort_index()
    benchmark_returns = benchmark_returns.sort_index()

    rows: list[dict[str, object]] = []
    for period_name, (start_date, end_date) in periods.items():
        period_portfolio_returns = _slice_period(
            portfolio_returns,
            start_date=start_date,
            end_date=end_date,
        )
        period_benchmark_returns = _slice_period(
            benchmark_returns,
            start_date=start_date,
            end_date=end_date,
        )
        summary = performance_summary(
            period_portfolio_returns,
            benchmark_returns=period_benchmark_returns,
            periods_per_year=periods_per_year,
        )

        row = {
            "period": period_name,
            "start_date": start_date,
            "end_date": end_date,
            "num_observations": int(period_portfolio_returns.dropna().shape[0]),
        }
        row.update(
            {
                metric: summary.get(metric, pd.NA)
                for metric in [
                    "total_return",
                    "annualized_return",
                    "annualized_volatility",
                    "sharpe_ratio",
                    "max_drawdown",
                    "hit_ratio",
                    "annualized_active_return",
                    "tracking_error",
                    "information_ratio",
                ]
            }
        )
        rows.append(row)

    columns = [
        "period",
        "start_date",
        "end_date",
        "num_observations",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "hit_ratio",
        "annualized_active_return",
        "tracking_error",
        "information_ratio",
    ]
    return pd.DataFrame(rows, columns=columns)


def compute_equal_weight_benchmark_returns(stock_returns: pd.DataFrame) -> pd.Series:
    """Compute a same-universe equal-weight benchmark return series.

    This is not a true S&P 500 benchmark. It is a current-universe equal-weight
    proxy that helps separate factor-tilt performance from SPY benchmark
    mismatch and universe effects.
    """
    return stock_returns.mean(axis=1, skipna=True).rename("EqualWeightUniverse")


def compare_benchmarks(
    portfolio_returns: pd.Series,
    spy_returns: pd.Series,
    equal_weight_returns: pd.Series,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Compare portfolio performance against SPY and equal-weight proxies."""
    benchmark_map = {
        "SPY": spy_returns,
        "EqualWeightUniverse": equal_weight_returns,
    }
    rows: list[dict[str, object]] = []

    for benchmark_name, benchmark_returns in benchmark_map.items():
        summary = performance_summary(
            portfolio_returns,
            benchmark_returns=benchmark_returns,
            periods_per_year=periods_per_year,
        )
        row = {"benchmark": benchmark_name}
        row.update(_summary_metrics(summary))
        rows.append(row)

    columns = [
        "benchmark",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "hit_ratio",
        "annualized_active_return",
        "tracking_error",
        "information_ratio",
    ]
    return pd.DataFrame(rows, columns=columns)


def run_subperiod_analysis_by_benchmark(
    portfolio_returns: pd.Series,
    benchmark_dict: dict[str, pd.Series],
    periods: dict[str, tuple[str | None, str | None]],
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Summarize subperiod performance against multiple benchmarks."""
    portfolio_returns = portfolio_returns.copy()
    portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
    portfolio_returns = portfolio_returns.sort_index()

    rows: list[dict[str, object]] = []
    for benchmark_name, benchmark_returns in benchmark_dict.items():
        clean_benchmark = benchmark_returns.copy()
        clean_benchmark.index = pd.to_datetime(clean_benchmark.index)
        clean_benchmark = clean_benchmark.sort_index()

        for period_name, (start_date, end_date) in periods.items():
            period_portfolio_returns = _slice_period(
                portfolio_returns,
                start_date=start_date,
                end_date=end_date,
            )
            period_benchmark_returns = _slice_period(
                clean_benchmark,
                start_date=start_date,
                end_date=end_date,
            )
            summary = performance_summary(
                period_portfolio_returns,
                benchmark_returns=period_benchmark_returns,
                periods_per_year=periods_per_year,
            )

            row = {
                "benchmark": benchmark_name,
                "period": period_name,
                "start_date": start_date,
                "end_date": end_date,
                "num_observations": int(period_portfolio_returns.dropna().shape[0]),
            }
            row.update(_summary_metrics(summary))
            rows.append(row)

    columns = [
        "benchmark",
        "period",
        "start_date",
        "end_date",
        "num_observations",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "hit_ratio",
        "annualized_active_return",
        "tracking_error",
        "information_ratio",
    ]
    return pd.DataFrame(rows, columns=columns)


def run_momentum_ic_sensitivity(
    stock_prices: pd.DataFrame,
    rebalance_dates,
    lookback_days_list,
    skip_days_list,
    periods: dict[str, tuple[str | None, str | None]] | None = None,
):
    """Evaluate momentum IC across lookback and skip-day parameter choices."""
    forward_returns = compute_forward_returns(stock_prices, rebalance_dates)
    overall_rows: list[dict[str, object]] = []
    subperiod_rows: list[dict[str, object]] = []

    for lookback_days in lookback_days_list:
        for skip_days in skip_days_list:
            momentum_scores = compute_momentum_12_1(
                stock_prices,
                lookback_days=lookback_days,
                skip_days=skip_days,
                use_log=True,
            )
            pearson_ic = calculate_ic(
                momentum_scores,
                forward_returns,
                method="pearson",
            )
            rank_ic = calculate_ic(
                momentum_scores,
                forward_returns,
                method="spearman",
            )

            overall_rows.append(
                _momentum_ic_row(
                    lookback_days=lookback_days,
                    skip_days=skip_days,
                    pearson_ic=pearson_ic,
                    rank_ic=rank_ic,
                )
            )

            if periods is not None:
                for period_name, (start_date, end_date) in periods.items():
                    subperiod_rows.append(
                        {
                            **_momentum_ic_row(
                                lookback_days=lookback_days,
                                skip_days=skip_days,
                                pearson_ic=_slice_period(
                                    pearson_ic,
                                    start_date=start_date,
                                    end_date=end_date,
                                ),
                                rank_ic=_slice_period(
                                    rank_ic,
                                    start_date=start_date,
                                    end_date=end_date,
                                ),
                            ),
                            "period": period_name,
                        }
                    )

    overall_summary = pd.DataFrame(
        overall_rows,
        columns=_momentum_ic_columns(include_std_ic=True),
    )
    if periods is None:
        return overall_summary

    subperiod_summary = pd.DataFrame(
        subperiod_rows,
        columns=_momentum_ic_subperiod_columns(),
    )
    return overall_summary, subperiod_summary


def _momentum_ic_row(
    lookback_days: int,
    skip_days: int,
    pearson_ic: pd.Series,
    rank_ic: pd.Series,
) -> dict[str, object]:
    pearson_summary = ic_summary(pearson_ic)
    rank_summary = ic_summary(rank_ic)

    return {
        "lookback_days": int(lookback_days),
        "skip_days": int(skip_days),
        "mean_ic": pearson_summary["mean_ic"],
        "std_ic": pearson_summary["std_ic"],
        "icir": pearson_summary["icir"],
        "hit_rate": pearson_summary["hit_rate"],
        "mean_rank_ic": rank_summary["mean_ic"],
        "rank_icir": rank_summary["icir"],
        "rank_hit_rate": rank_summary["hit_rate"],
        "count": pearson_summary["count"],
    }


def _momentum_ic_columns(include_std_ic: bool) -> list[str]:
    columns = [
        "lookback_days",
        "skip_days",
        "mean_ic",
    ]
    if include_std_ic:
        columns.append("std_ic")

    columns.extend(
        [
            "icir",
            "hit_rate",
            "count",
            "mean_rank_ic",
            "rank_icir",
            "rank_hit_rate",
        ]
    )
    return columns


def _momentum_ic_subperiod_columns() -> list[str]:
    return [
        "lookback_days",
        "skip_days",
        "period",
        "mean_ic",
        "icir",
        "hit_rate",
        "mean_rank_ic",
        "rank_icir",
        "rank_hit_rate",
        "count",
    ]


def _summary_metrics(summary: pd.Series) -> dict[str, object]:
    return {
        metric: summary.get(metric, pd.NA)
        for metric in [
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "hit_ratio",
            "annualized_active_return",
            "tracking_error",
            "information_ratio",
        ]
    }


def _slice_period(
    returns: pd.Series,
    start_date: str | None,
    end_date: str | None,
) -> pd.Series:
    start = pd.to_datetime(start_date) if start_date is not None else None
    end = pd.to_datetime(end_date) if end_date is not None else None

    if start is None and end is None:
        return returns
    if start is None:
        return returns.loc[:end]
    if end is None:
        return returns.loc[start:]

    return returns.loc[start:end]
