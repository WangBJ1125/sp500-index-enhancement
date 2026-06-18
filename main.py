"""Run the minimal Version 0 S&P 500 index-enhancement pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

# import necessary modules from src package
from src.alpha_model import build_alpha_score
from src.backtester import BacktestResult, run_backtest
from src.benchmark import (
    get_current_market_caps,
    load_market_caps,
    market_cap_weights,
    save_market_caps,
)
from src.data_loader import download_price_data
from src.diagnostics import (
    compute_active_exposure_diagnostics,
    run_factor_ic_analysis,
    run_factor_ic_subperiod_analysis,
    summarize_active_exposure,
)
from src.experiments import (
    compare_benchmarks,
    compute_equal_weight_benchmark_returns,
    run_active_budget_sensitivity,
    run_alpha_model_sensitivity,
    run_cost_sensitivity,
    run_factor_variant_sensitivity,
    run_momentum_backtest_sensitivity,
    run_momentum_ic_sensitivity,
    run_subperiod_analysis_by_benchmark,
    run_subperiod_analysis,
)
from src.factors import (
    compute_low_volatility,
    compute_momentum_12_1,
    compute_short_term_reversal,
)
from src.performance import performance_summary
from src.reporting import (
    plot_cumulative_active_return,
    plot_drawdown,
    plot_nav,
    plot_rolling_active_return,
    plot_rolling_tracking_error,
    plot_turnover,
    write_summary_report,
)
from src.universe import get_current_sp500_tickers
from src.utils import (
    compute_log_returns,
    compute_simple_returns,
    get_month_end_rebalance_dates,
)


CONFIG_PATH = Path("config.yaml")
PROCESSED_DATA_DIR = Path("data") / "processed"
EQUAL_WEIGHT_BENCHMARK_RETURNS_PATH = (
    PROCESSED_DATA_DIR / "equal_weight_benchmark_returns.csv"
)
CURRENT_MARKET_CAPS_PATH = PROCESSED_DATA_DIR / "current_market_caps.csv"
STATIC_MARKET_CAP_WEIGHTS_PATH = PROCESSED_DATA_DIR / "static_market_cap_weights.csv"
REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("reports") / "figures"
SUMMARY_REPORT_PATH = Path("reports") / "summary_report.md"
COST_SENSITIVITY_PATH = Path("reports") / "cost_sensitivity.csv"
ACTIVE_BUDGET_SENSITIVITY_PATH = Path("reports") / "active_budget_sensitivity.csv"
FACTOR_VARIANT_SENSITIVITY_PATH = Path("reports") / "factor_variant_sensitivity.csv"
ALPHA_MODEL_SENSITIVITY_PATH = Path("reports") / "alpha_model_sensitivity.csv"
SUBPERIOD_PERFORMANCE_PATH = Path("reports") / "subperiod_performance.csv"
BENCHMARK_COMPARISON_PATH = Path("reports") / "benchmark_comparison.csv"
SUBPERIOD_BY_BENCHMARK_PATH = Path("reports") / "subperiod_by_benchmark.csv"
FACTOR_IC_SUMMARY_PATH = Path("reports") / "factor_ic_summary.csv"
FACTOR_IC_SUBPERIOD_PATH = Path("reports") / "factor_ic_subperiod.csv"
MOMENTUM_IC_SENSITIVITY_PATH = Path("reports") / "momentum_ic_sensitivity.csv"
MOMENTUM_IC_SENSITIVITY_SUBPERIOD_PATH = (
    Path("reports") / "momentum_ic_sensitivity_subperiod.csv"
)
MOMENTUM_BACKTEST_SENSITIVITY_PATH = (
    Path("reports") / "momentum_backtest_sensitivity.csv"
)
ACTIVE_EXPOSURE_DIAGNOSTICS_PATH = Path("reports") / "active_exposure_diagnostics.csv"
ACTIVE_EXPOSURE_SUMMARY_PATH = Path("reports") / "active_exposure_summary.csv"
ALPHA_SCORES_PATH = PROCESSED_DATA_DIR / "alpha_scores.csv"


def main() -> None:
    """Execute the Version 0 research pipeline end to end."""
    config = _load_config(CONFIG_PATH)

    benchmark_ticker = str(_get_config(config, ["data", "benchmark"], "SPY")).strip()
    start_date = _get_config(config, ["data", "start_date"], "2015-01-01")
    end_date = _get_config(config, ["data", "end_date"], None)
    max_tickers = _get_config(config, ["data", "max_tickers"], None)
    refresh_market_caps = _as_bool(
        _get_config(config, ["data", "refresh_market_caps"], False)
    )
    benchmark_weight_method = _get_benchmark_weight_method(config)
    active_budget = _get_config(config, ["portfolio", "active_budget"], 0.20)
    transaction_cost_bps = _get_config(config, ["costs", "transaction_cost_bps"], 10.0)
    max_active_weight = _get_config(config, ["portfolio", "max_active_weight"], None)
    max_weight = _get_config_exact(config, ["portfolio", "max_weight"], None)
    max_turnover = _get_max_turnover(config)
    trade_lag_days = _get_config(config, ["backtest", "trade_lag_days"], 1)
    momentum_lookback_days = int(
        _get_config(config, ["factors", "momentum", "lookback_days"], 252)
    )
    momentum_skip_days = int(
        _get_config(config, ["factors", "momentum", "skip_days"], 21)
    )
    alpha_model = str(
        _get_config(config, ["alpha", "model"], "equal_weight_composite")
    ).strip()
    alpha_lowvol_penalty_weight = float(
        _get_config(config, ["alpha", "lowvol_penalty_weight"], 0.25)
    )
    alpha_reversal_weight = float(
        _get_config(config, ["alpha", "reversal_weight"], 0.20)
    )
    alpha_weights = _get_config(config, ["alpha", "weights"], None)

    print("Loading current S&P 500 universe...")
    stock_tickers = _select_stock_universe(
        benchmark_ticker=benchmark_ticker,
        max_tickers=max_tickers,
    )
    download_tickers = stock_tickers + [benchmark_ticker]

    print(
        f"Downloading price data for {len(stock_tickers)} stocks "
        f"plus {benchmark_ticker}..."
    )
    prices, _volumes = download_price_data(
        download_tickers,
        start=start_date,
        end=end_date,
    )

    stock_prices, benchmark_prices, stock_tickers = _split_stock_and_benchmark_prices(
        prices=prices,
        stock_tickers=stock_tickers,
        benchmark_ticker=benchmark_ticker,
    )
    benchmark_weights = _load_benchmark_weights(
        method=benchmark_weight_method,
        stock_tickers=stock_tickers,
        refresh_market_caps=refresh_market_caps,
    )

    print("Computing returns and factors...")
    stock_simple_returns = compute_simple_returns(stock_prices)
    benchmark_simple_returns = compute_simple_returns(benchmark_prices)
    stock_log_returns = compute_log_returns(stock_prices)

    print(
        "Momentum parameters: "
        f"lookback_days={momentum_lookback_days}, "
        f"skip_days={momentum_skip_days}"
    )
    momentum = compute_momentum_12_1(
        stock_prices,
        lookback_days=momentum_lookback_days,
        skip_days=momentum_skip_days,
        use_log=True,
    )
    low_volatility = compute_low_volatility(stock_log_returns, window=126)
    reversal = compute_short_term_reversal(stock_prices, window=21, use_log=True)
    factor_scores = {
        "momentum": momentum,
        "lowvol": low_volatility,
        "reversal": reversal,
    }
    print(f"Alpha model: {alpha_model}")
    alpha_scores = build_alpha_score(
        factor_scores,
        model=alpha_model,
        weights=alpha_weights,
        lowvol_penalty_weight=alpha_lowvol_penalty_weight,
        reversal_weight=alpha_reversal_weight,
        standardize=True,
    )

    rebalance_dates = get_month_end_rebalance_dates(stock_prices.index)

    _print_effective_constraints(
        benchmark_weight_method=benchmark_weight_method,
        active_budget=active_budget,
        max_active_weight=max_active_weight,
        max_weight=max_weight,
        max_turnover=max_turnover,
    )
    print(f"Running backtest across {len(rebalance_dates)} rebalance dates...")
    result = run_backtest(
        stock_returns=stock_simple_returns,
        benchmark_returns=benchmark_simple_returns,
        scores=alpha_scores,
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
    )

    _print_run_summary(
        result=result,
        summary=summary,
        stock_tickers=stock_tickers,
        stock_prices=stock_prices,
    )
    _save_outputs(result=result, summary=summary, output_dir=PROCESSED_DATA_DIR)
    alpha_scores.to_csv(ALPHA_SCORES_PATH)
    active_exposure_benchmark_weights = _active_exposure_benchmark_weights(
        benchmark_weight_method=benchmark_weight_method,
        benchmark_weights=benchmark_weights,
        portfolio_columns=result.weights.columns,
    )
    active_exposure_diagnostics = compute_active_exposure_diagnostics(
        result.weights,
        active_exposure_benchmark_weights,
    )
    active_exposure_summary = summarize_active_exposure(active_exposure_diagnostics)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    active_exposure_diagnostics.to_csv(ACTIVE_EXPOSURE_DIAGNOSTICS_PATH)
    active_exposure_summary.rename("value").to_csv(ACTIVE_EXPOSURE_SUMMARY_PATH)
    print("\nActive Exposure Summary")
    print(active_exposure_summary.to_string(float_format=lambda value: f"{value:.6f}"))
    equal_weight_benchmark_returns = compute_equal_weight_benchmark_returns(
        stock_simple_returns,
    ).reindex(result.portfolio_returns_net.index)
    _save_series(
        equal_weight_benchmark_returns,
        EQUAL_WEIGHT_BENCHMARK_RETURNS_PATH,
    )
    subperiods = {
        "2015_2019": ("2015-01-01", "2019-12-31"),
        "2020_2022": ("2020-01-01", "2022-12-31"),
        "2023_latest": ("2023-01-01", None),
    }
    benchmark_comparison = compare_benchmarks(
        portfolio_returns=result.portfolio_returns_net,
        spy_returns=result.benchmark_returns,
        equal_weight_returns=equal_weight_benchmark_returns,
    )
    _save_experiment_table(benchmark_comparison, BENCHMARK_COMPARISON_PATH)
    print("\nBenchmark Comparison")
    print(
        benchmark_comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    subperiod_by_benchmark = run_subperiod_analysis_by_benchmark(
        portfolio_returns=result.portfolio_returns_net,
        benchmark_dict={
            "SPY": result.benchmark_returns,
            "EqualWeightUniverse": equal_weight_benchmark_returns,
        },
        periods=subperiods,
    )
    _save_experiment_table(subperiod_by_benchmark, SUBPERIOD_BY_BENCHMARK_PATH)
    print("\nSubperiod Performance By Benchmark")
    print(
        subperiod_by_benchmark.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print(f"\nExperiment benchmark weight method: {benchmark_weight_method}")
    cost_sensitivity = run_cost_sensitivity(
        stock_returns=stock_simple_returns,
        benchmark_returns=benchmark_simple_returns,
        scores=alpha_scores,
        rebalance_dates=rebalance_dates,
        cost_bps_list=[0, 5, 10, 20],
        active_budget=active_budget,
        max_active_weight=max_active_weight,
        max_weight=max_weight,
        max_turnover=max_turnover,
        trade_lag_days=trade_lag_days,
        benchmark_weights=benchmark_weights,
    )
    _save_cost_sensitivity(cost_sensitivity, COST_SENSITIVITY_PATH)
    print("\nCost Sensitivity")
    print(
        cost_sensitivity.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    active_budget_sensitivity = run_active_budget_sensitivity(
        stock_returns=stock_simple_returns,
        benchmark_returns=benchmark_simple_returns,
        scores=alpha_scores,
        rebalance_dates=rebalance_dates,
        active_budget_list=[0.10, 0.20, 0.30],
        transaction_cost_bps=transaction_cost_bps,
        max_active_weight=max_active_weight,
        max_weight=max_weight,
        max_turnover=max_turnover,
        trade_lag_days=trade_lag_days,
        benchmark_weights=benchmark_weights,
    )
    _save_experiment_table(
        active_budget_sensitivity,
        ACTIVE_BUDGET_SENSITIVITY_PATH,
    )
    print("\nActive Budget Sensitivity")
    print(
        active_budget_sensitivity.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    factor_variant_sensitivity = run_factor_variant_sensitivity(
        stock_returns=stock_simple_returns,
        benchmark_returns=benchmark_simple_returns,
        factor_dict=factor_scores,
        rebalance_dates=rebalance_dates,
        variants={
            "momentum_only": ["momentum"],
            "lowvol_only": ["lowvol"],
            "reversal_only": ["reversal"],
            "momentum_lowvol": ["momentum", "lowvol"],
            "momentum_reversal": ["momentum", "reversal"],
            "lowvol_reversal": ["lowvol", "reversal"],
            "combined": ["momentum", "lowvol", "reversal"],
        },
        transaction_cost_bps=transaction_cost_bps,
        active_budget=active_budget,
        max_active_weight=max_active_weight,
        max_weight=max_weight,
        max_turnover=max_turnover,
        trade_lag_days=trade_lag_days,
        benchmark_weights=benchmark_weights,
    )
    _save_experiment_table(
        factor_variant_sensitivity,
        FACTOR_VARIANT_SENSITIVITY_PATH,
    )
    print("\nFactor Variant Sensitivity")
    print(
        factor_variant_sensitivity.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    alpha_model_sensitivity = run_alpha_model_sensitivity(
        factor_scores=factor_scores,
        stock_returns=stock_simple_returns,
        benchmark_returns=benchmark_simple_returns,
        rebalance_dates=rebalance_dates,
        alpha_model_configs=[
            {"model": "equal_weight_composite"},
            {"model": "momentum_only"},
            {
                "model": "risk_adjusted_momentum",
                "lowvol_penalty_weight": 0.25,
            },
            {
                "model": "momentum_reversal",
                "reversal_weight": 0.20,
            },
        ],
        benchmark_weights=benchmark_weights,
        transaction_cost_bps=transaction_cost_bps,
        active_budget=active_budget,
        max_active_weight=max_active_weight,
        max_weight=max_weight,
        max_turnover=max_turnover,
        trade_lag_days=trade_lag_days,
    )
    _save_experiment_table(
        alpha_model_sensitivity,
        ALPHA_MODEL_SENSITIVITY_PATH,
    )
    print("\nAlpha Model Sensitivity")
    print(
        alpha_model_sensitivity.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    subperiod_performance = run_subperiod_analysis(
        portfolio_returns=result.portfolio_returns_net,
        benchmark_returns=result.benchmark_returns,
        periods=subperiods,
    )
    _save_experiment_table(subperiod_performance, SUBPERIOD_PERFORMANCE_PATH)
    print("\nSubperiod Performance")
    print(
        subperiod_performance.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    ic_factor_scores = {
        **factor_scores,
        "composite": alpha_scores,
    }
    factor_ic_summary = run_factor_ic_analysis(
        factor_dict=ic_factor_scores,
        stock_prices=stock_prices,
        rebalance_dates=rebalance_dates,
    )
    _save_experiment_table(factor_ic_summary, FACTOR_IC_SUMMARY_PATH)
    print("\nFactor IC Summary")
    print(
        factor_ic_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    factor_ic_subperiod = run_factor_ic_subperiod_analysis(
        factor_dict=ic_factor_scores,
        stock_prices=stock_prices,
        rebalance_dates=rebalance_dates,
        periods=subperiods,
    )
    _save_experiment_table(factor_ic_subperiod, FACTOR_IC_SUBPERIOD_PATH)
    print("\nFactor IC Subperiod Analysis")
    print(
        factor_ic_subperiod.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    momentum_ic_sensitivity, momentum_ic_sensitivity_subperiod = (
        run_momentum_ic_sensitivity(
            stock_prices=stock_prices,
            rebalance_dates=rebalance_dates,
            lookback_days_list=[63, 126, 189, 252],
            skip_days_list=[0, 21],
            periods=subperiods,
        )
    )
    _save_experiment_table(
        momentum_ic_sensitivity,
        MOMENTUM_IC_SENSITIVITY_PATH,
    )
    _save_experiment_table(
        momentum_ic_sensitivity_subperiod,
        MOMENTUM_IC_SENSITIVITY_SUBPERIOD_PATH,
    )
    print("\nMomentum IC Sensitivity")
    print(
        momentum_ic_sensitivity.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    momentum_backtest_sensitivity = run_momentum_backtest_sensitivity(
        stock_prices=stock_prices,
        stock_returns=stock_simple_returns,
        benchmark_returns=benchmark_simple_returns,
        rebalance_dates=rebalance_dates,
        lookback_days_list=[63, 126, 189, 252],
        skip_days_list=[0, 21],
        benchmark_weights=benchmark_weights,
        transaction_cost_bps=transaction_cost_bps,
        active_budget=active_budget,
        max_active_weight=max_active_weight,
        max_weight=max_weight,
        max_turnover=max_turnover,
        trade_lag_days=trade_lag_days,
    )
    _save_experiment_table(
        momentum_backtest_sensitivity,
        MOMENTUM_BACKTEST_SENSITIVITY_PATH,
    )
    print("\nMomentum Backtest Sensitivity")
    print(
        momentum_backtest_sensitivity.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    _write_reports(
        result=result,
        summary=summary,
        config=config,
        benchmark_weight_method=benchmark_weight_method,
        stock_ticker_count=len(stock_tickers),
        start_date=stock_prices.index.min().date(),
        end_date=stock_prices.index.max().date(),
    )
    print(f"\nSaved outputs to {PROCESSED_DATA_DIR}")
    if benchmark_weight_method == "market_cap_static":
        print(f"Saved current market caps to {CURRENT_MARKET_CAPS_PATH}")
        print(f"Saved static market-cap weights to {STATIC_MARKET_CAP_WEIGHTS_PATH}")
    print(
        "Saved equal-weight benchmark returns to "
        f"{EQUAL_WEIGHT_BENCHMARK_RETURNS_PATH}"
    )
    print(f"Saved benchmark comparison table to {BENCHMARK_COMPARISON_PATH}")
    print(f"Saved subperiod-by-benchmark table to {SUBPERIOD_BY_BENCHMARK_PATH}")
    print(f"Saved cost sensitivity table to {COST_SENSITIVITY_PATH}")
    print(
        "Saved active budget sensitivity table to "
        f"{ACTIVE_BUDGET_SENSITIVITY_PATH}"
    )
    print(
        "Saved factor variant sensitivity table to "
        f"{FACTOR_VARIANT_SENSITIVITY_PATH}"
    )
    print(
        "Saved alpha model sensitivity table to "
        f"{ALPHA_MODEL_SENSITIVITY_PATH}"
    )
    print(f"Saved subperiod performance table to {SUBPERIOD_PERFORMANCE_PATH}")
    print(f"Saved factor IC summary table to {FACTOR_IC_SUMMARY_PATH}")
    print(f"Saved factor IC subperiod table to {FACTOR_IC_SUBPERIOD_PATH}")
    print(f"Saved momentum IC sensitivity table to {MOMENTUM_IC_SENSITIVITY_PATH}")
    print(
        "Saved momentum IC sensitivity subperiod table to "
        f"{MOMENTUM_IC_SENSITIVITY_SUBPERIOD_PATH}"
    )
    print(
        "Saved momentum backtest sensitivity table to "
        f"{MOMENTUM_BACKTEST_SENSITIVITY_PATH}"
    )
    print(f"Saved alpha scores to {ALPHA_SCORES_PATH}")
    print(f"Saved active exposure diagnostics to {ACTIVE_EXPOSURE_DIAGNOSTICS_PATH}")
    print(f"Saved active exposure summary to {ACTIVE_EXPOSURE_SUMMARY_PATH}")
    print(f"Saved figures to {FIGURES_DIR}")
    print(f"Saved summary report to {SUMMARY_REPORT_PATH}")


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if not isinstance(loaded, dict):
        return {}

    return loaded


def _get_config(config: dict[str, Any], keys: list[str], default: Any) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]

    return default if value is None else value


def _get_config_exact(config: dict[str, Any], keys: list[str], default: Any) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]

    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}

    return bool(value)


def _print_effective_constraints(
    benchmark_weight_method: str,
    active_budget: float,
    max_active_weight: Optional[float],
    max_weight: Optional[float],
    max_turnover: Optional[float],
) -> None:
    print("\nEffective Portfolio Constraints")
    print("===============================")
    print(f"benchmark_weight_method: {benchmark_weight_method}")
    print(f"active_budget: {_format_constraint(active_budget)}")
    print(f"max_active_weight: {_format_constraint(max_active_weight)}")
    print(f"max_weight: {_format_constraint(max_weight)}")
    print(f"max_turnover: {_format_constraint(max_turnover)}")


def _format_constraint(value: Any) -> str:
    if value is None or pd.isna(value):
        return "None"

    return f"{float(value):.6f}"


def _get_max_turnover(config: dict[str, Any]) -> Optional[float]:
    max_turnover = _get_config(config, ["portfolio", "max_turnover"], None)
    if max_turnover is not None:
        return max_turnover

    return _get_config(config, ["portfolio", "max_monthly_turnover"], None)


def _get_benchmark_weight_method(config: dict[str, Any]) -> str:
    method = str(
        _get_config(config, ["portfolio", "benchmark_weight_method"], "equal_weight")
    ).strip()
    supported_methods = {"equal_weight", "market_cap_static"}
    if method not in supported_methods:
        raise ValueError(
            "portfolio.benchmark_weight_method must be one of: "
            f"{sorted(supported_methods)}"
        )

    return method


def _load_benchmark_weights(
    method: str,
    stock_tickers: list[str],
    refresh_market_caps: bool,
) -> Optional[pd.Series]:
    print(f"Benchmark weight method: {method}")

    if method == "equal_weight":
        return None

    print(
        "WARNING: Static current market caps are not point-in-time. Using them "
        "in a historical backtest introduces look-ahead bias. Treat this as a "
        "prototype benchmark proxy only."
    )
    if CURRENT_MARKET_CAPS_PATH.exists() and not refresh_market_caps:
        print(f"Loading current market caps from cache: {CURRENT_MARKET_CAPS_PATH}")
        market_caps = load_market_caps(CURRENT_MARKET_CAPS_PATH)
    else:
        print(
            "Fetching current market caps live for static market-cap "
            "benchmark proxy..."
        )
        market_caps = get_current_market_caps(stock_tickers)
        save_market_caps(market_caps, CURRENT_MARKET_CAPS_PATH)

    selected_market_caps = market_caps.reindex(stock_tickers)
    static_weights = market_cap_weights(selected_market_caps)
    static_weights = static_weights.rename("weight")

    _save_series(static_weights, STATIC_MARKET_CAP_WEIGHTS_PATH)

    print(f"Valid market-cap tickers: {len(static_weights)}")
    print("Top 10 static benchmark weights")
    print(static_weights.sort_values(ascending=False).head(10).to_string())

    return static_weights


def _active_exposure_benchmark_weights(
    benchmark_weight_method: str,
    benchmark_weights: Optional[pd.Series],
    portfolio_columns: pd.Index,
) -> pd.Series:
    if benchmark_weight_method == "market_cap_static" and benchmark_weights is not None:
        return benchmark_weights

    equal_weight = 1.0 / len(portfolio_columns)
    return pd.Series(equal_weight, index=portfolio_columns, name="equal_weight")


def _select_stock_universe(
    benchmark_ticker: str,
    max_tickers: Optional[int],
) -> list[str]:
    tickers = [
        ticker for ticker in get_current_sp500_tickers() if ticker != benchmark_ticker
    ]

    if max_tickers is None:
        return tickers

    max_ticker_count = int(max_tickers)
    if max_ticker_count <= 0:
        raise ValueError("data.max_tickers must be positive when provided.")

    return tickers[:max_ticker_count]


def _split_stock_and_benchmark_prices(
    prices: pd.DataFrame,
    stock_tickers: list[str],
    benchmark_ticker: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if benchmark_ticker not in prices.columns:
        raise ValueError(f"Benchmark ticker {benchmark_ticker} is missing from prices.")

    available_stock_tickers = [
        ticker for ticker in stock_tickers if ticker in prices.columns
    ]
    if not available_stock_tickers:
        raise ValueError("No stock tickers were available in downloaded prices.")

    stock_prices = prices.loc[:, available_stock_tickers]
    benchmark_prices = prices.loc[:, benchmark_ticker].rename(benchmark_ticker)

    return stock_prices, benchmark_prices, available_stock_tickers


def _print_run_summary(
    result: BacktestResult,
    summary: pd.Series,
    stock_tickers: list[str],
    stock_prices: pd.DataFrame,
) -> None:
    final_nav = (1.0 + result.portfolio_returns_net.dropna()).prod()
    average_turnover = result.turnover.mean()
    annualized_turnover = average_turnover * 12 if pd.notna(average_turnover) else pd.NA

    print("\nVersion 0 Backtest Summary")
    print("==========================")
    print(f"Number of stock tickers used: {len(stock_tickers)}")
    print(f"Date range: {stock_prices.index.min().date()} to {stock_prices.index.max().date()}")
    print(f"Portfolio return observations: {len(result.portfolio_returns_net)}")
    print(f"Final portfolio NAV: {final_nav:.4f}")
    print(f"Average turnover: {_format_metric(average_turnover)}")
    print(f"Annualized turnover: {_format_metric(annualized_turnover)}")
    print("\nPerformance Summary")
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))


def _format_metric(value: Any) -> str:
    if pd.isna(value):
        return "nan"

    return f"{float(value):.6f}"


def _save_outputs(
    result: BacktestResult,
    summary: pd.Series,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_series(
        result.portfolio_returns_gross,
        output_dir / "portfolio_returns_gross.csv",
    )
    _save_series(result.portfolio_returns_net, output_dir / "portfolio_returns_net.csv")
    _save_series(result.benchmark_returns, output_dir / "benchmark_returns.csv")
    result.weights.to_csv(output_dir / "weights.csv")
    _save_series(result.turnover, output_dir / "turnover.csv")
    _save_series(result.transaction_costs, output_dir / "transaction_costs.csv")
    summary.rename("value").to_csv(output_dir / "performance_summary.csv")


def _save_series(series: pd.Series, path: Path) -> None:
    name = series.name or "value"
    series.rename(name).to_csv(path, header=True)


def _save_cost_sensitivity(cost_sensitivity: pd.DataFrame, path: Path) -> None:
    _save_experiment_table(cost_sensitivity, path)


def _save_experiment_table(table: pd.DataFrame, path: Path) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def _write_reports(
    result: BacktestResult,
    summary: pd.Series,
    config: dict[str, Any],
    benchmark_weight_method: str,
    stock_ticker_count: int,
    start_date,
    end_date,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_nav(
        result.portfolio_returns_net,
        result.benchmark_returns,
        FIGURES_DIR / "nav_vs_benchmark.png",
    )
    plot_cumulative_active_return(
        result.portfolio_returns_net,
        result.benchmark_returns,
        FIGURES_DIR / "cumulative_active_return.png",
    )
    plot_drawdown(
        result.portfolio_returns_net,
        FIGURES_DIR / "drawdown.png",
        title="Strategy Net Drawdown",
    )
    plot_turnover(result.turnover, FIGURES_DIR / "turnover.png")
    plot_rolling_active_return(
        result.portfolio_returns_net,
        result.benchmark_returns,
        FIGURES_DIR / "rolling_active_return.png",
    )
    plot_rolling_tracking_error(
        result.portfolio_returns_net,
        result.benchmark_returns,
        FIGURES_DIR / "rolling_tracking_error.png",
    )
    write_summary_report(
        output_path=SUMMARY_REPORT_PATH,
        performance_summary=summary,
        config=config,
        num_tickers=stock_ticker_count,
        date_start=start_date,
        date_end=end_date,
        num_observations=len(result.portfolio_returns_net),
        final_nav=(1.0 + result.portfolio_returns_net.dropna()).prod(),
        average_turnover=result.turnover.mean(),
        annualized_turnover=result.turnover.mean() * 12,
    )
    if benchmark_weight_method == "market_cap_static":
        _append_market_cap_static_warning(SUMMARY_REPORT_PATH)


def _append_market_cap_static_warning(path: Path) -> None:
    warning = [
        "",
        "## Static Market-Cap Benchmark Warning",
        "",
        (
            "This run used static current market caps as a benchmark-weight "
            "proxy. Current market caps are not point-in-time, so using them "
            "for historical backtests introduces look-ahead bias. Treat this "
            "as a prototype benchmark proxy only."
        ),
        "",
    ]
    with path.open("a", encoding="utf-8") as file:
        file.write("\n".join(warning))


if __name__ == "__main__":
    main()
