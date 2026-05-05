"""Run the minimal Version 0 S&P 500 index-enhancement pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

from src.backtester import BacktestResult, run_backtest
from src.data_loader import download_price_data
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
from src.signal_processing import combine_factors
from src.universe import get_current_sp500_tickers
from src.utils import (
    compute_log_returns,
    compute_simple_returns,
    get_month_end_rebalance_dates,
)


CONFIG_PATH = Path("config.yaml")
PROCESSED_DATA_DIR = Path("data") / "processed"
FIGURES_DIR = Path("reports") / "figures"
SUMMARY_REPORT_PATH = Path("reports") / "summary_report.md"


def main() -> None:
    """Execute the Version 0 research pipeline end to end."""
    config = _load_config(CONFIG_PATH)

    benchmark_ticker = str(_get_config(config, ["data", "benchmark"], "SPY")).strip()
    start_date = _get_config(config, ["data", "start_date"], "2015-01-01")
    end_date = _get_config(config, ["data", "end_date"], None)
    max_tickers = _get_config(config, ["data", "max_tickers"], None)

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

    print("Computing returns and factors...")
    stock_simple_returns = compute_simple_returns(stock_prices)
    benchmark_simple_returns = compute_simple_returns(benchmark_prices)
    stock_log_returns = compute_log_returns(stock_prices)

    momentum = compute_momentum_12_1(
        stock_prices,
        lookback_days=252,
        skip_days=21,
        use_log=True,
    )
    low_volatility = compute_low_volatility(stock_log_returns, window=126)
    reversal = compute_short_term_reversal(stock_prices, window=21, use_log=True)
    composite_scores = combine_factors(
        {
            "momentum": momentum,
            "low_volatility": low_volatility,
            "reversal": reversal,
        },
        weights=None,
        standardize_composite=True,
    )

    rebalance_dates = get_month_end_rebalance_dates(stock_prices.index)

    print(f"Running backtest across {len(rebalance_dates)} rebalance dates...")
    result = run_backtest(
        stock_returns=stock_simple_returns,
        benchmark_returns=benchmark_simple_returns,
        scores=composite_scores,
        rebalance_dates=rebalance_dates,
        active_budget=_get_config(config, ["portfolio", "active_budget"], 0.20),
        transaction_cost_bps=_get_config(
            config,
            ["costs", "transaction_cost_bps"],
            10.0,
        ),
        max_active_weight=_get_config(
            config,
            ["portfolio", "max_active_weight"],
            None,
        ),
        max_weight=_get_config(config, ["portfolio", "max_weight"], 0.03),
        max_turnover=_get_max_turnover(config),
        trade_lag_days=_get_config(config, ["backtest", "trade_lag_days"], 1),
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
    _write_reports(
        result=result,
        summary=summary,
        transaction_cost_bps=_get_config(
            config,
            ["costs", "transaction_cost_bps"],
            10.0,
        ),
        benchmark_ticker=benchmark_ticker,
        stock_ticker_count=len(stock_tickers),
        start_date=stock_prices.index.min().date(),
        end_date=stock_prices.index.max().date(),
    )
    print(f"\nSaved outputs to {PROCESSED_DATA_DIR}")
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


def _get_max_turnover(config: dict[str, Any]) -> Optional[float]:
    max_turnover = _get_config(config, ["portfolio", "max_turnover"], None)
    if max_turnover is not None:
        return max_turnover

    return _get_config(config, ["portfolio", "max_monthly_turnover"], None)


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


def _write_reports(
    result: BacktestResult,
    summary: pd.Series,
    transaction_cost_bps: float,
    benchmark_ticker: str,
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
        turnover=result.turnover,
        transaction_cost_bps=transaction_cost_bps,
        benchmark_name=benchmark_ticker,
        notes={
            "Stock tickers used": stock_ticker_count,
            "Date range": f"{start_date} to {end_date}",
            "Portfolio return observations": len(result.portfolio_returns_net),
        },
    )


if __name__ == "__main__":
    main()
