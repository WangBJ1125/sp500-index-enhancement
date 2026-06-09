"""Reporting and plotting helpers for Version 0 research outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd


matplotlib.use("Agg")

import matplotlib.pyplot as plt


def compute_nav(returns: pd.Series) -> pd.Series:
    """Compute cumulative NAV from simple returns, starting at 1.0."""
    return (1.0 + returns.dropna()).cumprod()


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown series from simple returns."""
    nav = compute_nav(returns)
    return nav / nav.cummax() - 1.0


def plot_nav(
    portfolio_returns_net: pd.Series,
    benchmark_returns: pd.Series,
    output_path: str | Path,
) -> None:
    """Plot cumulative NAV for strategy and benchmark."""
    portfolio, benchmark = _align_returns(portfolio_returns_net, benchmark_returns)
    nav = pd.DataFrame(
        {
            "Strategy Net": compute_nav(portfolio),
            "Benchmark": compute_nav(benchmark),
        }
    )

    _line_plot(nav, "Strategy NAV vs Benchmark", "NAV", output_path)


def plot_cumulative_active_return(
    portfolio_returns_net: pd.Series,
    benchmark_returns: pd.Series,
    output_path: str | Path,
) -> None:
    """Plot cumulative active return."""
    portfolio, benchmark = _align_returns(portfolio_returns_net, benchmark_returns)
    active_returns = portfolio - benchmark
    cumulative_active = (1.0 + active_returns).cumprod() - 1.0

    _line_plot(
        cumulative_active.rename("Cumulative Active Return"),
        "Cumulative Active Return",
        "Active Return",
        output_path,
    )


def plot_drawdown(
    returns: pd.Series,
    output_path: str | Path,
    title: str = "Drawdown",
) -> None:
    """Plot drawdown from a simple-return series."""
    drawdown = compute_drawdown_series(returns)
    min_dd = drawdown.min()
    if pd.isna(min_dd):
        min_dd = 0.0
    lower = min_dd * 1.10 if min_dd < 0 else -0.05
    upper = 0.02

    fig, ax = plt.subplots(figsize=(10, 5))
    drawdown.rename("Drawdown").plot(ax=ax)
    ax.axhline(min_dd, color="tab:red", linestyle="--", linewidth=1.0)
    ax.text(
        0.02,
        0.08,
        f"Max DD: {min_dd:.1%}",
        transform=ax.transAxes,
        color="tab:red",
        bbox={"facecolor": "white", "edgecolor": "tab:red", "alpha": 0.85},
    )
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.set_ylim(lower, upper)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    _save_figure(fig, output_path)


def plot_turnover(turnover: pd.Series, output_path: str | Path) -> None:
    """Plot turnover by rebalance."""
    clean_turnover = turnover.dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(clean_turnover.index, clean_turnover.values, width=20)
    ax.set_title("Turnover by Rebalance")
    ax.set_ylabel("Turnover")
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate()
    _save_figure(fig, output_path)


def plot_rolling_active_return(
    portfolio_returns_net: pd.Series,
    benchmark_returns: pd.Series,
    output_path: str | Path,
    window: int = 252,
) -> None:
    """Plot rolling active return over a trailing window."""
    portfolio, benchmark = _align_returns(portfolio_returns_net, benchmark_returns)
    active_returns = portfolio - benchmark
    rolling_active = active_returns.rolling(window).mean() * 252

    _line_plot(
        rolling_active.rename(f"Rolling {window}-Day Active Return"),
        f"Rolling {window}-Day Active Return",
        "Active Return",
        output_path,
    )


def plot_rolling_tracking_error(
    portfolio_returns_net: pd.Series,
    benchmark_returns: pd.Series,
    output_path: str | Path,
    window: int = 252,
    periods_per_year: int = 252,
) -> None:
    """Plot rolling annualized tracking error."""
    portfolio, benchmark = _align_returns(portfolio_returns_net, benchmark_returns)
    active = portfolio - benchmark
    rolling_te = active.rolling(window).std() * periods_per_year**0.5

    _line_plot(
        rolling_te.rename(f"Rolling {window}-Day Tracking Error"),
        f"Rolling {window}-Day Tracking Error",
        "Tracking Error",
        output_path,
    )


def write_summary_report(
    output_path: str | Path,
    performance_summary: pd.Series,
    config: dict[str, Any],
    num_tickers: int,
    date_start,
    date_end,
    num_observations: int,
    final_nav: float,
    average_turnover: float,
    annualized_turnover: float,
) -> None:
    """Write a markdown summary report for the Version 0 backtest."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark_name = _get_config(config, ["data", "benchmark"], "SPY")
    start_date = _get_config(config, ["data", "start_date"], date_start)
    end_date = _get_config(config, ["data", "end_date"], date_end)
    transaction_cost_bps = _get_config(config, ["costs", "transaction_cost_bps"], 10.0)
    active_budget = _get_config(config, ["portfolio", "active_budget"], 0.20)
    max_tickers = _get_config(config, ["data", "max_tickers"], None)
    total_return = performance_summary.get("total_return", pd.NA)
    information_ratio = performance_summary.get("information_ratio", pd.NA)
    tracking_error = performance_summary.get("tracking_error", pd.NA)

    report = [
        "# S&P 500 Index Enhancement Summary Report",
        "",
        "## Project Objective",
        "",
        (
            "This project tests whether transparent price-based cross-sectional "
            "factors can create a long-only portfolio with modest active tilts "
            f"versus {benchmark_name}."
        ),
        "",
        "## Version 0 Description",
        "",
        (
            "Version 0 is a simple rule-based baseline using price-based factors, "
            "monthly rebalancing, a one-day trade lag, long-only portfolio "
            "construction, turnover tracking, and transaction costs."
        ),
        "",
        "## Data and Universe",
        "",
        f"- Date range: {date_start} to {date_end}",
        f"- Configured start date: {start_date}",
        f"- Configured end date: {end_date}",
        f"- Number of stock tickers used: {num_tickers}",
        f"- Configured max tickers: {max_tickers}",
        "- Universe source: current S&P 500 constituents from Wikipedia",
        "- Price data: adjusted close and volume from yfinance",
        "",
        "## Survivorship Bias Warning",
        "",
        (
            "This prototype uses the current S&P 500 constituents as a static "
            "universe. This introduces survivorship bias because stocks that "
            "were previously in the index but later removed are not included. "
            "Results should be interpreted as research-prototype evidence, not "
            "production-level evidence."
        ),
        "",
        "## Benchmark Mismatch Warning",
        "",
        (
            f"The benchmark return series is proxied by {benchmark_name}. The "
            "portfolio construction currently starts from an equal-weight stock "
            "benchmark proxy, which is not the true historical S&P 500 weight "
            "structure."
        ),
        "",
        "## Transaction Cost Assumption",
        "",
        f"- Transaction cost: {transaction_cost_bps:g} bps one-way",
        "- Cost convention: `2 * one_way_cost * turnover`",
        f"- Active budget: {_format_value(active_budget)}",
        "",
        "## Performance Summary",
        "",
        _series_to_markdown_table(performance_summary),
        "",
        "## Turnover Summary",
        "",
        f"- Average turnover: {_format_value(average_turnover)}",
        f"- Annualized turnover: {_format_value(annualized_turnover)}",
        f"- Portfolio return observations: {num_observations}",
        f"- Final portfolio NAV: {_format_value(final_nav)}",
        "",
        "## Short Interpretation",
        "",
        (
            f"The strategy total return was {_format_value(total_return)}. "
            f"Tracking error was {_format_value(tracking_error)}, and the "
            f"information ratio was {_format_value(information_ratio)}. These "
            "results should be read together with turnover, transaction costs, "
            "and the static-universe survivorship limitation."
        ),
        "",
        "## Next Steps",
        "",
        "- Add cost sensitivity, active budget sensitivity, and factor mix sensitivity.",
        "- Compare against an equal-weight universe benchmark as well as SPY.",
        "- Add subperiod analysis and factor diagnostics such as IC and Rank IC.",
        "- Add sector exposure reporting and optional sector neutralization.",
    ]

    output_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def _align_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    aligned = pd.concat(
        [
            portfolio_returns.rename("portfolio"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    return aligned["portfolio"], aligned["benchmark"]


def _line_plot(
    data: pd.Series | pd.DataFrame,
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    data.plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    _save_figure(fig, output_path)


def _save_figure(fig, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _series_to_markdown_table(series: pd.Series) -> str:
    rows = ["| Metric | Value |", "|---|---:|"]
    for metric, value in series.items():
        rows.append(f"| {metric} | {_format_value(value)} |")

    return "\n".join(rows)


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "nan"

    return f"{float(value):.6f}"


def _get_config(config: dict[str, Any], keys: list[str], default: Any) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]

    return default if value is None else value
