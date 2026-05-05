"""Reporting and plotting helpers for Version 0 research outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib
import pandas as pd


matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_nav(
    portfolio_returns_net: pd.Series,
    benchmark_returns: pd.Series,
    output_path: str | Path,
) -> None:
    """Plot cumulative NAV for strategy and benchmark."""
    portfolio, benchmark = _align_returns(portfolio_returns_net, benchmark_returns)
    nav = pd.DataFrame(
        {
            "Strategy Net": _nav(portfolio),
            "Benchmark": _nav(benchmark),
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
    cumulative_active = (portfolio - benchmark).cumsum()

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
    nav = _nav(returns.dropna())
    drawdown = nav / nav.cummax() - 1.0

    _line_plot(drawdown.rename("Drawdown"), title, "Drawdown", output_path)


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
    rolling_active = (portfolio - benchmark).rolling(window).sum()

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
    turnover: pd.Series,
    transaction_cost_bps: float,
    benchmark_name: str = "SPY",
    universe_description: str = "current S&P 500 constituents",
    notes: Optional[dict[str, Any]] = None,
) -> None:
    """Write a markdown summary report for the Version 0 backtest."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notes = notes or {}

    average_turnover = turnover.dropna().mean()
    annualized_turnover = average_turnover * 12 if pd.notna(average_turnover) else pd.NA
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
        "## Universe Limitation",
        "",
        (
            "This prototype uses the current S&P 500 constituents as a static "
            "universe. This introduces survivorship bias because stocks that "
            "were previously in the index but later removed are not included. "
            "Results should be interpreted as research-prototype evidence, not "
            "production-level evidence."
        ),
        "",
        f"Universe used: {universe_description}.",
        "",
        "## Benchmark Limitation",
        "",
        (
            f"The benchmark return series is proxied by {benchmark_name}. The "
            "portfolio construction currently starts from an equal-weight stock "
            "benchmark proxy, which is not the true historical S&P 500 weight "
            "structure."
        ),
        "",
        "## Performance Summary",
        "",
        _series_to_markdown_table(performance_summary),
        "",
        "## Turnover Summary",
        "",
        f"- Average turnover: {_format_value(average_turnover)}",
        f"- Annualized turnover: {_format_value(annualized_turnover)}",
        "",
        "## Transaction Cost Assumption",
        "",
        f"- Transaction cost: {transaction_cost_bps:g} bps one-way",
        "- Cost convention: `2 * one_way_cost * turnover`",
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
    ]

    if notes:
        report.extend(["", "## Run Details", ""])
        for key, value in notes.items():
            report.append(f"- {key}: {value}")

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


def _nav(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.dropna()).cumprod()


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
