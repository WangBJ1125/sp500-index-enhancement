"""Performance analytics for simple-return backtests."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def total_return(returns: pd.Series) -> float:
    """Compute compounded total return from periodic simple returns."""
    clean_returns = _clean_returns(returns)

    if clean_returns.empty:
        return np.nan

    return (1.0 + clean_returns).prod() - 1.0


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute compounded annualized return."""
    clean_returns = _clean_returns(returns)
    
    if clean_returns.empty:
        return np.nan

    compounded_growth = (1.0 + clean_returns).prod()
    return compounded_growth ** (periods_per_year / len(clean_returns)) - 1.0


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized volatility of periodic simple returns."""
    clean_returns = _clean_returns(returns)
    if clean_returns.empty:
        return np.nan

    return clean_returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio using an annualized risk-free rate."""
    clean_returns = _clean_returns(returns)
    if clean_returns.empty:
        return np.nan

    period_risk_free_rate = risk_free_rate / periods_per_year
    excess_returns = clean_returns - period_risk_free_rate
    excess_volatility = excess_returns.std()
    if np.isclose(excess_volatility, 0.0) or pd.isna(excess_volatility):
        return np.nan

    return excess_returns.mean() / excess_volatility * np.sqrt(periods_per_year)


def active_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """Align portfolio and benchmark returns, then compute active returns."""
    aligned_portfolio, aligned_benchmark = portfolio_returns.align(
        benchmark_returns,
        join="inner",
    )

    return aligned_portfolio - aligned_benchmark


def tracking_error(
    active_return_series: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized tracking error."""
    clean_active_returns = _clean_returns(active_return_series)
    if clean_active_returns.empty:
        return np.nan

    return clean_active_returns.std() * np.sqrt(periods_per_year)


def information_ratio(
    active_return_series: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute information ratio from active returns."""
    clean_active_returns = _clean_returns(active_return_series)
    if clean_active_returns.empty:
        return np.nan

    te = tracking_error(clean_active_returns, periods_per_year=periods_per_year)
    if np.isclose(te, 0.0) or pd.isna(te):
        return np.nan

    annualized_active_return = periods_per_year * clean_active_returns.mean()
    return annualized_active_return / te


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown from periodic simple returns."""
    clean_returns = _clean_returns(returns)
    if clean_returns.empty:
        return np.nan

    nav = (1.0 + clean_returns).cumprod()
    drawdowns = nav / nav.cummax() - 1.0

    return drawdowns.min()


def hit_ratio(returns: pd.Series) -> float:
    """Compute the proportion of periods with positive returns."""
    clean_returns = _clean_returns(returns)
    if clean_returns.empty:
        return np.nan

    return (clean_returns > 0).mean()


def performance_summary(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    periods_per_year: int = 252,
) -> pd.Series:
    """Summarize absolute and, optionally, benchmark-relative performance."""
    summary = {
        "total_return": total_return(portfolio_returns),
        "annualized_return": annualized_return(
            portfolio_returns,
            periods_per_year=periods_per_year,
        ),
        "annualized_volatility": annualized_volatility(
            portfolio_returns,
            periods_per_year=periods_per_year,
        ),
        "sharpe_ratio": sharpe_ratio(
            portfolio_returns,
            periods_per_year=periods_per_year,
        ),
        "max_drawdown": max_drawdown(portfolio_returns),
        "hit_ratio": hit_ratio(portfolio_returns),
    }

    if benchmark_returns is not None:
        active_return_series = active_returns(portfolio_returns, benchmark_returns)
        clean_active_returns = _clean_returns(active_return_series)
        summary.update(
            {
                "annualized_active_return": (
                    periods_per_year * clean_active_returns.mean()
                    if not clean_active_returns.empty
                    else np.nan
                ),
                "tracking_error": tracking_error(
                    active_return_series,
                    periods_per_year=periods_per_year,
                ),
                "information_ratio": information_ratio(
                    active_return_series,
                    periods_per_year=periods_per_year,
                ),
            }
        )

    return pd.Series(summary, dtype=float)


def _clean_returns(returns: pd.Series) -> pd.Series:
    return returns.dropna().astype(float)
