"""Data loading helpers for price and volume inputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import yfinance as yf


ADJUSTED_CLOSE_FIELD = "Adj Close"
VOLUME_FIELD = "Volume"


def download_price_data(
    tickers: Iterable[str] | str,
    start: str,
    end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted close prices and volume from yfinance.

    Missing price histories are left as missing values. The function does not
    forward-fill prices because that can hide stale or unavailable security
    histories in a backtest.

    Args:
        tickers: Ticker symbol or iterable of ticker symbols.
        start: Start date accepted by yfinance.
        end: Optional end date accepted by yfinance.

    Returns:
        A pair of DataFrames: adjusted close prices and volumes. Both use a
        DatetimeIndex and ticker symbols as columns.
    """
    ticker_list = _normalize_ticker_input(tickers)
    raw_data = yf.download(
        tickers=ticker_list,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    prices = _extract_yfinance_field(raw_data, ADJUSTED_CLOSE_FIELD, ticker_list)
    volumes = _extract_yfinance_field(raw_data, VOLUME_FIELD, ticker_list)

    return _ensure_datetime_index(prices), _ensure_datetime_index(volumes)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to CSV, creating parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a CSV DataFrame saved by `save_dataframe`."""
    df = pd.read_csv(Path(path), index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


def _normalize_ticker_input(tickers: Iterable[str] | str) -> list[str]:
    if isinstance(tickers, str):
        raw_tickers = [tickers]
    else:
        raw_tickers = list(tickers)

    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in raw_tickers:
        clean_ticker = str(ticker).strip()
        if clean_ticker and clean_ticker not in seen:
            normalized.append(clean_ticker)
            seen.add(clean_ticker)

    if not normalized:
        raise ValueError("At least one ticker is required.")

    return normalized


def _extract_yfinance_field(
    raw_data: pd.DataFrame,
    field: str,
    tickers: list[str],
) -> pd.DataFrame:
    if isinstance(raw_data.columns, pd.MultiIndex):
        return _extract_multiindex_field(raw_data, field, tickers)

    if field not in raw_data.columns:
        raise ValueError(f"Downloaded data does not contain required field: {field}")
    if len(tickers) != 1:
        raise ValueError("Single-level yfinance data can only be mapped to one ticker.")

    return raw_data[[field]].rename(columns={field: tickers[0]})


def _extract_multiindex_field(
    raw_data: pd.DataFrame,
    field: str,
    tickers: list[str],
) -> pd.DataFrame:
    field_level = _find_field_level(raw_data.columns, field)
    field_data = raw_data.xs(field, axis=1, level=field_level).copy()

    if isinstance(field_data, pd.Series):
        if len(tickers) != 1:
            raise ValueError(f"Could not map {field} data to multiple tickers.")
        field_data = field_data.to_frame(name=tickers[0])

    field_data.columns = _normalize_extracted_columns(field_data.columns, tickers)
    field_data = field_data.loc[:, ~field_data.columns.duplicated()]

    return field_data.reindex(columns=tickers)


def _find_field_level(columns: pd.MultiIndex, field: str) -> int:
    for level in range(columns.nlevels):
        if field in columns.get_level_values(level):
            return level

    raise ValueError(f"Downloaded data does not contain required field: {field}")


def _normalize_extracted_columns(
    columns: pd.Index | pd.MultiIndex,
    tickers: list[str],
) -> pd.Index:
    if not isinstance(columns, pd.MultiIndex):
        return pd.Index(str(column).strip() for column in columns)

    ticker_set = set(tickers)
    for level in range(columns.nlevels):
        level_values = pd.Index(
            str(value).strip() for value in columns.get_level_values(level)
        )
        if ticker_set.intersection(level_values):
            return level_values

    return pd.Index(str(column).strip() for column in columns.get_level_values(-1))


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df
