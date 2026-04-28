"""Universe construction helpers for the S&P 500 research framework."""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd
import requests


WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKIPEDIA_SYMBOL_COLUMN = "Symbol"
WIKIPEDIA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}
UNIVERSE_TYPE = "static_current"


def _clean_yfinance_ticker(ticker: Any) -> str:
    """Convert a raw ticker value into the symbol format expected by yfinance."""
    return str(ticker).strip().replace(".", "-")


def get_current_sp500_tickers() -> list[str]:
    """Load current S&P 500 tickers from Wikipedia.

    Warning:
        This function returns a static current S&P 500 universe. It is not a
        historical point-in-time constituent list and therefore introduces
        survivorship bias in historical backtests. Results that use this
        universe should be interpreted as research-prototype evidence, not
        production-level evidence.

    Returns:
        Sorted unique ticker symbols cleaned for yfinance compatibility.
    """
    response = requests.get(WIKIPEDIA_SP500_URL, headers=WIKIPEDIA_HEADERS)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))

    constituents = next(
        (table for table in tables if WIKIPEDIA_SYMBOL_COLUMN in table.columns),
        None,
    )
    if constituents is None:
        raise ValueError("Could not find an S&P 500 constituents table on Wikipedia.")

    tickers: set[str] = set()
    for raw_ticker in constituents[WIKIPEDIA_SYMBOL_COLUMN].dropna():
        ticker = _clean_yfinance_ticker(raw_ticker)
        if ticker:
            tickers.add(ticker)

    return sorted(tickers)
