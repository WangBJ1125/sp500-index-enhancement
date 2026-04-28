from io import StringIO

import pandas as pd

from src import universe


class FakeResponse:
    def __init__(self, text="<html></html>"):
        self.text = text
        self.raise_for_status_called = False

    def raise_for_status(self):
        self.raise_for_status_called = True


def test_get_current_sp500_tickers_cleans_deduplicates_and_sorts(monkeypatch):
    response = FakeResponse()
    tables = [
        pd.DataFrame({"Other": ["ignore"]}),
        pd.DataFrame({"Symbol": ["MSFT", "BRK.B", "AAPL", "BRK-B", " A "]}),
    ]

    def fake_get(url, headers):
        assert url == universe.WIKIPEDIA_SP500_URL
        assert "User-Agent" in headers
        return response

    def fake_read_html(html):
        assert isinstance(html, StringIO)
        assert html.getvalue() == response.text
        assert response.raise_for_status_called
        return tables

    monkeypatch.setattr(universe.requests, "get", fake_get)
    monkeypatch.setattr(universe.pd, "read_html", fake_read_html)

    assert universe.get_current_sp500_tickers() == ["A", "AAPL", "BRK-B", "MSFT"]


def test_get_current_sp500_tickers_raises_when_symbol_table_missing(monkeypatch):
    monkeypatch.setattr(universe.requests, "get", lambda url, headers: FakeResponse())
    monkeypatch.setattr(
        universe.pd,
        "read_html",
        lambda html: [pd.DataFrame({"Ticker": ["AAPL"]})],
    )

    try:
        universe.get_current_sp500_tickers()
    except ValueError as exc:
        assert "S&P 500 constituents table" in str(exc)
    else:
        raise AssertionError("Expected ValueError when Symbol column is missing.")
