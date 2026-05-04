import pandas as pd

from src import data_loader


def test_download_price_data_normalizes_field_first_multiindex(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=2)
    raw_data = pd.DataFrame(
        [
            [100.0, 200.0, 1_000, 2_000],
            [101.0, 201.0, 1_100, 2_100],
        ],
        index=dates,
        columns=pd.MultiIndex.from_tuples(
            [
                ("Adj Close", "AAPL"),
                ("Adj Close", "MSFT"),
                ("Volume", "AAPL"),
                ("Volume", "MSFT"),
            ]
        ),
    )

    def fake_download(tickers, start, end, auto_adjust, progress):
        assert tickers == ["AAPL", "MSFT"]
        assert start == "2024-01-01"
        assert end is None
        assert auto_adjust is False
        assert progress is False
        return raw_data

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    prices, volumes = data_loader.download_price_data(["AAPL", "MSFT"], "2024-01-01")

    assert isinstance(prices.index, pd.DatetimeIndex)
    assert list(prices.columns) == ["AAPL", "MSFT"]
    assert list(volumes.columns) == ["AAPL", "MSFT"]
    assert prices.loc[dates[0], "AAPL"] == 100.0
    assert volumes.loc[dates[1], "MSFT"] == 2_100


def test_download_price_data_normalizes_ticker_first_multiindex_and_keeps_missing(
    monkeypatch,
):
    dates = pd.date_range("2024-01-01", periods=2)
    raw_data = pd.DataFrame(
        [
            [100.0, 1_000, 200.0, 2_000],
            [None, 1_100, 201.0, 2_100],
        ],
        index=dates,
        columns=pd.MultiIndex.from_tuples(
            [
                ("AAPL", "Adj Close"),
                ("AAPL", "Volume"),
                ("MSFT", "Adj Close"),
                ("MSFT", "Volume"),
            ]
        ),
    )

    monkeypatch.setattr(
        data_loader.yf,
        "download",
        lambda tickers, start, end, auto_adjust, progress: raw_data,
    )

    prices, volumes = data_loader.download_price_data(["MSFT", "AAPL"], "2024-01-01")

    assert list(prices.columns) == ["MSFT", "AAPL"]
    assert list(volumes.columns) == ["MSFT", "AAPL"]
    assert pd.isna(prices.loc[dates[1], "AAPL"])
