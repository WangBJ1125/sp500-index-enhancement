import pandas as pd

from src.signal_processing import (
    combine_factors,
    winsorize_cross_section,
    zscore_cross_section,
)


def test_winsorize_cross_section_caps_extreme_values_by_date():
    factor = pd.DataFrame(
        {
            "A": [0.0, None],
            "B": [10.0, 0.0],
            "C": [20.0, 10.0],
            "D": [30.0, 20.0],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )

    winsorized = winsorize_cross_section(factor, lower=0.25, upper=0.75)

    expected = pd.DataFrame(
        {
            "A": [7.5, None],
            "B": [10.0, 5.0],
            "C": [20.0, 10.0],
            "D": [22.5, 15.0],
        },
        index=factor.index,
    )
    pd.testing.assert_frame_equal(winsorized, expected)


def test_zscore_cross_section_uses_columns_not_time_series():
    factor = pd.DataFrame(
        {
            "A": [1.0, 10.0],
            "B": [2.0, 20.0],
            "C": [3.0, 30.0],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )

    zscores = zscore_cross_section(factor)

    expected = pd.DataFrame(
        {
            "A": [-1.0, -1.0],
            "B": [0.0, 0.0],
            "C": [1.0, 1.0],
        },
        index=factor.index,
    )
    pd.testing.assert_frame_equal(zscores, expected)


def test_zscore_cross_section_returns_nan_when_row_std_is_zero():
    factor = pd.DataFrame(
        {
            "A": [5.0, 1.0],
            "B": [5.0, 2.0],
            "C": [5.0, 3.0],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )

    zscores = zscore_cross_section(factor)

    assert zscores.loc["2024-01-31"].isna().all()
    assert zscores.loc["2024-02-29"].notna().all()


def test_combine_factors_uses_equal_weights_by_default():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    factor_one = pd.DataFrame({"A": [1.0, 3.0], "B": [2.0, 4.0]}, index=dates)
    factor_two = pd.DataFrame({"A": [5.0, 7.0], "B": [6.0, 8.0]}, index=dates)

    composite = combine_factors(
        {"one": factor_one, "two": factor_two},
        standardize_composite=False,
    )

    expected = pd.DataFrame({"A": [3.0, 5.0], "B": [4.0, 6.0]}, index=dates)
    pd.testing.assert_frame_equal(composite, expected)


def test_combine_factors_aligns_common_dates_and_tickers():
    factor_one = pd.DataFrame(
        {
            "A": [1.0, 2.0],
            "B": [3.0, 4.0],
            "C": [5.0, 6.0],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    factor_two = pd.DataFrame(
        {
            "B": [10.0, 20.0],
            "C": [30.0, 40.0],
            "D": [50.0, 60.0],
        },
        index=pd.to_datetime(["2024-02-29", "2024-03-31"]),
    )

    composite = combine_factors(
        {"one": factor_one, "two": factor_two},
        standardize_composite=False,
    )

    expected = pd.DataFrame(
        {
            "B": [7.0],
            "C": [18.0],
        },
        index=pd.to_datetime(["2024-02-29"]),
    )
    pd.testing.assert_frame_equal(composite, expected)
