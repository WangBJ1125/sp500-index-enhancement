import pandas as pd

from src.reporting import (
    compute_drawdown_series,
    compute_nav,
    plot_cumulative_active_return,
    plot_drawdown,
    plot_nav,
    plot_rolling_active_return,
    plot_rolling_tracking_error,
    plot_turnover,
    write_summary_report,
)


def test_compute_nav_compounds_simple_returns():
    returns = pd.Series([0.10, -0.10, None, 0.20])

    nav = compute_nav(returns)

    expected = pd.Series([1.10, 0.99, 1.188], index=[0, 1, 3])
    pd.testing.assert_series_equal(nav, expected, check_exact=False, atol=1e-12)


def test_compute_drawdown_series_uses_running_max_nav():
    returns = pd.Series([0.10, -0.20, 0.25])

    drawdown = compute_drawdown_series(returns)

    expected = pd.Series([0.0, -0.20, 0.0])
    pd.testing.assert_series_equal(drawdown, expected, check_exact=False, atol=1e-12)


def test_plotting_functions_create_files(tmp_path):
    dates = pd.date_range("2024-01-01", periods=6)
    portfolio_returns = pd.Series(
        [0.01, 0.02, -0.01, 0.00, 0.03, 0.01],
        index=dates,
    )
    benchmark_returns = pd.Series(
        [0.00, 0.01, -0.005, 0.01, 0.02, 0.00],
        index=dates,
    )
    turnover = pd.Series(
        [0.10, 0.20],
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )

    output_paths = [
        tmp_path / "nav.png",
        tmp_path / "active.png",
        tmp_path / "drawdown.png",
        tmp_path / "turnover.png",
        tmp_path / "rolling_active.png",
        tmp_path / "rolling_te.png",
    ]

    plot_nav(portfolio_returns, benchmark_returns, output_paths[0])
    plot_cumulative_active_return(portfolio_returns, benchmark_returns, output_paths[1])
    plot_drawdown(portfolio_returns, output_paths[2])
    plot_turnover(turnover, output_paths[3])
    plot_rolling_active_return(
        portfolio_returns,
        benchmark_returns,
        output_paths[4],
        window=3,
    )
    plot_rolling_tracking_error(
        portfolio_returns,
        benchmark_returns,
        output_paths[5],
        window=3,
        periods_per_year=252,
    )

    for output_path in output_paths:
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_write_summary_report_creates_markdown_file(tmp_path):
    performance_summary = pd.Series(
        {
            "total_return": 0.10,
            "tracking_error": 0.05,
            "information_ratio": 0.50,
        }
    )
    config = {
        "data": {"benchmark": "SPY", "start_date": "2024-01-01", "end_date": None},
        "costs": {"transaction_cost_bps": 10},
        "portfolio": {"active_budget": 0.20},
    }
    output_path = tmp_path / "summary_report.md"

    write_summary_report(
        output_path=output_path,
        performance_summary=performance_summary,
        config=config,
        num_tickers=50,
        date_start="2024-01-01",
        date_end="2024-12-31",
        num_observations=252,
        final_nav=1.10,
        average_turnover=0.08,
        annualized_turnover=0.96,
    )

    report = output_path.read_text(encoding="utf-8")
    assert "Project Objective" in report
    assert "Survivorship Bias Warning" in report
    assert "Benchmark Mismatch Warning" in report
    assert "Performance Summary" in report
