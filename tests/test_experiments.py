import types

import pandas as pd
import src.experiments as experiments

from src.experiments import (
    compare_benchmarks,
    compute_equal_weight_benchmark_returns,
    run_active_budget_sensitivity,
    run_alpha_model_sensitivity,
    run_cost_sensitivity,
    run_factor_variant_sensitivity,
    run_momentum_backtest_sensitivity,
    run_momentum_ic_sensitivity,
    run_subperiod_analysis_by_benchmark,
    run_subperiod_analysis,
)


def _synthetic_inputs():
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )
    stock_returns = pd.DataFrame(
        {
            "A": [1.00, 0.10, 0.00, 0.00],
            "B": [0.00, 0.00, 0.10, 0.10],
        },
        index=dates,
    )
    benchmark_returns = pd.Series(
        [0.01, 0.02, 0.03, 0.04],
        index=dates,
        name="SPY",
    )
    scores = pd.DataFrame(
        {
            "A": [1.0, -1.0],
            "B": [-1.0, 1.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )
    rebalance_dates = pd.to_datetime(["2024-01-02", "2024-01-04"])

    return stock_returns, benchmark_returns, scores, rebalance_dates


def _synthetic_factor_dict():
    dates = pd.to_datetime(["2024-01-02", "2024-01-04"])
    return {
        "momentum": pd.DataFrame(
            {
                "A": [1.0, -1.0],
                "B": [-1.0, 1.0],
            },
            index=dates,
        ),
        "lowvol": pd.DataFrame(
            {
                "A": [-1.0, 1.0],
                "B": [1.0, -1.0],
            },
            index=dates,
        ),
        "reversal": pd.DataFrame(
            {
                "A": [0.5, 0.25],
                "B": [-0.5, -0.25],
            },
            index=dates,
        ),
    }


def _synthetic_momentum_prices():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    prices = pd.DataFrame(
        {
            "A": [100.0, 120.0, 144.0],
            "B": [100.0, 110.0, 121.0],
            "C": [100.0, 100.0, 100.0],
        },
        index=dates,
    )
    return prices, dates[1:]


def _fake_backtest_result():
    dates = pd.to_datetime(["2024-01-03"])
    return types.SimpleNamespace(
        portfolio_returns_net=pd.Series([0.01], index=dates),
        benchmark_returns=pd.Series([0.00], index=dates),
        turnover=pd.Series([0.10], index=dates),
    )


def _fake_backtest_result_with_weights():
    dates = pd.to_datetime(["2024-01-03"])
    return types.SimpleNamespace(
        portfolio_returns_net=pd.Series([0.01], index=dates),
        benchmark_returns=pd.Series([0.00], index=dates),
        turnover=pd.Series([0.10], index=dates),
        weights=pd.DataFrame({"A": [0.55], "B": [0.45]}, index=dates),
    )


def test_cost_sensitivity_passes_benchmark_weights_to_backtest(monkeypatch):
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()
    benchmark_weights = pd.Series({"A": 0.8, "B": 0.2})
    captured = []

    def fake_run_backtest(**kwargs):
        captured.append(kwargs["benchmark_weights"])
        return _fake_backtest_result()

    monkeypatch.setattr(experiments, "run_backtest", fake_run_backtest)

    run_cost_sensitivity(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        scores=scores,
        rebalance_dates=rebalance_dates,
        cost_bps_list=[0, 10],
        benchmark_weights=benchmark_weights,
    )

    assert len(captured) == 2
    assert all(item is benchmark_weights for item in captured)


def test_active_budget_sensitivity_passes_benchmark_weights_to_backtest(monkeypatch):
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()
    benchmark_weights = pd.Series({"A": 0.8, "B": 0.2})
    captured = []

    def fake_run_backtest(**kwargs):
        captured.append(kwargs["benchmark_weights"])
        return _fake_backtest_result()

    monkeypatch.setattr(experiments, "run_backtest", fake_run_backtest)

    run_active_budget_sensitivity(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        scores=scores,
        rebalance_dates=rebalance_dates,
        active_budget_list=[0.10, 0.20],
        benchmark_weights=benchmark_weights,
    )

    assert len(captured) == 2
    assert all(item is benchmark_weights for item in captured)


def test_factor_variant_sensitivity_passes_benchmark_weights_to_backtest(monkeypatch):
    stock_returns, benchmark_returns, _scores, rebalance_dates = _synthetic_inputs()
    benchmark_weights = pd.Series({"A": 0.8, "B": 0.2})
    captured = []

    def fake_run_backtest(**kwargs):
        captured.append(kwargs["benchmark_weights"])
        return _fake_backtest_result()

    monkeypatch.setattr(experiments, "run_backtest", fake_run_backtest)

    run_factor_variant_sensitivity(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        factor_dict=_synthetic_factor_dict(),
        rebalance_dates=rebalance_dates,
        variants={
            "momentum_only": ["momentum"],
            "combined": ["momentum", "lowvol", "reversal"],
        },
        benchmark_weights=benchmark_weights,
    )

    assert len(captured) == 2
    assert all(item is benchmark_weights for item in captured)


def test_cost_sensitivity_returns_one_row_per_cost():
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()

    result = run_cost_sensitivity(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        scores=scores,
        rebalance_dates=rebalance_dates,
        cost_bps_list=[0, 5, 10, 20],
    )

    expected_columns = {
        "cost_bps",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "hit_ratio",
        "annualized_active_return",
        "tracking_error",
        "information_ratio",
        "average_turnover",
        "annualized_turnover",
    }
    assert len(result) == 4
    assert expected_columns.issubset(set(result.columns))
    assert result["cost_bps"].tolist() == [0.0, 5.0, 10.0, 20.0]


def test_higher_transaction_cost_does_not_improve_net_total_return():
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()

    result = run_cost_sensitivity(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        scores=scores,
        rebalance_dates=rebalance_dates,
        cost_bps_list=[0, 20],
    ).set_index("cost_bps")

    assert result.loc[20.0, "total_return"] <= result.loc[0.0, "total_return"]


def test_active_budget_sensitivity_returns_one_row_per_budget():
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()

    result = run_active_budget_sensitivity(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        scores=scores,
        rebalance_dates=rebalance_dates,
        active_budget_list=[0.10, 0.20, 0.30],
    )

    expected_columns = {
        "active_budget",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "hit_ratio",
        "annualized_active_return",
        "tracking_error",
        "information_ratio",
        "average_turnover",
        "annualized_turnover",
    }
    assert len(result) == 3
    assert expected_columns.issubset(set(result.columns))


def test_active_budget_values_appear_in_output():
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()

    result = run_active_budget_sensitivity(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        scores=scores,
        rebalance_dates=rebalance_dates,
        active_budget_list=[0.10, 0.20, 0.30],
    )

    assert result["active_budget"].tolist() == [0.10, 0.20, 0.30]


def test_higher_active_budget_produces_higher_or_equal_turnover():
    stock_returns, benchmark_returns, scores, rebalance_dates = _synthetic_inputs()

    result = run_active_budget_sensitivity(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        scores=scores,
        rebalance_dates=rebalance_dates,
        active_budget_list=[0.0, 0.10, 0.20],
        transaction_cost_bps=0.0,
    )

    turnover = result["average_turnover"]
    assert turnover.is_monotonic_increasing


def test_factor_variant_sensitivity_returns_one_row_per_variant():
    stock_returns, benchmark_returns, _scores, rebalance_dates = _synthetic_inputs()
    variants = {
        "momentum_only": ["momentum"],
        "combined": ["momentum", "lowvol", "reversal"],
    }

    result = run_factor_variant_sensitivity(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        factor_dict=_synthetic_factor_dict(),
        rebalance_dates=rebalance_dates,
        variants=variants,
    )

    assert len(result) == 2
    assert result["variant"].tolist() == ["momentum_only", "combined"]


def test_factor_variant_sensitivity_records_specified_factor_names():
    stock_returns, benchmark_returns, _scores, rebalance_dates = _synthetic_inputs()
    variants = {
        "momentum_lowvol": ["momentum", "lowvol"],
        "reversal_only": ["reversal"],
    }

    result = run_factor_variant_sensitivity(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        factor_dict=_synthetic_factor_dict(),
        rebalance_dates=rebalance_dates,
        variants=variants,
    ).set_index("variant")

    assert result.loc["momentum_lowvol", "factors"] == "momentum,lowvol"
    assert result.loc["reversal_only", "factors"] == "reversal"


def test_factor_variant_sensitivity_includes_active_metrics():
    stock_returns, benchmark_returns, _scores, rebalance_dates = _synthetic_inputs()

    result = run_factor_variant_sensitivity(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        factor_dict=_synthetic_factor_dict(),
        rebalance_dates=rebalance_dates,
        variants={"momentum_only": ["momentum"]},
    )

    assert "information_ratio" in result.columns
    assert "annualized_active_return" in result.columns


def test_alpha_model_sensitivity_returns_one_row_per_alpha_model_config(monkeypatch):
    stock_returns, benchmark_returns, _scores, rebalance_dates = _synthetic_inputs()
    alpha_model_configs = [
        {"model": "equal_weight_composite"},
        {"model": "momentum_only"},
        {"model": "risk_adjusted_momentum", "lowvol_penalty_weight": 0.25},
    ]

    def fake_run_backtest(**kwargs):
        return _fake_backtest_result()

    monkeypatch.setattr(experiments, "run_backtest", fake_run_backtest)

    result = run_alpha_model_sensitivity(
        factor_scores=_synthetic_factor_dict(),
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        rebalance_dates=rebalance_dates,
        alpha_model_configs=alpha_model_configs,
    )

    assert len(result) == 3
    assert result["alpha_model"].tolist() == [
        "equal_weight_composite",
        "momentum_only",
        "risk_adjusted_momentum",
    ]


def test_alpha_model_sensitivity_calls_build_alpha_score_for_each_config(
    monkeypatch,
):
    stock_returns, benchmark_returns, _scores, rebalance_dates = _synthetic_inputs()
    factor_scores = _synthetic_factor_dict()
    captured_models = []

    def fake_build_alpha_score(factor_scores_arg, **kwargs):
        captured_models.append(kwargs["model"])
        return factor_scores_arg["momentum"]

    def fake_run_backtest(**kwargs):
        return _fake_backtest_result()

    monkeypatch.setattr(experiments, "build_alpha_score", fake_build_alpha_score)
    monkeypatch.setattr(experiments, "run_backtest", fake_run_backtest)

    run_alpha_model_sensitivity(
        factor_scores=factor_scores,
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        rebalance_dates=rebalance_dates,
        alpha_model_configs=[
            {"model": "momentum_only"},
            {"model": "momentum_reversal", "reversal_weight": 0.20},
        ],
    )

    assert captured_models == ["momentum_only", "momentum_reversal"]


def test_alpha_model_sensitivity_includes_turnover_columns(monkeypatch):
    stock_returns, benchmark_returns, _scores, rebalance_dates = _synthetic_inputs()

    def fake_run_backtest(**kwargs):
        return _fake_backtest_result()

    monkeypatch.setattr(experiments, "run_backtest", fake_run_backtest)

    result = run_alpha_model_sensitivity(
        factor_scores=_synthetic_factor_dict(),
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        rebalance_dates=rebalance_dates,
        alpha_model_configs=[{"model": "momentum_only"}],
    )

    assert "average_turnover" in result.columns
    assert "annualized_turnover" in result.columns
    assert result.loc[0, "average_turnover"] == 0.10
    assert abs(result.loc[0, "annualized_turnover"] - 1.20) < 1e-12


def test_alpha_model_sensitivity_includes_active_exposure_with_benchmark_weights(
    monkeypatch,
):
    stock_returns, benchmark_returns, _scores, rebalance_dates = _synthetic_inputs()
    benchmark_weights = pd.Series({"A": 0.50, "B": 0.50})

    def fake_run_backtest(**kwargs):
        return _fake_backtest_result_with_weights()

    monkeypatch.setattr(experiments, "run_backtest", fake_run_backtest)

    result = run_alpha_model_sensitivity(
        factor_scores=_synthetic_factor_dict(),
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        rebalance_dates=rebalance_dates,
        alpha_model_configs=[{"model": "momentum_only"}],
        benchmark_weights=benchmark_weights,
    )

    assert "average_active_share" in result.columns
    assert "max_active_share" in result.columns
    assert "max_absolute_active_weight" in result.columns
    assert abs(result.loc[0, "average_active_share"] - 0.05) < 1e-12
    assert abs(result.loc[0, "max_absolute_active_weight"] - 0.05) < 1e-12


def test_alpha_model_sensitivity_handles_missing_active_exposure_gracefully(
    monkeypatch,
):
    stock_returns, benchmark_returns, _scores, rebalance_dates = _synthetic_inputs()

    def fake_run_backtest(**kwargs):
        return _fake_backtest_result()

    monkeypatch.setattr(experiments, "run_backtest", fake_run_backtest)

    result = run_alpha_model_sensitivity(
        factor_scores=_synthetic_factor_dict(),
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        rebalance_dates=rebalance_dates,
        alpha_model_configs=[{"model": "momentum_only"}],
        benchmark_weights=None,
    )

    assert pd.isna(result.loc[0, "average_active_share"])
    assert pd.isna(result.loc[0, "max_active_share"])
    assert pd.isna(result.loc[0, "max_absolute_active_weight"])


def test_subperiod_analysis_returns_one_row_per_period():
    portfolio_returns = pd.Series(
        [0.01, 0.02, -0.01],
        index=pd.to_datetime(["2020-01-02", "2021-01-04", "2023-01-03"]),
    )
    benchmark_returns = pd.Series(
        [0.005, 0.01, -0.02],
        index=portfolio_returns.index,
    )
    periods = {
        "early": ("2020-01-01", "2020-12-31"),
        "later": ("2021-01-01", None),
    }

    result = run_subperiod_analysis(portfolio_returns, benchmark_returns, periods)

    assert len(result) == 2
    assert result["period"].tolist() == ["early", "later"]


def test_subperiod_analysis_slices_dates_correctly():
    portfolio_returns = pd.Series(
        [0.01, 0.02, -0.01],
        index=pd.to_datetime(["2020-01-02", "2020-06-01", "2021-01-04"]),
    )
    benchmark_returns = pd.Series(
        [0.005, 0.01, -0.02],
        index=portfolio_returns.index,
    )
    periods = {"first_half_2020": ("2020-01-01", "2020-06-30")}

    result = run_subperiod_analysis(
        portfolio_returns,
        benchmark_returns,
        periods,
    ).set_index("period")

    assert result.loc["first_half_2020", "num_observations"] == 2
    assert abs(result.loc["first_half_2020", "total_return"] - 0.0302) < 1e-12


def test_subperiod_analysis_handles_empty_periods_gracefully():
    portfolio_returns = pd.Series(
        [0.01, 0.02],
        index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
    )
    benchmark_returns = pd.Series(
        [0.005, 0.01],
        index=portfolio_returns.index,
    )
    periods = {"empty": ("2019-01-01", "2019-12-31")}

    result = run_subperiod_analysis(
        portfolio_returns,
        benchmark_returns,
        periods,
    ).set_index("period")

    assert result.loc["empty", "num_observations"] == 0
    assert pd.isna(result.loc["empty", "total_return"])
    assert pd.isna(result.loc["empty", "information_ratio"])


def test_equal_weight_benchmark_returns_equal_row_mean():
    stock_returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, None],
            "B": [0.03, None, 0.04],
            "C": [-0.01, 0.01, 0.02],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    result = compute_equal_weight_benchmark_returns(stock_returns)

    expected = pd.Series(
        [0.01, 0.015, 0.03],
        index=stock_returns.index,
        name="EqualWeightUniverse",
    )
    pd.testing.assert_series_equal(result, expected)


def test_compare_benchmarks_returns_two_rows():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    portfolio_returns = pd.Series([0.01, 0.02, -0.01], index=dates)
    spy_returns = pd.Series([0.005, 0.01, -0.02], index=dates)
    equal_weight_returns = pd.Series([0.01, 0.015, -0.005], index=dates)

    result = compare_benchmarks(
        portfolio_returns,
        spy_returns,
        equal_weight_returns,
    )

    assert len(result) == 2
    assert result["benchmark"].tolist() == ["SPY", "EqualWeightUniverse"]
    assert "information_ratio" in result.columns


def test_subperiod_analysis_by_benchmark_returns_combinations():
    dates = pd.to_datetime(["2020-01-02", "2021-01-04", "2023-01-03"])
    portfolio_returns = pd.Series([0.01, 0.02, -0.01], index=dates)
    benchmark_dict = {
        "SPY": pd.Series([0.005, 0.01, -0.02], index=dates),
        "EqualWeightUniverse": pd.Series([0.01, 0.015, -0.005], index=dates),
    }
    periods = {
        "early": ("2020-01-01", "2020-12-31"),
        "later": ("2021-01-01", None),
    }

    result = run_subperiod_analysis_by_benchmark(
        portfolio_returns,
        benchmark_dict,
        periods,
    )

    combinations = set(zip(result["benchmark"], result["period"]))
    expected = {
        ("SPY", "early"),
        ("SPY", "later"),
        ("EqualWeightUniverse", "early"),
        ("EqualWeightUniverse", "later"),
    }
    assert len(result) == 4
    assert combinations == expected


def test_momentum_ic_sensitivity_returns_one_row_per_parameter_combination():
    dates = pd.to_datetime(
        ["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"]
    )
    prices = pd.DataFrame(
        {
            "A": [100.0, 120.0, 144.0, 150.0],
            "B": [100.0, 110.0, 121.0, 123.0],
            "C": [100.0, 100.0, 100.0, 101.0],
        },
        index=dates,
    )

    result = run_momentum_ic_sensitivity(
        stock_prices=prices,
        rebalance_dates=dates,
        lookback_days_list=[1, 2],
        skip_days_list=[0, 1],
    )

    assert len(result) == 4
    assert set(zip(result["lookback_days"], result["skip_days"])) == {
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    }
    assert result.columns.tolist() == [
        "lookback_days",
        "skip_days",
        "mean_ic",
        "std_ic",
        "icir",
        "hit_rate",
        "count",
        "mean_rank_ic",
        "rank_icir",
        "rank_hit_rate",
    ]


def test_momentum_ic_sensitivity_positive_synthetic_momentum_has_positive_ic():
    prices, rebalance_dates = _synthetic_momentum_prices()

    result = run_momentum_ic_sensitivity(
        stock_prices=prices,
        rebalance_dates=rebalance_dates,
        lookback_days_list=[1],
        skip_days_list=[0],
    )

    assert result.loc[0, "mean_ic"] > 0.99
    assert result.loc[0, "mean_rank_ic"] > 0.99


def test_momentum_ic_sensitivity_handles_skip_days_zero():
    prices, rebalance_dates = _synthetic_momentum_prices()

    result = run_momentum_ic_sensitivity(
        stock_prices=prices,
        rebalance_dates=rebalance_dates,
        lookback_days_list=[1],
        skip_days_list=[0],
    )

    assert result.loc[0, "skip_days"] == 0
    assert result.loc[0, "count"] == 1
    assert pd.notna(result.loc[0, "mean_ic"])


def test_momentum_ic_sensitivity_handles_skip_days_21():
    dates = pd.date_range("2024-01-01", periods=24, freq="D")
    prices = pd.DataFrame(
        100.0,
        index=dates,
        columns=["A", "B", "C"],
    )
    prices.loc[dates[1], ["A", "B", "C"]] = [120.0, 110.0, 100.0]
    prices.loc[dates[22], ["A", "B", "C"]] = [100.0, 100.0, 100.0]
    prices.loc[dates[23], ["A", "B", "C"]] = [120.0, 110.0, 100.0]

    result = run_momentum_ic_sensitivity(
        stock_prices=prices,
        rebalance_dates=pd.DatetimeIndex([dates[22], dates[23]]),
        lookback_days_list=[22],
        skip_days_list=[21],
    )

    assert result.loc[0, "skip_days"] == 21
    assert result.loc[0, "count"] == 1
    assert result.loc[0, "mean_ic"] > 0.99


def test_momentum_ic_sensitivity_returns_subperiod_table_when_periods_provided():
    prices, rebalance_dates = _synthetic_momentum_prices()
    periods = {
        "sample": ("2024-02-01", "2024-12-31"),
        "empty": ("2025-01-01", "2025-12-31"),
    }

    overall, subperiod = run_momentum_ic_sensitivity(
        stock_prices=prices,
        rebalance_dates=rebalance_dates,
        lookback_days_list=[1],
        skip_days_list=[0],
        periods=periods,
    )

    assert len(overall) == 1
    assert subperiod["period"].tolist() == ["sample", "empty"]
    assert subperiod.columns.tolist() == [
        "lookback_days",
        "skip_days",
        "period",
        "mean_ic",
        "icir",
        "hit_rate",
        "mean_rank_ic",
        "rank_icir",
        "rank_hit_rate",
        "count",
    ]
    assert subperiod.loc[subperiod["period"] == "empty", "count"].iloc[0] == 0


def test_momentum_backtest_sensitivity_returns_one_row_per_parameter_combination(
    monkeypatch,
):
    prices, rebalance_dates = _synthetic_momentum_prices()
    stock_returns = prices.pct_change()
    benchmark_returns = pd.Series(0.0, index=prices.index)

    def fake_run_backtest(**kwargs):
        return _fake_backtest_result()

    monkeypatch.setattr(experiments, "run_backtest", fake_run_backtest)

    result = run_momentum_backtest_sensitivity(
        stock_prices=prices,
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        rebalance_dates=rebalance_dates,
        lookback_days_list=[1, 2],
        skip_days_list=[0, 1],
    )

    assert len(result) == 4
    assert set(zip(result["lookback_days"], result["skip_days"])) == {
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    }
    assert result.columns.tolist() == [
        "lookback_days",
        "skip_days",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "hit_ratio",
        "annualized_active_return",
        "tracking_error",
        "information_ratio",
        "average_turnover",
        "annualized_turnover",
    ]


def test_momentum_backtest_sensitivity_passes_benchmark_weights_to_backtest(
    monkeypatch,
):
    prices, rebalance_dates = _synthetic_momentum_prices()
    stock_returns = prices.pct_change()
    benchmark_returns = pd.Series(0.0, index=prices.index)
    benchmark_weights = pd.Series({"A": 0.6, "B": 0.3, "C": 0.1})
    captured = []

    def fake_run_backtest(**kwargs):
        captured.append(kwargs["benchmark_weights"])
        return _fake_backtest_result()

    monkeypatch.setattr(experiments, "run_backtest", fake_run_backtest)

    run_momentum_backtest_sensitivity(
        stock_prices=prices,
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        rebalance_dates=rebalance_dates,
        lookback_days_list=[1],
        skip_days_list=[0],
        benchmark_weights=benchmark_weights,
    )

    assert len(captured) == 1
    assert captured[0] is benchmark_weights


def test_momentum_backtest_sensitivity_handles_skip_days_zero(monkeypatch):
    prices, rebalance_dates = _synthetic_momentum_prices()
    stock_returns = prices.pct_change()
    benchmark_returns = pd.Series(0.0, index=prices.index)

    def fake_run_backtest(**kwargs):
        return _fake_backtest_result()

    monkeypatch.setattr(experiments, "run_backtest", fake_run_backtest)

    result = run_momentum_backtest_sensitivity(
        stock_prices=prices,
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        rebalance_dates=rebalance_dates,
        lookback_days_list=[1],
        skip_days_list=[0],
    )

    assert result.loc[0, "skip_days"] == 0
    assert len(result) == 1
