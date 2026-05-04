import pandas as pd

from src.portfolio import (
    apply_active_weight_caps,
    apply_turnover_limit,
    calculate_turnover,
    construct_long_only_portfolio,
    equal_weight_benchmark,
    score_to_active_weights,
)


def test_equal_weight_benchmark_returns_weights_summing_to_one():
    scores = pd.Series({"A": 1.0, "B": None, "C": -0.5})

    weights = equal_weight_benchmark(scores)

    assert list(weights.index) == ["A", "C"]
    assert abs(weights.sum() - 1.0) < 1e-12
    assert weights.loc["A"] == 0.5
    assert weights.loc["C"] == 0.5


def test_score_to_active_weights_sums_to_zero_and_uses_active_budget():
    scores = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0, "D": None})

    active_weights = score_to_active_weights(scores, active_budget=0.20)

    assert list(active_weights.index) == ["A", "B", "C"]
    assert abs(active_weights.sum()) < 1e-12
    assert abs(active_weights.abs().sum() - 0.20) < 1e-12


def test_score_to_active_weights_returns_zero_when_scores_are_equal():
    scores = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0})

    active_weights = score_to_active_weights(scores, active_budget=0.20)

    assert active_weights.eq(0.0).all()


def test_apply_active_weight_caps_clips_and_recenters_active_weights():
    active_weights = pd.Series({"A": 0.50, "B": 0.50, "C": -1.00})

    capped = apply_active_weight_caps(active_weights, max_active_weight=0.20)

    assert capped.max() <= 0.20 + 1e-12
    assert capped.min() >= -0.20 - 1e-12
    assert abs(capped.sum()) < 1e-8


def test_construct_long_only_portfolio_returns_nonnegative_weights_summing_to_one():
    benchmark = pd.Series({"A": 0.50, "B": 0.50})
    active = pd.Series({"A": 0.20, "B": -0.80, "C": 0.10})

    portfolio = construct_long_only_portfolio(benchmark, active)

    assert (portfolio >= 0).all()
    assert abs(portfolio.sum() - 1.0) < 1e-12
    assert set(portfolio.index) == {"A", "B", "C"}


def test_construct_long_only_portfolio_respects_max_weight_when_feasible():
    benchmark = pd.Series({"A": 0.70, "B": 0.20, "C": 0.10})
    active = pd.Series({"A": 0.20, "B": -0.10, "C": -0.10})

    portfolio = construct_long_only_portfolio(benchmark, active, max_weight=0.60)

    assert (portfolio >= 0).all()
    assert abs(portfolio.sum() - 1.0) < 1e-12
    assert portfolio.max() <= 0.60 + 1e-12


def test_calculate_turnover_works_with_aligned_tickers():
    old_weights = pd.Series({"A": 0.50, "B": 0.50})
    new_weights = pd.Series({"A": 0.25, "B": 0.75})

    turnover = calculate_turnover(old_weights, new_weights)

    assert turnover == 0.25


def test_calculate_turnover_works_with_misaligned_tickers():
    old_weights = pd.Series({"A": 0.60, "B": 0.40})
    new_weights = pd.Series({"B": 0.25, "C": 0.75})

    turnover = calculate_turnover(old_weights, new_weights)

    assert turnover == 0.75


def test_apply_turnover_limit_reduces_turnover_when_raw_turnover_is_too_high():
    current = pd.Series({"A": 1.0, "B": 0.0})
    target = pd.Series({"A": 0.0, "B": 1.0})

    limited = apply_turnover_limit(current, target, max_turnover=0.25)
    limited_turnover = calculate_turnover(current, limited)

    assert abs(limited.sum() - 1.0) < 1e-12
    assert limited_turnover <= 0.25 + 1e-12
    assert limited_turnover < calculate_turnover(current, target)
