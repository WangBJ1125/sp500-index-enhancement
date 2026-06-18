"""Alpha score construction layer."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src.signal_processing import combine_factors, zscore_cross_section


def build_alpha_score(
    factor_scores: Mapping[str, pd.DataFrame],
    model: str = "equal_weight_composite",
    weights: Mapping[str, float] | None = None,
    lowvol_penalty_weight: float = 0.25,
    reversal_weight: float = 0.20,
    standardize: bool = True,
) -> pd.DataFrame:
    """Build a final cross-sectional alpha score from factor score DataFrames.

    This layer separates raw factor calculation from alpha model selection. It
    does not change factor definitions; it combines already-computed factor
    scores into the score used by portfolio construction.
    """
    if not factor_scores:
        raise ValueError("factor_scores must contain at least one factor.")

    model = model.strip().lower()

    if model == "equal_weight_composite":
        return combine_factors(
            factor_scores,
            weights=None,
            standardize_composite=standardize,
        )

    if model == "momentum_only":
        momentum = _require_factor(factor_scores, "momentum")
        return zscore_cross_section(momentum) if standardize else momentum.copy()

    if model == "risk_adjusted_momentum":
        return _combine_standardized_components(
            factor_scores=factor_scores,
            component_weights={
                "momentum": 1.0,
                "lowvol": float(lowvol_penalty_weight),
            },
            standardize=standardize,
        )

    if model == "momentum_reversal":
        return _combine_standardized_components(
            factor_scores=factor_scores,
            component_weights={
                "momentum": 1.0,
                "reversal": float(reversal_weight),
            },
            standardize=standardize,
        )

    if model == "custom_weighted":
        if weights is None:
            raise ValueError("custom_weighted alpha model requires weights.")
        selected_factors = {
            name: _require_factor(factor_scores, name) for name in weights.keys()
        }
        return combine_factors(
            selected_factors,
            weights=weights,
            standardize_composite=standardize,
        )

    raise ValueError(f"Unknown alpha model: {model}")


def _combine_standardized_components(
    factor_scores: Mapping[str, pd.DataFrame],
    component_weights: Mapping[str, float],
    standardize: bool,
) -> pd.DataFrame:
    standardized_factors = {
        name: zscore_cross_section(_require_factor(factor_scores, name))
        for name in component_weights
    }
    return combine_factors(
        standardized_factors,
        weights=component_weights,
        standardize_composite=standardize,
    )


def _require_factor(
    factor_scores: Mapping[str, pd.DataFrame],
    factor_name: str,
) -> pd.DataFrame:
    if factor_name not in factor_scores:
        raise ValueError(f"Missing required factor for alpha model: {factor_name}")

    return factor_scores[factor_name]
