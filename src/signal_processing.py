"""Cross-sectional signal processing utilities."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def winsorize_cross_section(
    factor_df: pd.DataFrame,
    lower: float = 0.05,
    upper: float = 0.95,
) -> pd.DataFrame:
    """Winsorize factor values cross-sectionally for each date."""
    if not 0 <= lower <= upper <= 1:
        raise ValueError("Quantile bounds must satisfy 0 <= lower <= upper <= 1.")

    lower_bounds = factor_df.quantile(lower, axis=1)
    upper_bounds = factor_df.quantile(upper, axis=1)

    return factor_df.clip(lower=lower_bounds, upper=upper_bounds, axis=0)


def zscore_cross_section(factor_df: pd.DataFrame) -> pd.DataFrame:
    """Compute row-wise cross-sectional z-scores by date."""
    means = factor_df.mean(axis=1)
    stds = factor_df.std(axis=1).replace(0, np.nan)

    return factor_df.sub(means, axis=0).div(stds, axis=0)


def combine_factors(
    factor_dict: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float] | None = None,
    standardize_composite: bool = True,
) -> pd.DataFrame:
    """Align and combine factor DataFrames into a composite score."""
    if not factor_dict:
        raise ValueError("At least one factor DataFrame is required.")

    factor_names = list(factor_dict)
    common_index, common_columns = _common_factor_axes(factor_dict)
    factor_weights = _resolve_factor_weights(factor_names, weights)

    composite = pd.DataFrame(0.0, index=common_index, columns=common_columns)
    for name in factor_names:
        aligned_factor = factor_dict[name].loc[common_index, common_columns]
        composite = composite + aligned_factor * factor_weights[name]

    if standardize_composite:
        return zscore_cross_section(composite)

    return composite


def _common_factor_axes(
    factor_dict: Mapping[str, pd.DataFrame],
) -> tuple[pd.Index, pd.Index]:
    factor_names = list(factor_dict)
    first_factor = factor_dict[factor_names[0]]
    common_index = first_factor.index
    common_columns = first_factor.columns

    for name in factor_names[1:]:
        factor = factor_dict[name]
        common_index = common_index.intersection(factor.index)
        common_columns = common_columns.intersection(factor.columns)

    ordered_index = first_factor.index[first_factor.index.isin(common_index)]
    ordered_columns = first_factor.columns[first_factor.columns.isin(common_columns)]

    return ordered_index, ordered_columns


def _resolve_factor_weights(
    factor_names: list[str],
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    if weights is None:
        equal_weight = 1.0 / len(factor_names)
        return {name: equal_weight for name in factor_names}

    missing_weights = [name for name in factor_names if name not in weights]
    if missing_weights:
        raise ValueError(f"Missing weights for factors: {missing_weights}")

    return {name: float(weights[name]) for name in factor_names}
