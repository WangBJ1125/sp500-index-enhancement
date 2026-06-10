# AGENTS.md

## Project Name

S&P 500 Index Enhancement Research Framework

## Project Objective

This project builds a Python-based S&P 500 index enhancement research framework.

The goal is not to forecast the broad market direction. The goal is to construct a long-only equity portfolio that stays close to an S&P 500 benchmark proxy while taking small, systematic active tilts based on transparent cross-sectional signals.

The central portfolio structure is:

```math
w_{i,t} = b_{i,t} + a_{i,t}
```

where:

- `w_{i,t}` is the strategy portfolio weight.
- `b_{i,t}` is the benchmark or proxy benchmark weight.
- `a_{i,t}` is the active weight generated from alpha scores.

The project is a research prototype, not a production trading strategy. It should demonstrate a complete quant research workflow:

1. data collection,
2. factor construction,
3. signal processing,
4. benchmark-aware portfolio construction,
5. backtesting with correct timing,
6. transaction costs and turnover,
7. performance reporting,
8. robustness checks,
9. factor diagnostics,
10. active-risk diagnostics.

The project currently has two major research stages:

- **Version 0**: Working baseline and research diagnostics.
- **Version 1**: Risk-controlled index enhancement framework.

---

## Current Project Status

The project is currently around:

```text
Version 0.4 / early Version 1 transition
```

Completed so far:

- End-to-end `python main.py` pipeline.
- Current S&P 500 universe loader.
- yfinance adjusted close and volume loader.
- Simple and log return utilities.
- Price-based factors:
  - momentum,
  - low volatility,
  - short-term reversal.
- Cross-sectional signal processing.
- Monthly backtester with one-day trade lag.
- Transaction cost handling.
- Performance metrics.
- Reporting and plots.
- Robustness experiments:
  - cost sensitivity,
  - active budget sensitivity,
  - factor variant sensitivity,
  - subperiod analysis.
- Benchmark hygiene:
  - comparison versus SPY,
  - comparison versus same-universe equal-weight benchmark.
- Static current market-cap benchmark proxy.
- Factor IC and Rank IC diagnostics.
- Momentum parameter IC sensitivity.
- Research notebooks for presentation and inspection.

Important conclusion so far:

> Static current market-cap benchmark weights make the framework conceptually closer to S&P 500 index enhancement, but they are not point-in-time and therefore introduce look-ahead bias. Strong performance under static current market-cap weights must not be interpreted as tradable historical evidence.

---

## Core Financial Concepts

Portfolio simple return:

```math
r_{p,t} = \sum_i w_{i,t-1} r_{i,t}
```

For factor construction, the project may use log returns because they are time-additive and convenient for momentum and volatility estimation.

For portfolio aggregation, NAV, and backtesting, the project must use simple returns because portfolio returns are linear in asset simple returns.

Benchmark return is usually proxied by SPY:

```math
r_{b,t} = r_{\text{SPY},t}
```

Active return:

```math
r^{active}_t = r_{p,t} - r_{b,t}
```

Active weights should satisfy approximately:

```math
\sum_i a_{i,t} = 0
```

Portfolio weights should satisfy:

```math
\sum_i w_{i,t} = 1
```

The portfolio is long-only unless explicitly stated otherwise:

```math
w_{i,t} \geq 0
```

---

## Important Research Constraints

### 1. Avoid Look-Ahead Bias

At rebalance date `t`, the strategy may only use information available up to `t`.

Signals computed using data up to close of date `t` should be traded from the next trading day onward.

Correct logic:

1. Use historical data up to date `t`.
2. Compute factors at date `t`.
3. Build target portfolio weights at date `t`.
4. Apply these weights to returns from `t+1` until the next rebalance date.

Incorrect logic:

- Using returns from `t+1` to choose weights at `t`.
- Using full-sample means and standard deviations for signal standardization.
- Using future constituents or future benchmark weights.
- Using current fundamentals for historical backtests.
- Treating current market-cap weights as point-in-time historical weights.

### 2. Survivorship Bias

The current implementation may use current S&P 500 constituents as a static universe.

This introduces survivorship bias because historical removed constituents are missing.

Reports and README must explicitly state:

> This prototype uses current S&P 500 constituents as a static universe. This introduces survivorship bias. Results should be interpreted as research-prototype evidence, not institutional-grade evidence.

### 3. Static Market-Cap Benchmark Bias

The project supports:

```yaml
portfolio:
  benchmark_weight_method: "market_cap_static"
```

This uses current market capitalizations from yfinance to approximate benchmark weights.

This is useful for making the framework conceptually closer to index enhancement, but it is not point-in-time.

Reports must explicitly state:

> Static current market-cap weights introduce look-ahead bias when applied to historical backtests. Strong results under this mode should be interpreted as framework demonstration, not tradable historical evidence.

### 4. Fundamental Data Limitation

Do not use current fundamental data such as current P/E, P/B, ROE, margins, or earnings growth for historical backtests unless point-in-time data is available.

Allowed first-stage signals:

- Price momentum.
- Low volatility.
- Short-term reversal.
- Liquidity / average dollar volume.
- Beta estimated from historical returns.
- Residual or risk-adjusted momentum estimated using historical returns.

Avoid non-point-in-time fundamentals.

### 5. Benchmark Interpretation

Different benchmarks answer different questions.

- `SPY` comparison answers: how did the strategy perform versus a tradable S&P 500 ETF proxy?
- Same-universe equal-weight comparison answers: did the factor tilt add value relative to the same stock universe?
- Static market-cap proxy comparison answers: how does the framework behave when starting from a market-cap-like benchmark proxy?

Do not mix these interpretations.

---

# Version 0: Completed Baseline and Diagnostics

## Version 0.1 — Working Baseline

Completed features:

- Static current S&P 500 universe from Wikipedia.
- yfinance adjusted close and volume data.
- Simple returns and log returns.
- Price-based factors:
  - momentum,
  - low volatility,
  - short-term reversal.
- Cross-sectional signal processing:
  - winsorization,
  - z-scoring,
  - composite score.
- Score-based active weights.
- Long-only portfolio construction.
- Monthly rebalancing.
- One trading day trade lag.
- Transaction costs.
- Basic performance metrics.

## Version 0.2 — Reporting and Robustness

Completed features:

- NAV plot versus benchmark.
- Cumulative active return plot.
- Drawdown plot.
- Turnover plot.
- Rolling active return.
- Rolling tracking error.
- Markdown summary report.
- Cost sensitivity.
- Active budget sensitivity.
- Factor variant sensitivity.
- Subperiod performance.

## Version 0.3 — Benchmark Hygiene

Completed features:

- Strategy versus SPY comparison.
- Strategy versus same-universe equal-weight benchmark comparison.
- Subperiod analysis by benchmark.
- Identification of benchmark mismatch.

Key finding:

> The equal-weight proxy version could look positive versus SPY while being weak versus the same-universe equal-weight benchmark. This indicated that apparent alpha could be caused by benchmark mismatch.

## Version 0.4 — Static Market-Cap Benchmark Proxy

Completed features:

- `src/benchmark.py`.
- Current market-cap retrieval via yfinance.
- Static market-cap benchmark weights.
- Backtester support for external benchmark weights.
- Configurable benchmark mode:
  - `equal_weight`,
  - `market_cap_static`.
- Active-weight cap via `max_active_weight`.
- Fixed bug where `max_weight=0.03` was unintentionally applied even when `max_weight: null`.

Important rule:

> In `market_cap_static` mode, avoid a hard absolute `max_weight` unless carefully justified. A 3% absolute cap can force mega-cap benchmark weights down and create unintended anti-mega-cap active bets.

Preferred market-cap static config:

```yaml
portfolio:
  benchmark_weight_method: "market_cap_static"
  active_budget: 0.20
  max_active_weight: 0.005
  max_weight: null
  max_monthly_turnover: null
```

## Version 0 Diagnostics

Completed diagnostics:

- Factor IC.
- Rank IC.
- IC subperiod analysis.
- Momentum parameter IC sensitivity.

Important findings:

- The original equal-weight composite factor does not show robust positive IC.
- Momentum is the most promising of the initial price-based signals, but its IC is modest.
- Low volatility has weak or negative IC and may be better treated as risk control rather than a direct alpha factor.
- Short-term reversal is unstable and tends to create high turnover.
- Momentum parameter IC sensitivity suggests `lookback_days=126` and `skip_days=0` may be better than the original 12-1 momentum in the tested sample.

---

# Version 1: Risk-Controlled Index Enhancement Framework

## Version 1 Objective

Version 1 should improve the project from a factor backtest prototype into a more realistic index-enhancement research framework.

Version 1 should focus on:

- active exposure diagnostics,
- better alpha model construction,
- risk-aware active tilts,
- turnover and cost control,
- sector exposure reporting,
- clearer separation between alpha and risk control,
- better interpretability.

Version 1 still does not require machine learning.

---

## Version 1.1 — Active Exposure Diagnostics

### Objective

Measure how far the portfolio is from the benchmark proxy.

For each rebalance or active date, compute:

```math
\text{Gross Active Exposure}_t = \sum_i |w_{i,t} - b_{i,t}|
```

```math
\text{Active Share}_t = \frac{1}{2}\sum_i |w_{i,t} - b_{i,t}|
```

```math
\text{Net Active Weight}_t = \sum_i (w_{i,t} - b_{i,t})
```

```math
\text{Max Absolute Active Weight}_t = \max_i |w_{i,t} - b_{i,t}|
```

Required outputs:

- `reports/active_exposure_diagnostics.csv`
- `reports/active_exposure_summary.csv`

Expected checks:

- `net_active_weight` should be close to zero.
- `max_absolute_active_weight` should respect `max_active_weight` when configured.
- `active_share` should remain in a range consistent with index enhancement.

### Implementation Guidance

Add to `src/diagnostics.py`:

- `compute_active_exposure_diagnostics(weights, benchmark_weights)`
- `summarize_active_exposure(active_exposure_df)`

Do not change the backtester or portfolio construction logic when adding diagnostics.

---

## Version 1.2 — Momentum Backtest Sensitivity

### Objective

IC sensitivity is useful, but IC does not guarantee portfolio performance.

Run momentum-only backtests for different momentum parameter combinations:

```text
lookback_days = [63, 126, 189, 252]
skip_days = [0, 21]
```

Required output:

- `reports/momentum_backtest_sensitivity.csv`

Columns should include:

- `lookback_days`
- `skip_days`
- `total_return`
- `annualized_return`
- `annualized_volatility`
- `sharpe_ratio`
- `max_drawdown`
- `annualized_active_return`
- `tracking_error`
- `information_ratio`
- `average_turnover`
- `annualized_turnover`

Purpose:

> Compare IC sensitivity with actual portfolio backtest sensitivity before changing the default momentum specification.

---

## Version 1.3 — Alpha Model Layer

### Objective

Separate raw factor calculation from alpha-score construction.

Create a dedicated alpha model layer, for example:

```text
src/alpha_model.py
```

The alpha model layer should build the final alpha score from processed factor scores.

Supported alpha models should include:

- `equal_weight_composite`
- `momentum_only`
- `risk_adjusted_momentum`
- `momentum_reversal`
- `custom_weighted`

### Rationale

Version 0 used:

```math
\alpha_{i,t}
=
\frac{1}{3} z^{mom}_{i,t}
+
\frac{1}{3} z^{lowvol}_{i,t}
+
\frac{1}{3} z^{rev}_{i,t}
```

But diagnostics showed that this equal-weight composite does not have robust positive IC.

Version 1 should allow:

#### Momentum-only alpha

```math
\alpha_{i,t} = z(MOM_{i,t})
```

#### Risk-adjusted momentum

```math
\alpha_{i,t} = z(MOM_{i,t}) + \lambda z(LOWVOL_{i,t})
```

where `LOWVOL` is used as a risk-control penalty or stabilizer, not as a main alpha factor.

#### Momentum + reversal

```math
\alpha_{i,t} = z(MOM_{i,t}) + \lambda z(REV_{i,t})
```

Because reversal has high turnover, `lambda` should generally be small.

Suggested config:

```yaml
alpha:
  model: "momentum_only"
  lowvol_penalty_weight: 0.25
  reversal_weight: 0.20
```

The default can preserve old behavior:

```yaml
alpha:
  model: "equal_weight_composite"
```

---

## Version 1.4 — Sector Exposure Reporting

### Objective

Diagnose whether active return comes from stock selection or sector bets.

Compute:

```math
\text{Sector Weight}_{g,t} = \sum_{i \in g} w_{i,t}
```

```math
\text{Active Sector Weight}_{g,t} =
\sum_{i \in g} (w_{i,t} - b_{i,t})
```

Suggested outputs:

- `reports/sector_exposure_latest.csv`
- `reports/active_sector_exposure_latest.csv`
- `reports/sector_exposure_timeseries.csv`

Important limitation:

> Current sector labels from yfinance are not point-in-time. Sector reporting is acceptable as a prototype diagnostic only if clearly documented.

Do not implement sector neutralization before sector exposure reporting is stable.

---

## Version 1.5 — Improved Factor Design

After active diagnostics and alpha model modularization, consider better factor designs.

Potential improvements:

### 1. Residual Momentum

Estimate historical market beta using SPY returns:

```math
r_{i,t} = \alpha_i + \beta_i r_{m,t} + \varepsilon_{i,t}
```

Then compute momentum from residual returns instead of raw returns.

Purpose:

> Reduce the extent to which momentum is simply market beta or mega-cap trend exposure.

### 2. Risk-Adjusted Momentum

Use momentum scaled or penalized by volatility:

```math
RAMOM_{i,t} =
\frac{MOM_{i,t}}{\sigma_{i,t}}
```

or:

```math
RAMOM_{i,t} = z(MOM_{i,t}) + \lambda z(LOWVOL_{i,t})
```

### 3. Beta-Controlled Features

Compute rolling beta to SPY and use it for diagnostics or active-risk control.

### 4. Liquidity / Volume Features

Possible prototype features:

- average dollar volume,
- change in average dollar volume,
- volume-adjusted momentum.

Do not add too many features before diagnostics are stable.

---

## Version 1.6 — Optional Machine Learning

Machine learning should only be considered after the baseline and diagnostics are stable.

ML should be used to generate alpha scores, not direct portfolio weights.

Potential feature matrix:

```text
X_{i,t} = [
  MOM_{3,0},
  MOM_{6,0},
  MOM_{9,0},
  MOM_{12,0},
  volatility,
  reversal,
  beta_to_spy,
  liquidity
]
```

Potential target:

```math
y_{i,t} = r_{i,t \to t+1}
```

or cross-sectional rank/relative return.

Recommended first models:

- Ridge regression.
- Elastic Net.
- Gradient boosting only after linear baselines are stable.

Required ML validation:

- walk-forward training,
- no future data,
- no full-sample scaling,
- comparison against simple factor baseline,
- IC and portfolio performance out-of-sample.

Avoid deep learning unless there is a clear reason and enough data.

---

## Current Factor Definitions

### Momentum

Implemented by:

```python
compute_momentum_12_1(prices, lookback_days, skip_days, use_log=True)
```

General formula:

```math
MOM_{i,t} =
\log\left(\frac{P_{i,t-\text{skip}}}{P_{i,t-\text{lookback}}}\right)
```

Original default:

```yaml
lookback_days: 252
skip_days: 21
```

Preferred tested candidate from IC sensitivity:

```yaml
lookback_days: 126
skip_days: 0
```

### Low Volatility

Low volatility score:

```math
LOWVOL_{i,t} =
-\sqrt{252}
\cdot
\operatorname{std}(\ell_{i,t-126}, \ldots, \ell_{i,t-1})
```

where `ell` is log return.

Current interpretation:

> Low-volatility should be treated carefully. IC evidence suggests it is not a strong standalone alpha factor in the current setup. It may be more suitable as a risk-control component.

### Short-Term Reversal

Short-term reversal:

```math
REV_{i,t}
=
-\log\left(\frac{P_{i,t}}{P_{i,t-21}}\right)
```

Current interpretation:

> Reversal is unstable and tends to increase turnover. It should not receive a large weight without further evidence.

### Composite

Original equal-weight composite:

```math
\alpha_{i,t}
=
\frac{1}{3}z^{mom}_{i,t}
+
\frac{1}{3}z^{lowvol}_{i,t}
+
\frac{1}{3}z^{rev}_{i,t}
```

Current interpretation:

> The original equal-weight composite is useful as a baseline but should not be treated as the preferred final alpha model.

---

## Signal Processing Requirements

For each rebalance date and each factor:

1. Drop stocks with insufficient data.
2. Winsorize cross-sectionally.
3. Z-score cross-sectionally.
4. Combine factor scores according to the selected alpha model.
5. Standardize final alpha score cross-sectionally if configured.

Cross-sectional standardization must be date-by-date.

Correct:

```python
df.mean(axis=1)
df.std(axis=1)
```

Incorrect for this project:

```python
df.mean(axis=0)
df.std(axis=0)
```

because that standardizes each stock through time rather than each date across stocks.

---

## Portfolio Construction Requirements

The portfolio should be benchmark-aware:

```math
w_t = b_t + a_t
```

Active weights should be score-proportional and net-zero before constraints:

```math
a_{i,t}
=
A
\cdot
\frac{\alpha_{i,t} - \bar{\alpha}_t}
{\sum_j |\alpha_{j,t} - \bar{\alpha}_t|}
```

Use:

```yaml
active_budget: 0.20
```

as a default active exposure budget.

Use:

```yaml
max_active_weight: 0.005
```

as a default single-name active weight cap for index enhancement.

In `market_cap_static` mode, prefer:

```yaml
max_weight: null
```

unless a carefully justified absolute cap is added.

Reason:

> An absolute cap such as 3% can force mega-cap names below their benchmark weights and create unintended active bets.

---

## Turnover and Transaction Costs

Turnover:

```math
\text{Turnover}_t
=
\frac{1}{2}
\sum_i |w^{new}_{i,t} - w^{old}_{i,t}|
```

Transaction cost:

```math
\text{Cost}_t =
2 \cdot c \cdot \text{Turnover}_t
```

where:

```python
c = transaction_cost_bps / 10000
```

Transaction costs should be deducted only on the first active holding day after a rebalance, not every day.

---

## Performance Metrics

The final report should include:

- total return,
- annualized return,
- annualized volatility,
- Sharpe ratio,
- max drawdown,
- hit ratio,
- annualized active return,
- tracking error,
- information ratio,
- average turnover,
- annualized turnover.

Do not report only total return and Sharpe.

Always include active-return and benchmark-relative metrics.

---

## Robustness and Diagnostics

The project should maintain and extend these diagnostics:

### Performance Robustness

- Cost sensitivity.
- Active budget sensitivity.
- Factor variant sensitivity.
- Momentum backtest sensitivity.
- Subperiod performance.
- Benchmark comparison.
- Subperiod by benchmark.

### Factor Diagnostics

- IC.
- Rank IC.
- ICIR.
- Hit rate.
- IC subperiod analysis.
- Momentum parameter IC sensitivity.

### Portfolio Diagnostics

- Active exposure diagnostics.
- Active share.
- Max absolute active weight.
- Top overweight / underweight names.
- Turnover by rebalance.
- Optional sector exposure diagnostics.

---

## Expected Outputs

The project should generate:

### Processed Data

Saved under:

```text
data/processed/
```

Typical files:

- `portfolio_returns_gross.csv`
- `portfolio_returns_net.csv`
- `benchmark_returns.csv`
- `weights.csv`
- `turnover.csv`
- `transaction_costs.csv`
- `performance_summary.csv`
- `static_market_cap_weights.csv`
- `current_market_caps.csv`
- `alpha_scores.csv` after Version 1.3

### Reports

Saved under:

```text
reports/
```

Typical files:

- `summary_report.md`
- `benchmark_comparison.csv`
- `subperiod_by_benchmark.csv`
- `cost_sensitivity.csv`
- `active_budget_sensitivity.csv`
- `factor_variant_sensitivity.csv`
- `factor_ic_summary.csv`
- `factor_ic_subperiod.csv`
- `momentum_ic_sensitivity.csv`
- `momentum_ic_sensitivity_subperiod.csv`
- `momentum_backtest_sensitivity.csv` after Version 1.2
- `active_exposure_diagnostics.csv` after Version 1.1
- `active_exposure_summary.csv` after Version 1.1

### Figures

Saved under:

```text
reports/figures/
```

Typical figures:

- strategy NAV vs benchmark NAV,
- cumulative active return,
- drawdown,
- turnover,
- rolling active return,
- rolling tracking error.

---

## Suggested Repository Structure

```text
sp500-index-enhancement/
│
├── AGENTS.md
├── PROJECT_REVIEW.md
├── README.md
├── requirements.txt
├── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_data_download.ipynb
│   ├── 02_factor_research.ipynb
│   ├── 03_backtest_baseline.ipynb
│   └── 04_performance_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── universe.py
│   ├── data_loader.py
│   ├── utils.py
│   ├── factors.py
│   ├── signal_processing.py
│   ├── portfolio.py
│   ├── backtester.py
│   ├── performance.py
│   ├── reporting.py
│   ├── benchmark.py
│   ├── experiments.py
│   ├── diagnostics.py
│   └── alpha_model.py       # Version 1.3
│
├── tests/
│   ├── test_universe.py
│   ├── test_data_loader.py
│   ├── test_utils.py
│   ├── test_factors.py
│   ├── test_signal_processing.py
│   ├── test_portfolio.py
│   ├── test_backtester.py
│   ├── test_performance.py
│   ├── test_reporting.py
│   ├── test_benchmark.py
│   ├── test_experiments.py
│   ├── test_diagnostics.py
│   └── test_alpha_model.py  # Version 1.3
│
├── reports/
│   ├── figures/
│   └── summary_report.md
│
└── main.py
```

---

## Configuration Guidance

Recommended current Version 1 transition config:

```yaml
data:
  start_date: "2015-01-01"
  end_date: null
  benchmark: "SPY"
  universe: "current_sp500"
  max_tickers: null
  refresh_market_caps: false

backtest:
  rebalance_frequency: "M"
  trade_lag_days: 1
  initial_capital: 1.0

factors:
  momentum:
    enabled: true
    lookback_days: 126
    skip_days: 0
    weight: 0.3333
  low_volatility:
    enabled: true
    lookback_days: 126
    weight: 0.3333
  reversal:
    enabled: true
    lookback_days: 21
    weight: 0.3333

alpha:
  model: "equal_weight_composite"
  lowvol_penalty_weight: 0.25
  reversal_weight: 0.20

signal_processing:
  winsorize_lower: 0.05
  winsorize_upper: 0.95
  zscore: true
  sector_neutralize: false

portfolio:
  construction_method: "score_active_weight"
  benchmark_weight_method: "market_cap_static"
  active_budget: 0.20
  max_active_weight: 0.005
  max_weight: null
  long_only: true
  max_monthly_turnover: null

costs:
  transaction_cost_bps: 10
```

---

## Coding Style Requirements

- Use Python 3.10+.
- Use `pandas`, `numpy`, `yfinance`, and `matplotlib`.
- Optional libraries such as `scipy`, `statsmodels`, and `scikit-learn` are allowed only when they add clear value.
- Keep functions small and testable.
- Use type hints where reasonable.
- Avoid hidden global state.
- Avoid hard-coded paths.
- Use `pathlib.Path`.
- Write clear docstrings explaining financial assumptions.
- Do not duplicate core logic in notebooks.
- Notebooks should call `src/` functions or read saved outputs.

---

## Testing Requirements

Maintain unit tests for:

- universe loading,
- data loading,
- return utilities,
- factor calculations,
- signal processing,
- portfolio construction,
- backtester timing,
- performance metrics,
- reporting,
- benchmark utilities,
- experiments,
- diagnostics,
- alpha model layer once added.

Tests should use synthetic data or mocks when possible.

Avoid live internet calls in tests.

---

## Things Not To Do

Do not:

- use future data,
- use current fundamentals for historical backtests,
- claim institutional-grade results with static constituents or static current market caps,
- optimize too many parameters without reporting all experiments,
- add machine learning before baseline diagnostics are stable,
- ignore transaction costs,
- ignore turnover,
- ignore benchmark-relative metrics,
- hide bad subperiods,
- overstate headline returns,
- treat static current market-cap backtests as tradable evidence,
- use hard absolute `max_weight=0.03` in market-cap benchmark mode unless explicitly justified.

---

## Development Priority From Here

Proceed in this order:

1. **Version 1.1**: Active exposure diagnostics.
2. **Version 1.2**: Momentum parameter backtest sensitivity.
3. **Version 1.3**: Alpha model layer.
4. **Version 1.4**: Sector exposure reporting.
5. **Version 1.5**: Improved factor design:
   - residual momentum,
   - risk-adjusted momentum,
   - beta-adjusted features,
   - liquidity features.
6. **Version 1.6**: Optional simple ML alpha model after diagnostics are stable.

---

## Definition of Done: Version 1

Version 1 is complete when:

- Active exposure diagnostics are implemented and saved.
- Active share is reported.
- Max absolute active weight is reported and checked against config.
- Momentum backtest sensitivity is implemented.
- The alpha model layer is implemented and configurable.
- Low-volatility is no longer blindly treated as equal alpha unless explicitly selected.
- Sector exposure reporting is available.
- Robustness tables remain compatible with the selected benchmark mode.
- Reports and notebooks explain:
  - benchmark choice,
  - active exposure,
  - turnover,
  - costs,
  - IC results,
  - limitations.
- The project clearly distinguishes:
  - factor alpha,
  - benchmark mismatch,
  - static market-cap look-ahead bias,
  - implementation effects.

---

## Final Research Attitude

This project should be treated as a transparent quant research prototype.

The goal is not to produce a perfect strategy.

The goal is to demonstrate:

- correct financial thinking,
- clean implementation,
- awareness of backtesting pitfalls,
- understanding of benchmark-relative performance,
- understanding of active risk,
- ability to diagnose false alpha,
- ability to explain results honestly.

A modest but robust result is better than an impressive but biased backtest.
