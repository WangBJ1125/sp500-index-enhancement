# AGENTS.md

## Project Name

S&P 500 Index Enhancement Research Framework

## Project Objective

This project builds a Python-based S&P 500 index enhancement research framework.

The goal is not to forecast the broad market direction. The goal is to construct a long-only equity portfolio that stays close to the S&P 500 benchmark while taking small, systematic active tilts based on transparent cross-sectional factors.

The main research question is:

> Can simple, price-based cross-sectional factors generate positive active returns versus SPY after realistic implementation assumptions such as monthly rebalancing, transaction costs, turnover analysis, and active weight constraints?

The project should be implemented in two stages:

- **Version 0**: Simple rule-based baseline.
- **Version 1**: More realistic index-enhancement framework with active-weight controls, transaction costs, turnover analysis, and optional sector neutralization.

The code should be modular, readable, testable, and suitable for a student/research project in quantitative asset management.

---

## Core Financial Concepts

The portfolio return is:

```math
r_{p,t} = \sum_i w_{i,t-1} r_{i,t}
```
For factor construction, the project primarily uses log returns because they are time-additive and convenient for momentum and volatility estimation. For portfolio aggregation and NAV calculation, the project uses simple returns because portfolio returns are linear in asset simple returns.

The benchmark return is usually proxied by SPY:

```math
r_{b,t} = r_{\text{SPY},t}
```

The active return is:

```math
r^{active}_t = r_{p,t} - r_{b,t}
```

The index-enhanced portfolio should be interpreted as:

```math
w_{i,t} = b_{i,t} + a_{i,t}
```

where:

- `w_{i,t}` is the strategy portfolio weight.
- `b_{i,t}` is the benchmark or proxy benchmark weight.
- `a_{i,t}` is the active weight.

The active weights should satisfy:

```math
\sum_i a_{i,t} = 0
```

The portfolio weights should satisfy:

```math
\sum_i w_{i,t} = 1
```

For this project, the portfolio should be long-only unless explicitly stated otherwise:

```math
w_{i,t} \geq 0
```

---

## Important Research Constraints

### 1. Avoid Look-Ahead Bias

At rebalance date `t`, the strategy may only use information available up to `t`.

Signals computed using data up to close of date `t` should be traded from the next trading day onward.

Do not use future returns, future constituents, future benchmark weights, future fundamentals, or any data unavailable at the decision time.

Correct logic:

1. Use historical data up to date `t`.
2. Compute factors at date `t`.
3. Build target portfolio weights at date `t`.
4. Apply these weights to returns from `t+1` until the next rebalance date.

Incorrect logic:

- Using returns from `t+1` to `t+21` to choose weights at `t`.
- Using full-sample mean and standard deviation for signal standardization.
- Using future fundamental data in past backtests.
- Using today's S&P 500 constituents without explicitly labeling survivorship bias.

---

### 2. Survivorship Bias

The first implementation may use the current S&P 500 constituents as a static universe because this is easy to obtain from public sources.

However, this introduces survivorship bias.

The README and reports must explicitly state:

> This prototype uses the current S&P 500 constituents as a static universe. This introduces survivorship bias because stocks that were previously in the index but later removed are not included. Therefore, results should be interpreted as a research prototype rather than production-level evidence.

Do not claim that the results are fully institutional-grade unless historical point-in-time constituents are added.

---

### 3. Fundamental Data Limitation

Do not use current fundamental data, such as current P/E, P/B, ROE, margins, or earnings growth, to backtest historical strategies.

Unless point-in-time fundamental data is available, only use price-based and volume-based signals.

Allowed first-stage signals:

- Momentum
- Low volatility
- Short-term reversal
- Liquidity / average dollar volume
- Beta, if estimated only with historical returns

Avoid using non-point-in-time fundamentals.

---

# Version 0: Simple Rule-Based Baseline

## Objective

Build a simple, transparent, fully functioning index enhancement backtest.

Version 0 should prioritize:

- Correct time alignment.
- Clean factor construction.
- Clear portfolio construction.
- Transaction cost sensitivity.
- Basic performance analytics.

Version 0 does not need optimization, machine learning, or a full risk model.

---

## Version 0 Data

Use:

- Current S&P 500 constituents.
- Daily adjusted close prices from `yfinance`.
- Daily volume from `yfinance`.
- SPY adjusted close as benchmark.

Suggested time range:

- Start: `2015-01-01`
- End: latest available date, or configurable.

Use adjusted close prices for return calculations.

---

## Version 0 Rebalancing

Use monthly rebalancing.

Default rebalance dates:

- Last available trading day of each month.

At each rebalance date:

1. Compute factor values using historical prices up to the rebalance date.
2. Process signals cross-sectionally.
3. Construct portfolio weights.
4. Hold weights until the next rebalance period.
5. Deduct transaction costs at rebalance.

---

## Version 0 Factors

Implement at least three price-based factors.

### 1. 12-1 Momentum

Use cumulative return from approximately 12 months ago to 1 month ago.

```math
\text{MOM}_{i,t} = \prod_{k=22}^{252} (1 + r_{i,t-k}) - 1
```

This skips the most recent 21 trading days to reduce short-term reversal noise.

Higher momentum should imply higher score.

---

### 2. 6-Month Low Volatility

Estimate annualized volatility using the previous 126 trading days.

```math
\text{VOL}_{i,t} = \sqrt{252} \cdot \operatorname{std}(r_{i,t-126}, ..., r_{i,t-1})
```

Low volatility score should be:

```math
\text{LOWVOL}_{i,t} = - \text{VOL}_{i,t}
```

Lower volatility should imply higher score.

---

### 3. 1-Month Short-Term Reversal

Use negative recent 1-month return.

```math
\text{REV}_{i,t} = - \left( \frac{P_{i,t}}{P_{i,t-21}} - 1 \right)
```

Stocks that performed poorly in the most recent month receive a higher reversal score.

---

## Version 0 Signal Processing

For each rebalance date and for each factor:

1. Select the current universe.
2. Drop stocks with insufficient data.
3. Winsorize cross-sectionally.
4. Z-score cross-sectionally.

Winsorization:

```math
x^{win}_{i,t} = \min(\max(x_{i,t}, q_{5\%,t}), q_{95\%,t})
```

Z-score:

```math
z_{i,t} = \frac{x^{win}_{i,t} - \mu_t}{\sigma_t}
```

Composite alpha score:

```math
\alpha_{i,t} = \frac{1}{3} z^{mom}_{i,t}
              + \frac{1}{3} z^{lowvol}_{i,t}
              + \frac{1}{3} z^{rev}_{i,t}
```

Then standardize the composite score again cross-sectionally.

---

## Version 0 Portfolio Construction

Recommended default construction:

- Use equal-weight benchmark proxy within the available universe:

```math
b_{i,t} = \frac{1}{N_t}
```

- Convert composite scores into score-proportional active weights:

```math
a_{i,t} = A \cdot \frac{\alpha_{i,t}}{\sum_j |\alpha_{j,t}|}
```

where `A` is the active budget, for example `A = 0.20`.

Then:

```math
w_{i,t} = b_{i,t} + a_{i,t}
```

After that:

1. Clip negative weights to zero.
2. Normalize weights to sum to 1.
3. Store target weights.

Alternative construction:

- Rank stocks by composite alpha score.
- Top 20% receive overweight.
- Bottom 20% receive underweight.
- Middle 60% remain close to benchmark.

This top-bottom rule is simpler to explain but less smooth than score-proportional active weights.

---

## Version 0 Transaction Costs

At every rebalance date, compute turnover:

```math
\text{Turnover}_t = \frac{1}{2} \sum_i |w^{new}_{i,t} - w^{old}_{i,t}|
```

Cost model:

```math
\text{Cost}_t = 2 \cdot c \cdot \text{Turnover}_t
```

where `c` is one-way transaction cost.

Run at least three assumptions:

- `0 bps`
- `5 bps`
- `10 bps`

For example:

```python
cost_bps = 10
c = cost_bps / 10000
```

Net portfolio return:

```math
r^{net}_{p,t} = r^{gross}_{p,t} - \text{Cost}_t
```

The transaction cost should be deducted on the rebalance date or the first holding day after rebalance. Be consistent and document the convention.

---

## Version 0 Performance Metrics

Implement the following metrics:

### Total Return

```math
R_{total} = \prod_t (1 + r_t) - 1
```

### Annualized Return

```math
R_{ann} = \left(\prod_t (1 + r_t)\right)^{252/T} - 1
```

### Annualized Volatility

```math
\sigma_{ann} = \sqrt{252} \cdot \operatorname{std}(r_t)
```

### Sharpe Ratio

Assume zero risk-free rate in Version 0:

```math
\text{Sharpe} = \frac{R_{ann}}{\sigma_{ann}}
```

### Active Return

```math
r^{active}_t = r_{p,t} - r_{b,t}
```

### Annualized Active Return

Recommended simple convention:

```math
R^{active}_{ann} = 252 \cdot \operatorname{mean}(r^{active}_t)
```

### Tracking Error

```math
\text{TE} = \sqrt{252} \cdot \operatorname{std}(r^{active}_t)
```

### Information Ratio

```math
\text{IR} = \frac{R^{active}_{ann}}{\text{TE}}
```

### Maximum Drawdown

NAV:

```math
V_t = \prod_{k=1}^{t} (1+r_k)
```

Running maximum:

```math
M_t = \max_{s \leq t} V_s
```

Drawdown:

```math
DD_t = \frac{V_t}{M_t} - 1
```

Maximum drawdown:

```math
\text{MDD} = \min_t DD_t
```

### Turnover

Report:

- Average rebalance turnover.
- Annualized turnover.

For monthly rebalancing:

```math
\text{Annualized Turnover} = 12 \times \text{Average Monthly Turnover}
```

### Hit Ratio

Monthly active hit ratio:

```math
\text{Hit Ratio} = \frac{\#\{m: r^{active}_m > 0\}}{\#\{m\}}
```

---

## Version 0 Factor Diagnostics

Implement basic factor diagnostics.

For each factor and for the composite score, calculate monthly cross-sectional IC.

At rebalance date `t`, compute the cross-sectional correlation between factor score at `t` and future next-period return.

```math
\text{IC}_t = \operatorname{corr}(x_{i,t}, r_{i,t+1:t+h})
```

Also compute Rank IC using Spearman correlation.

Report:

- Mean IC.
- Standard deviation of IC.
- ICIR:

```math
\text{ICIR} = \frac{\operatorname{mean}(\text{IC})}{\operatorname{std}(\text{IC})}
```

---

# Version 1: More Realistic Index Enhancement

## Objective

Version 1 should improve Version 0 by making the portfolio construction more realistic.

Add:

- Explicit active weights.
- Active budget control.
- Single-name active weight caps.
- Optional sector neutralization.
- Turnover constraints or turnover penalty.
- Better reporting of implementation costs.
- More robustness checks.

Version 1 still does not require machine learning.

---

## Version 1 Benchmark Weights

If true S&P 500 historical benchmark weights are not available, use one of the following approximations:

1. Equal-weight benchmark within available universe.
2. Market-cap proxy weights if historical market cap data can be reliably obtained.
3. SPY only for return benchmark, while portfolio starts from equal-weight proxy.

If using equal-weight proxy, clearly state this limitation.

Do not pretend equal-weight proxy is the true S&P 500 benchmark.

---

## Version 1 Active Weight Construction

Use:

```math
w_t = b_t + a_t
```

where:

```math
\sum_i a_{i,t} = 0
```

Suggested active score construction:

```math
a_{i,t}^{raw} = \alpha_{i,t}
```

Normalize active weights:

```math
a_{i,t} = A \cdot \frac{\alpha_{i,t}}{\sum_j |\alpha_{j,t}|}
```

where `A` is the active budget.

Recommended defaults:

```yaml
active_budget: 0.20
max_active_weight: 0.005
max_weight: 0.03
long_only: true
```

Interpretation:

- `active_budget = 0.20` means total absolute active weight is controlled.
- `max_active_weight = 0.005` means no single stock can be overweighted or underweighted by more than 50 bps relative to benchmark.
- `max_weight = 0.03` means no single stock can exceed 3% portfolio weight.

After constraints:

1. Enforce active weight caps.
2. Add active weights to benchmark weights.
3. Enforce long-only.
4. Enforce max individual weight.
5. Renormalize.
6. Recalculate realized active weights.

---

## Version 1 Turnover Constraint

Compute turnover:

```math
\text{Turnover}_t = \frac{1}{2} \sum_i |w^{target}_{i,t} - w^{current}_{i,t}|
```

Optional hard constraint:

```math
\text{Turnover}_t \leq T_{\max}
```

Suggested default:

```yaml
max_monthly_turnover: 0.20
```

If raw target turnover exceeds the maximum, scale the trade vector:

```math
\Delta w_t = w^{target}_t - w^{current}_t
```

Scale:

```math
\Delta w^{scaled}_t =
\Delta w_t \cdot \frac{T_{\max}}{\text{Turnover}_t}
```

Then:

```math
w^{new}_t = w^{current}_t + \Delta w^{scaled}_t
```

Finally normalize if necessary.

---

## Version 1 Sector Neutralization

If sector data is available, implement optional sector neutralization.

For each rebalance date, run a cross-sectional regression:

```math
z_{i,t} = \alpha_t + \sum_g \beta_{g,t} D_{i,g,t} + \varepsilon_{i,t}
```

where `D_{i,g,t}` is a sector dummy.

Use residuals as sector-neutral scores:

```math
z^{neutral}_{i,t} = \varepsilon_{i,t}
```

If sector data is not available, skip sector neutralization and document it.

Do not use current sector classification if historical sector classification is required for a fully point-in-time backtest. For this project prototype, using current sector labels is acceptable only if clearly stated as an approximation.

---

## Version 1 Risk Controls

At minimum, report:

- Number of holdings.
- Top 10 weights.
- Top 10 active weights.
- Sector weights, if sector data exists.
- Active sector weights, if sector data exists.
- Turnover per rebalance.
- Cost per rebalance.
- Active return by month.

Optional but recommended:

- Rolling 12-month active return.
- Rolling 12-month tracking error.
- Rolling 12-month information ratio.

---

## Version 1 Robustness Checks

Run strategy variants:

### Factor Variants

- Momentum only.
- Low-volatility only.
- Reversal only.
- Momentum + low-volatility.
- Momentum + low-volatility + reversal.

### Cost Variants

- 0 bps.
- 5 bps.
- 10 bps.
- 20 bps.

### Rebalance Frequency

- Monthly.
- Quarterly.

### Time Subperiods

At least:

- 2015-2019
- 2020-2022
- 2023-latest

### Active Budget Sensitivity

Test:

- `active_budget = 0.10`
- `active_budget = 0.20`
- `active_budget = 0.30`

---

## Suggested Repository Structure

Use the following structure:

```text
sp500-index-enhancement/
│
├── AGENTS.md
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
│   ├── data_loader.py
│   ├── universe.py
│   ├── factors.py
│   ├── signal_processing.py
│   ├── portfolio.py
│   ├── backtester.py
│   ├── performance.py
│   ├── diagnostics.py
│   └── utils.py
│
├── tests/
│   ├── test_factors.py
│   ├── test_signal_processing.py
│   ├── test_portfolio.py
│   ├── test_backtester.py
│   └── test_performance.py
│
├── reports/
│   ├── figures/
│   └── summary_report.md
│
└── main.py
```

---

## Module Responsibilities

### `src/data_loader.py`

Responsible for:

- Downloading price data from `yfinance`.
- Downloading SPY benchmark data.
- Loading saved local data.
- Saving raw and processed data.
- Handling missing values.
- Aligning trading calendars.

Do not silently forward-fill prices across long missing periods.

---

### `src/universe.py`

Responsible for:

- Loading current S&P 500 tickers.
- Cleaning ticker symbols for `yfinance`.
- Creating static universe for Version 0.
- Later extension: point-in-time universe.

Must clearly expose whether the universe is static or point-in-time.

---

### `src/factors.py`

Responsible for computing raw factor values.

Functions should include:

- `compute_momentum_12_1(prices)`
- `compute_low_volatility(returns, window=126)`
- `compute_short_term_reversal(prices, window=21)`
- Optional: `compute_average_dollar_volume(prices, volumes, window=21)`
- Optional: `compute_beta_to_spy(returns, spy_returns, window=252)`

All factor functions must use only historical rolling windows.

---

### `src/signal_processing.py`

Responsible for:

- Winsorization.
- Z-scoring.
- Composite score construction.
- Optional sector neutralization.
- Handling missing factor values.

Functions should include:

- `winsorize_cross_section(factor_df, lower=0.05, upper=0.95)`
- `zscore_cross_section(factor_df)`
- `combine_factors(factor_dict, weights=None)`
- `neutralize_by_sector(scores, sector_map)`

Cross-sectional processing must be done date by date.

Do not calculate mean or standard deviation using the full time series across all dates.

---

### `src/portfolio.py`

Responsible for converting scores to portfolio weights.

Functions should include:

- `equal_weight_benchmark(universe)`
- `score_to_active_weights(scores, active_budget)`
- `apply_active_weight_caps(active_weights, max_active_weight)`
- `construct_long_only_portfolio(benchmark_weights, active_weights, max_weight=None)`
- `calculate_turnover(old_weights, new_weights)`
- `apply_turnover_limit(current_weights, target_weights, max_turnover)`

Weights must sum to 1 after construction.

Long-only portfolios must not contain negative weights.

---

### `src/backtester.py`

Responsible for the event-driven or period-based backtest loop.

At each rebalance date:

1. Get universe.
2. Get historical data available up to that date.
3. Compute factor scores.
4. Construct target weights.
5. Compute turnover.
6. Deduct transaction cost.
7. Apply weights to next period returns.
8. Store results.

The backtester must ensure proper timing.

No future data should be used.

---

### `src/performance.py`

Responsible for performance metrics.

Functions should include:

- `annualized_return(returns, periods_per_year=252)`
- `annualized_volatility(returns, periods_per_year=252)`
- `sharpe_ratio(returns, risk_free_rate=0.0)`
- `active_returns(portfolio_returns, benchmark_returns)`
- `tracking_error(active_returns, periods_per_year=252)`
- `information_ratio(active_returns, periods_per_year=252)`
- `max_drawdown(returns)`
- `hit_ratio(returns)`
- `turnover_stats(turnover_series)`

---

### `src/diagnostics.py`

Responsible for factor and portfolio diagnostics.

Functions should include:

- `calculate_ic(scores, forward_returns)`
- `calculate_rank_ic(scores, forward_returns)`
- `factor_ic_summary(ic_series)`
- `rolling_performance_metrics(...)`
- `sector_exposure(weights, sector_map)`
- `active_sector_exposure(portfolio_weights, benchmark_weights, sector_map)`

---

### `src/utils.py`

Responsible for:

- Date handling.
- Rebalance date generation.
- Common validation helpers.
- Logging.

---

## Configuration

Use `config.yaml` to store project parameters.

Suggested default config:

```yaml
data:
  start_date: "2015-01-01"
  end_date: null
  benchmark: "SPY"
  universe: "current_sp500"
  price_field: "Adj Close"

backtest:
  rebalance_frequency: "M"
  trade_lag_days: 1
  initial_capital: 1.0

factors:
  momentum:
    enabled: true
    lookback_days: 252
    skip_days: 21
    weight: 0.3333
  low_volatility:
    enabled: true
    lookback_days: 126
    weight: 0.3333
  reversal:
    enabled: true
    lookback_days: 21
    weight: 0.3333

signal_processing:
  winsorize_lower: 0.05
  winsorize_upper: 0.95
  zscore: true
  sector_neutralize: false

portfolio:
  construction_method: "score_active_weight"
  benchmark_weight_method: "equal_weight"
  active_budget: 0.20
  max_active_weight: 0.005
  max_weight: 0.03
  long_only: true
  max_monthly_turnover: null

costs:
  transaction_cost_bps: 10
```

---

## Coding Style Requirements

- Use Python 3.10+.
- Use `pandas`, `numpy`, `yfinance`, `matplotlib`.
- Optional: `scipy`, `statsmodels`, `scikit-learn`, but avoid unnecessary complexity in Version 0.
- Keep functions small and testable.
- Use type hints where reasonable.
- Avoid hidden global state.
- Avoid hard-coded paths.
- Use `pathlib.Path` for paths.
- Save intermediate data to `data/processed/` when useful.
- Write clear docstrings explaining financial assumptions.

---

## Testing Requirements

Create basic unit tests for:

### Factor Tests

- Momentum uses the correct lookback and skip window.
- Low-volatility factor uses only past returns.
- Reversal uses only the most recent past window.
- Factor output has expected shape.

### Signal Processing Tests

- Winsorization caps extreme values.
- Z-score produces approximately zero mean and unit standard deviation cross-sectionally.
- Composite score handles missing values.

### Portfolio Tests

- Weights sum to 1.
- Long-only constraint is respected.
- Active budget is approximately respected.
- Turnover calculation is correct.
- Max weight cap is respected.

### Backtester Tests

- No future returns are used for signal generation.
- Rebalance dates are generated correctly.
- Transaction costs reduce returns.
- Portfolio returns are aligned correctly with future holding-period returns.

### Performance Tests

- Annualized return calculation is correct for simple synthetic data.
- Max drawdown calculation is correct.
- Tracking error and IR are computed correctly.

---

## Expected Outputs

The project should generate:

### Tables

- Performance summary table.
- Cost sensitivity table.
- Factor IC summary table.
- Turnover summary table.
- Subperiod performance table.

### Figures

- Strategy NAV vs SPY NAV.
- Active return cumulative curve.
- Drawdown curve.
- Rolling 12-month active return.
- Rolling tracking error.
- Monthly turnover.
- Factor IC time series.

Save figures to:

```text
reports/figures/
```

Save summary report to:

```text
reports/summary_report.md
```

---

## Performance Summary Table

The final report should contain at least:

| Metric | Strategy Gross | Strategy Net | SPY | Active Net |
|---|---:|---:|---:|---:|
| Annualized Return | | | | |
| Annualized Volatility | | | | |
| Sharpe Ratio | | | | |
| Tracking Error | | | | |
| Information Ratio | | | | |
| Max Drawdown | | | | |
| Hit Ratio | | | | |
| Annualized Turnover | | | | |

---

## Cost Sensitivity Table

Run the same strategy under multiple transaction cost assumptions:

| Cost Assumption | Annual Active Return | Tracking Error | Information Ratio | Max Drawdown | Annual Turnover |
|---:|---:|---:|---:|---:|---:|
| 0 bps | | | | | |
| 5 bps | | | | | |
| 10 bps | | | | | |
| 20 bps | | | | | |

---

## Robustness Analysis

The report should discuss whether the strategy is robust across:

- Different transaction costs.
- Different subperiods.
- Different factor combinations.
- Different active budgets.
- Different rebalance frequencies.

Do not overstate results if performance is concentrated in one period or one factor.

---

## Interpretation Guidelines

When interpreting results, focus on:

1. Does the strategy generate positive active return after costs?
2. Is the information ratio positive and stable?
3. Is tracking error controlled?
4. Is turnover reasonable?
5. Are results robust across subperiods?
6. Which factor contributes most to performance?
7. Does the strategy suffer in specific market regimes?
8. Are results likely inflated by survivorship bias?

---

## Things Not To Do

Do not:

- Use future data.
- Use current fundamental data for historical backtests.
- Claim institutional-grade results with static current constituents.
- Optimize too many parameters without reporting all experiments.
- Add machine learning before the baseline is complete.
- Ignore transaction costs.
- Ignore turnover.
- Ignore benchmark-relative metrics.
- Only report total return and Sharpe.
- Hide bad subperiods.
- Overfit factor weights to maximize backtest performance.

---

## Development Priority

Build in this order:

1. Data download and cleaning.
2. Static S&P 500 universe.
3. Return calculation.
4. Factor calculation.
5. Signal processing.
6. Simple portfolio construction.
7. Monthly backtest loop.
8. Transaction cost handling.
9. Performance metrics.
10. Factor IC diagnostics.
11. Plots and reports.
12. Version 1 constraints and robustness checks.
13. Optional sector neutralization.
14. Optional risk model.
15. Optional machine learning only after the above is stable.

---

## Definition of Done: Version 0

Version 0 is complete when:

- The project can download data from `yfinance`.
- It can build a static current S&P 500 universe.
- It computes 12-1 momentum, low volatility, and short-term reversal.
- It combines signals into a composite score.
- It creates a long-only monthly rebalanced portfolio.
- It deducts transaction costs.
- It compares performance against SPY.
- It reports annual return, volatility, Sharpe, active return, tracking error, IR, max drawdown, turnover, and hit ratio.
- It saves at least three plots:
  - NAV vs SPY
  - cumulative active return
  - drawdown
- It clearly states survivorship bias and other limitations.

---

## Definition of Done: Version 1

Version 1 is complete when:

- Active weights are explicitly modeled.
- Active budget is controlled.
- Single-name active weight caps are implemented.
- Max stock weight is implemented.
- Turnover calculation is reliable.
- Optional turnover limit is implemented.
- Cost sensitivity analysis is included.
- Robustness checks are included.
- Factor IC and Rank IC are reported.
- Optional sector neutralization is implemented if sector data is available.
- The report explains both strengths and limitations of the strategy.

---

## Final Research Attitude

This project should be treated as a transparent quant research prototype.

The goal is not to produce a perfect strategy. The goal is to demonstrate:

- Correct financial thinking.
- Clean implementation.
- Awareness of backtesting pitfalls.
- Understanding of active risk.
- Ability to evaluate a strategy beyond headline returns.
- Ability to explain results honestly.

A modest but robust result is better than an impressive but biased backtest.
