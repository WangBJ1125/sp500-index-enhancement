# Project Review: S&P 500 Index Enhancement Framework

## 1. Project Objective

The goal of this project is to build a Python-based S&P 500 index enhancement research framework. The framework is designed to test whether transparent, price-based cross-sectional factors can create modest active returns while keeping the portfolio close to an S&P 500 benchmark proxy.

The intended portfolio structure is:

```text
w_i,t = b_i,t + a_i,t
```

where:

- `b_i,t` is the benchmark proxy weight.
- `a_i,t` is the active factor tilt.
- `w_i,t` is the final portfolio weight.

The project is a research prototype, not a production trading strategy. It is meant to demonstrate a complete quant research workflow: data loading, factor construction, signal processing, portfolio construction, backtesting, transaction costs, benchmark diagnostics, reporting, and factor IC analysis.

## 2. Current Repository Architecture

### `AGENTS.md`

- Defines the project objective, financial assumptions, research constraints, and development roadmap.
- States the key backtesting pitfalls: look-ahead bias, survivorship bias, non-point-in-time data, and benchmark mismatch.
- Provides module responsibilities and the intended Version 0 and Version 1 development priorities.

### `config.yaml`

- Stores run configuration for data range, benchmark ticker, universe, factor settings, portfolio constraints, and transaction costs.
- Currently supports `portfolio.benchmark_weight_method` with `equal_weight` and `market_cap_static`.
- Stores configurable momentum parameters through `factors.momentum.lookback_days` and `factors.momentum.skip_days`.
- Uses `portfolio.max_weight: null` in the current setup, which avoids forcing a 3% single-name cap in static market-cap benchmark mode.

### `main.py`

- Runs the end-to-end research pipeline.
- Loads config, downloads data, computes returns, computes factors, combines signals, runs the backtest, saves outputs, and writes reports.
- Supports equal-weight and static current market-cap benchmark proxy modes.
- Runs robustness and diagnostics tables including cost sensitivity, active budget sensitivity, factor variant sensitivity, subperiod analysis, benchmark comparison, factor IC diagnostics, and momentum IC sensitivity.

### `src/universe.py`

- Loads the current S&P 500 universe from Wikipedia.
- Cleans ticker symbols for `yfinance`, including replacing `.` with `-`.
- Returns sorted unique tickers.
- Documents that using current constituents as a static historical universe introduces survivorship bias.

### `src/data_loader.py`

- Downloads adjusted close prices and volume data using `yfinance`.
- Handles yfinance MultiIndex column structures and normalizes output columns to ticker symbols.
- Returns separate adjusted close and volume DataFrames.
- Provides save/load helpers for DataFrames.

### `src/utils.py`

- Computes simple returns for portfolio aggregation.
- Computes log returns for factor research.
- Generates month-end rebalance dates from the available trading index.
- Aligns Series/DataFrames on common DatetimeIndex values.

### `src/factors.py`

- Computes raw price-based factor values.
- Implements momentum with configurable lookback and skip windows.
- Implements low-volatility score as negative annualized rolling volatility of returns.
- Implements short-term reversal as negative recent return.
- Uses historical shifts and rolling windows so factors do not use future data.

### `src/signal_processing.py`

- Winsorizes factor values cross-sectionally by date.
- Z-scores factor values cross-sectionally by date.
- Combines factor DataFrames into an equal-weight or weighted composite.
- Aligns factor inputs on common dates and tickers before combining.

### `src/portfolio.py`

- Builds equal-weight benchmark proxy weights.
- Converts cross-sectional scores into active weights with zero net active exposure.
- Applies optional active weight caps.
- Constructs long-only portfolios from benchmark weights plus active weights.
- Computes turnover and applies optional turnover limits.

### `src/backtester.py`

- Implements the monthly backtest loop.
- Applies weights only after the configured trade lag, avoiding same-day signal use.
- Aggregates daily portfolio returns using simple stock returns.
- Deducts transaction costs only on the first active holding day after rebalance.
- Supports external benchmark weights for static market-cap benchmark proxy mode.

### `src/performance.py`

- Computes total return, annualized return, volatility, Sharpe ratio, active returns, tracking error, information ratio, max drawdown, and hit ratio.
- Aligns portfolio and benchmark returns for active-return calculations.
- Produces performance summary output used by reports and experiments.
- Uses simple return compounding for NAV and total return calculations.

### `src/reporting.py`

- Computes NAV and drawdown series.
- Creates matplotlib plots for NAV, cumulative active return, drawdown, turnover, rolling active return, and rolling tracking error.
- Writes `reports/summary_report.md`.
- Documents survivorship bias, benchmark limitations, transaction cost assumptions, and market-cap proxy warnings.

### `src/benchmark.py`

- Loads current market capitalizations from `yfinance`.
- Prefers `fast_info["market_cap"]` and falls back to `info["marketCap"]`.
- Converts market caps into static market-cap benchmark weights.
- Saves and loads market-cap data from CSV.
- Explicitly documents that current market caps are not point-in-time and create look-ahead bias in historical backtests.

### `src/experiments.py`

- Runs robustness experiments such as cost sensitivity, active budget sensitivity, factor variant sensitivity, and subperiod analysis.
- Computes same-universe equal-weight benchmark returns.
- Compares the strategy against SPY and equal-weight universe benchmark returns.
- Runs benchmark-specific subperiod analysis.
- Runs momentum parameter IC sensitivity across lookback and skip-day combinations.

### `src/diagnostics.py`

- Computes forward returns from one rebalance date to the next for diagnostics only.
- Calculates Pearson IC and Spearman Rank IC cross-sectionally by date.
- Summarizes IC series with mean IC, standard deviation, ICIR, hit rate, and count.
- Runs factor IC diagnostics for individual factors and composite scores.
- Runs factor IC subperiod analysis.

### `tests/`

- Contains unit tests for universe loading, data loading, utilities, factors, signal processing, portfolio construction, backtester timing, performance metrics, reporting, benchmark utilities, experiments, and diagnostics.
- Tests use synthetic data and mocks where appropriate.
- Live internet calls are avoided in tests.
- The test suite is designed to protect core behavior such as no same-day weight application, turnover calculation, cross-sectional processing, and IC calculation.

## 3. Completed Development Milestones

### Version 0.1 — Working Baseline

- Static current S&P 500 universe from Wikipedia.
- `yfinance` adjusted close and volume data.
- Simple returns and log returns.
- Price-based factors:
  - Momentum.
  - Low-volatility.
  - Short-term reversal.
- Cross-sectional signal processing.
- Active-weight portfolio construction.
- Monthly backtester.
- Transaction costs.
- Performance metrics.

### Version 0.2 — Reporting and Robustness

- NAV, drawdown, active return, turnover plots.
- Markdown summary report.
- Cost sensitivity.
- Active budget sensitivity.
- Factor variant sensitivity.
- Subperiod analysis.

### Version 0.3 — Benchmark Hygiene

- Compared strategy versus SPY.
- Compared strategy versus same-universe equal-weight benchmark.
- Found that equal-weight proxy results were affected by benchmark mismatch.
- Improved interpretation of active return by separating SPY-relative results from same-universe benchmark-relative results.

### Version 0.4 — Static Market-Cap Benchmark Proxy

- Added current market-cap loading via `yfinance`.
- Added static market-cap benchmark weights.
- Backtester supports external benchmark weights.
- `main.py` supports `portfolio.benchmark_weight_method = "market_cap_static"`.
- Portfolio construction is now closer to:

```text
w_i,t = b_i^static_mcap + a_i,t
```

- Added active-weight caps using `max_active_weight`.
- Fixed the bug where `max_weight=0.03` was unintentionally applied even when config had `max_weight: null`.

### Diagnostics and Research Tools

- Factor IC and Rank IC diagnostics.
- IC subperiod analysis.
- Momentum parameter IC sensitivity.

## 4. Important Implementation Decisions

1. Log returns are used for factor research because they are time-additive and convenient for momentum and volatility calculations.
2. Simple returns are used for portfolio aggregation because portfolio returns are linear in asset simple returns.
3. Monthly rebalancing is used as the Version 0 default to keep turnover and implementation assumptions simple.
4. A one trading day trade lag is used so signals formed at rebalance date `t` are applied from the next trading day onward.
5. Transaction costs are deducted only on rebalance execution dates, not every holding day.
6. Signal standardization uses cross-sectional z-scoring by date, not time-series z-scoring across the full sample.
7. The portfolio uses active weights around a benchmark proxy rather than direct top-N stock picking.
8. `max_active_weight` is used to make the portfolio behave more like an index enhancement portfolio with small active tilts.
9. `max_weight` should be avoided in market-cap benchmark mode unless carefully justified, because a hard cap such as 3% can distort mega-cap benchmark weights and damage the benchmark proxy.

## 5. Current Factor Definitions

### Momentum

The original default momentum definition was:

```text
log(P_{t-21} / P_{t-252})
```

This is the standard 12-1 momentum idea: approximately 12-month return skipping the most recent month.

After IC sensitivity, the preferred tested specification appears to be:

```text
log(P_t / P_{t-126})
```

This corresponds to 6-month no-skip momentum. In the current `reports/momentum_ic_sensitivity.csv`, the `lookback_days=126`, `skip_days=0` variant has the highest mean Pearson IC among the tested grid, though the IC is still modest.

`main.py` now reads the following from `config.yaml`:

- `factors.momentum.lookback_days`
- `factors.momentum.skip_days`

### Low Volatility

Low volatility is defined as:

```text
lowvol = - annualized rolling volatility of log returns
```

The negative sign means lower volatility receives a higher score. IC diagnostics suggest that low-volatility has negative or weak cross-sectional predictive power in the current setup, so it may be better treated as a risk-control idea rather than a direct alpha factor.

### Short-Term Reversal

Short-term reversal is defined as:

```text
reversal = - recent 1-month log return
```

Stocks with weaker recent short-term performance receive higher reversal scores. Current IC evidence is unstable, and reversal tends to create higher turnover than the other factors.

### Composite

The composite score is an equal-weight combination of z-scored factors:

- Momentum.
- Low-volatility.
- Short-term reversal.

The composite is standardized cross-sectionally again after combination. IC diagnostics show that the initial equal-weight composite does not have robust positive IC in the tested sample.

## 6. Research Findings So Far

1. The initial equal-weight benchmark proxy created benchmark mismatch versus SPY.
2. Strategy results versus SPY looked positive, but strategy results versus a same-universe equal-weight benchmark were weaker and sometimes much less convincing.
3. The static current market-cap benchmark proxy makes the framework conceptually closer to index enhancement.
4. Current market caps are not point-in-time, so using them historically introduces look-ahead bias.
5. Factor IC diagnostics show weak standalone cross-sectional predictive power for the original factors.
6. Momentum is the most promising factor, but the default 12-1 definition is not necessarily best.
7. Momentum IC sensitivity suggests 6-month no-skip momentum performed better than 12-1 momentum in the tested sample.
8. Low-volatility and reversal should be handled carefully because IC evidence is weak or unstable.
9. Strong static market-cap backtest performance should not be interpreted as tradable historical evidence because the benchmark weights use current market caps.

## 7. Known Limitations

- Static current S&P 500 constituents create survivorship bias.
- Static current market-cap weights create look-ahead bias.
- `yfinance` market cap data is not point-in-time.
- Dual-class shares such as GOOG/GOOGL may be double-counted or inconsistently represented.
- The current framework does not use true historical S&P 500 constituent weights.
- Corporate actions and delistings are not fully handled.
- Fundamentals are intentionally avoided because point-in-time fundamental data is unavailable.
- Machine learning has not yet been added because baseline diagnostics are still being improved.
- Results depend strongly on benchmark construction.

## 8. Current Status

The project currently has:

- An end-to-end runnable pipeline via `python main.py`.
- Configurable benchmark mode:
  - `equal_weight`.
  - `market_cap_static`.
- Configurable momentum parameters.
- Robustness reports saved under `reports/`.
- Processed outputs saved under `data/processed/`.
- Figures saved under `reports/figures/`.
- Unit tests for core modules.

The current status is around Version 0.4 / early research diagnostics stage. The framework is now useful for structured experimentation, but the evidence is still prototype-level because of static constituents, static current market caps, and benchmark construction limitations.

## 9. Recommended Next Steps

1. Run the full pipeline with `momentum.lookback_days=126` and `momentum.skip_days=0`.
2. Compare old 12-1 momentum versus new 6-0 momentum using:
   - `factor_ic_summary.csv`
   - `factor_variant_sensitivity.csv`
   - `benchmark_comparison.csv`
   - `subperiod_by_benchmark.csv`
3. Add momentum parameter backtest sensitivity, not only IC sensitivity.
4. Improve the market-cap proxy:
   - Cache market caps.
   - Investigate GOOG/GOOGL treatment.
   - Consider approximate historical weights if possible.
5. Reconsider factor design:
   - Residual momentum.
   - Risk-adjusted momentum.
   - Beta-adjusted features.
   - Volume/liquidity features.
6. Treat low-volatility more as risk control than pure alpha.
7. Add factor weight sensitivity.
8. Only after the above, consider simple machine learning:
   - Ridge / Elastic Net.
   - Tree-based models.
   - Walk-forward validation.
   - ML used to generate alpha scores, not full portfolio weights.

## 10. How to Interpret the Project in an Interview

This project should not be presented as just a backtest. Its strongest value is that it demonstrates the full quant research workflow:

- Data pipeline.
- Factor construction.
- Signal processing.
- Portfolio construction.
- Backtesting with correct timing.
- Transaction cost modeling.
- Robustness checks.
- Benchmark diagnostics.
- Factor IC analysis.

The most important finding is the ability to identify benchmark mismatch and potential false alpha. The project shows that a strategy can look strong versus SPY while being less convincing versus a same-universe benchmark, and that using current market-cap weights can make the framework more realistic conceptually while introducing serious look-ahead bias.

In an interview, the project should be framed as an honest research prototype. The best talking point is not that the strategy has solved index enhancement, but that the framework can diagnose whether apparent performance is likely coming from factor skill, benchmark mismatch, implementation assumptions, or biased data. That is a much stronger and more professional message than overstating headline returns.
