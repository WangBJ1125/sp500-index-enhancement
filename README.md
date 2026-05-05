# S&P 500 Index Enhancement Research Framework

This project is a Python research framework for testing simple S&P 500 index-enhancement ideas. The goal is not to forecast the broad market. The goal is to build a long-only equity portfolio that stays close to an S&P 500 benchmark proxy while taking small, transparent active tilts from cross-sectional price-based factors.

The core research question is:

> Can simple, price-based cross-sectional factors generate positive active returns versus SPY after realistic implementation assumptions such as monthly rebalancing, transaction costs, turnover analysis, and active weight constraints?

## Current Status

Implemented so far:

- Static current S&P 500 universe loader from Wikipedia in `src/universe.py`.
- Price and volume download helpers using `yfinance` in `src/data_loader.py`.
- Simple and log return utilities plus rebalance-date helpers in `src/utils.py`.
- Price-based factor functions in `src/factors.py`:
  - 12-1 momentum.
  - Low-volatility score.
  - Short-term reversal.
- Cross-sectional signal processing in `src/signal_processing.py`:
  - Winsorization.
  - Z-scoring.
  - Factor combination.
- Portfolio construction helpers in `src/portfolio.py`:
  - Equal-weight benchmark proxy.
  - Score-proportional active weights.
  - Active weight caps.
  - Long-only portfolio construction.
  - Turnover calculation.
  - Turnover limit scaling.
- Minimal Version 0 daily backtester in `src/backtester.py`:
  - Rebalance-date score lookup without same-day return use.
  - Trade-lagged weight activation.
  - Gross and net simple-return aggregation.
  - One-time transaction-cost deduction on active trade dates.
- Reporting helpers in `src/reporting.py`:
  - NAV, active return, drawdown, turnover, and rolling risk plots.
  - Markdown summary report with performance, costs, turnover, and limitations.

Not implemented yet:

- Factor IC diagnostics.
- Robustness analysis.

## Research Design

Version 0 is a simple rule-based baseline:

- Use current S&P 500 constituents as the static universe.
- Use adjusted close prices and volume from `yfinance`.
- Use SPY as the return benchmark.
- Compute factors from historical price data only.
- Process signals cross-sectionally by date.
- Build a long-only portfolio around an equal-weight benchmark proxy.
- Apply rebalance weights after a configurable trade lag.
- Track turnover and deduct transaction costs on trade activation dates.

For factor construction, the project primarily uses log returns because they are time-additive and convenient for momentum and volatility estimation. For portfolio aggregation and NAV calculation, the project uses simple returns because portfolio returns are linear in asset simple returns.

## Important Limitation

This prototype uses the current S&P 500 constituents as a static universe. This introduces survivorship bias because stocks that were previously in the index but later removed are not included. Therefore, results should be interpreted as a research prototype rather than production-level evidence.

The project also avoids current fundamental data in historical backtests unless point-in-time data is available. The first-stage signals are price-based.

## Repository Structure

```text
sp500-index-enhancement/
|-- config.yaml
|-- main.py
|-- requirements.txt
|-- src/
|   |-- data_loader.py
|   |-- factors.py
|   |-- portfolio.py
|   |-- signal_processing.py
|   |-- universe.py
|   `-- utils.py
|-- tests/
|-- data/
|-- notebooks/
`-- reports/
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Run the Version 0 pipeline:

```bash
python main.py
```

Outputs are saved to `data/processed/`, plots are saved to `reports/figures/`, and the markdown report is saved to `reports/summary_report.md`.

## Portfolio Construction

The current portfolio construction follows:

```text
weights = benchmark_weights + active_weights
```

The benchmark proxy is equal-weight across valid tickers. Active weights are created by centering cross-sectional scores, scaling them to an active budget, and optionally applying active weight caps. Final portfolio weights are clipped to long-only, optionally capped by single-name maximum weight, and normalized to sum to one.

This structure is intentionally simple for Version 0 but leaves room for Version 1 controls such as tighter active-weight constraints, turnover limits, sector neutralization, and more realistic benchmark weights.

## Backtest Timing

The current Version 0 backtester forms weights at rebalance date `t` from scores available at or before `t`. With the default `trade_lag_days=1`, those weights are first applied on the next trading day, so rebalance-day returns are never earned with same-day signals. Gross portfolio returns are computed from held weights and daily stock simple returns. If a held ticker has a missing return on a holding day, the backtester treats that missing return as zero; production research should filter or investigate those gaps before relying on results.

Transaction costs are deducted only on the first active holding day for each rebalance. The current cost convention is:

```text
cost = 2 * (transaction_cost_bps / 10000) * turnover
```
