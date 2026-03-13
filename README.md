# Trade Testing

An algorithmic trading backtester built to explore technical analysis and systematic strategy development in Python.

---

## Overview

Trade Testing is a Python-based backtesting framework that simulates trading strategies against historical end-of-day (EOD) stock price data. It was built as a learning project based on the book "Algorithmic Trading with Python" by Chris Conlan to develop both software engineering skills and a deeper understanding of quantitative finance concepts — particularly how technical indicators translate into systematic buy/sell decisions and how those decisions perform over time relative to the market average.

The system supports:
- Generating buy/sell signals from a library of technical indicators
- Running simulated trades with configurable cash, position limits, slippage, and fees
- Grid-searching across parameter ranges to find optimal strategy configurations
- Computing comprehensive portfolio performance metrics (returns, risk-adjusted metrics, alpha/beta vs. S&P 500)

---

## Architecture

Data flows through four sequential stages:

```
1. Historical EOD CSVs      
2. data_loading.py  - Load price data into DataFrames     
3. signal_generator.py - Apply technical indicators, generate signal DataFrames
4. simulator.py - Execute trades against signals, track cash and positions        
5. portfolio.py - Aggregate positions, compute performance metrics
6. grid_search_optimizer - Run all parameter combinations, save results to CSV
```

`simulations.py` is the top-level runner that ties these stages together and supports running multiple optimization jobs concurrently via `multiprocessing`.

---

## Features

**Technical Indicators (`ToolKit/signal_generator.py`)**
- Moving averages: SMA, TMA, WMA, EMA, DEMA, TEMA, KAMA
- Oscillators: MACD, RSI, Stochastic, Stochastic RSI, Williams %R, CCI, ROC, PPO
- Volume-based: VWAP, Money Flow Index, Chaikin Money Flow, On-Balance Volume
- Trend: Bollinger Bands, ATR, ADX, Aroon, Trix, Ultimate Oscillator

**Signal Generators**
- Moving average crossovers
- Zero-crossing (oscillator-based)
- Range-exceeding (price breaks bands/thresholds)
- Indicator-specific signals (RSI overbought/oversold, Bollinger Bands, KAMA, VWAP)

**Performance Metrics**
- Return metrics: percent return, CAGR, log returns
- Risk metrics: annualized volatility, max drawdown, Sharpe ratio, Sortino ratio, Calmar ratio
- Benchmark comparison: alpha, beta, R-squared, Jensen's alpha vs. S&P 500
- Trade statistics: win ratio, average trade length, average return per trade

**Optimization**
- Grid search across any combination of strategy parameters
- Supports concurrent simulation runs via multiprocessing
- Results saved to organized CSV files per simulation set

---

## Limitations & Shortcomings

This project was built to learn — it works, but has known limitations:

- **No live trading support.** This is strictly a backtesting tool; it has no broker integration or real-time data feeds.
- **Test data is not included.** Historical EOD CSV files must be sourced and added manually (see Setup below). Only the S&P 500 (SPY) benchmark file structure is referenced in code.
- **No configuration system.** Parameters are set directly in code rather than via config files or CLI arguments.
- **Optimization is CPU-bound.** Grid search can be slow for large parameter spaces; the multiprocessing support is basic and has not been heavily optimized.
- **No walk-forward or out-of-sample validation.** Results are in-sample only and are susceptible to overfitting.

---

## What This Project Demonstrates

- **Object-oriented design** — Clear separation of concerns across `Position`, `PortfolioHistory`, `SimpleSimulator`, `BoundSimulators`, `GridSearchOptimizer`, `Metrics`/`Indicators`/`Signals`
- **Inheritance hierarchy** — `Signals` extends `Indicators` extends `Metrics`, keeping financial calculations composable and reusable
- **Performance awareness** — Use of `itertuples` over `iterrows` for simulation loops, documented in code
- **Financial domain modeling** — Slippage, trade fees, cash allocation, and benchmark-relative metrics modeled explicitly
- **Growing comfort with type safety** — Type hints and structured docstrings applied throughout the codebase

---

## Setup

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Add historical price data:**

Place EOD CSV files in `StrategyTesting/test_data/eod/`, one file per ticker (e.g. `AAPL.csv`). Files must have columns: `date, open, high, low, close, volume` with `date` as the index.

Place the S&P 500 benchmark file at `StrategyTesting/test_data/SPY.csv` in the same format.

**Run a simulation:**
```bash
cd StrategyTesting
python simulations.py
```

Each module also has runnable example code under its `if __name__ == "__main__":` block.

---

## Project Status

This is an intermediate personal project built while learning Python and algorithmic trading. It is functional, but is not production-ready. Contributions and feedback are welcome.
