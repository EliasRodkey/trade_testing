# Trade Algorithm Backtester

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

**Run a simulation:**
```bash
cd StrategyTesting
python simulations.py
```

Each module also has runnable example code under its `if __name__ == "__main__":` block.

---

## Project Status

This is an intermediate personal project built while learning Python and algorithmic trading. It is functional, but is not production-ready. Contributions and feedback are welcome.

---

## Future Direction

A number of improvements have been considered for future application to this project to make it more practical and usable. Some possible additions are listed below:

**1. Simulation UI**

Adding a graphic user interface to allow dynamic simulation parameter assignment would increase the functionality and overall usefulness of this project. A programmatic interface could also be added to allow other applications or programs run simulations and utilize the results.

**2. Automated Results Sorting**

Each simulation generates a large data file containing results and metadata about the performance of each strategy. These files are difficult to parse manually, especially if multiple strategies are being tested in bulk. A component that can compare test results from different strategies against one another and against a standard market performance would meaningfully increase the value of the project as a stock trading tool.

**3. Forward Testing**

This system tests the strategies on historical EOD stock data. It is a possible that a loose form of overfitting could account for the performance of some strategies. This means some trading strategies may not work in the current market or reveal meaningful insights that can be used for making trades. Some data should be set aside for verification during testing, and there should also be a means to deploy a testing strategy using paper testing to see whether or not it is a viable approach.

**4. Automated Trade Placement**

Once a strategy has been selected and the selection process refined, empowering agentic trading bots using Alpaca to place trades autonomously is a logical next step.
