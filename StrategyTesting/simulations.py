#! contains simulation funcitons to run multiple optimizations concurrently

from multiprocessing import Process
from simulator_21LNG001 import BoundSimulators
from grid_search_optimizer import GridSearchOptimizer
from ToolKit.signal_generator import Signals
from ToolKit.data_loading import get_all_symbols, load_data_as_pd, load_eod_matrix
import pandas as pd
from typing import Callable, List


for i in range(5, 30, 5):
    sim_class = BoundSimulators(
        Signals().create_bollinger_band_signal,
        Signals().calculate_rolling_sharpe_ratio,
        initial_cash=10000, max_active_positions=i
        )
    boll_opt = GridSearchOptimizer(sim_class.simulate_lookback_only)
    boll_opt.optimize(signal_n=range(5, 100, 5), preference_n=range(10, 100, 5))
    boll_opt.save_results()

# boll_simulate = bind_bollinger_simulator(initial_cash=10000, max_active_positions=5)
# boll_optimizer = GridSearchOptimizer(boll_simulate)
# boll_optimizer.optimize(bollinger_n=range(5, 100, 5), sharpe_n=range(10, 100, 10))
# boll_optimizer.save_results()

# mom_simulate = bind_MOM_simulator(initial_cash=10000, max_active_positions=5)
# mom_optimizer = GridSearchOptimizer(mom_simulate)
# mom_optimizer.optimize(momentum_n=range(5, 100, 5), sharpe_n=range(10, 100, 10))
# mom_optimizer.save_results()

# sma_simulate = bind_SMA_simulator("simple_moving_average", initial_cash=10000, max_active_positions=5)
# sma_optimizer = GridSearchOptimizer(sma_simulate)
# sma_optimizer.optimize(moveaverage_n=range(5, 100, 5), sharpe_n=range(10, 100, 10))
# sma_optimizer.save_results()

# tma_simulate = bind_SMA_simulator("triangular_moving_average", initial_cash=10000, max_active_positions=5)
# tma_optimizer = GridSearchOptimizer(tma_simulate)
# tma_optimizer.optimize(moveaverage_n=range(5, 100, 5), sharpe_n=range(10, 100, 10))
# tma_optimizer.save_results()

# wma_simulate = bind_SMA_simulator("weighted_moving_average", initial_cash=10000, max_active_positions=5)
# wma_optimizer = GridSearchOptimizer(wma_simulate)
# wma_optimizer.optimize(moveaverage_n=range(5, 100, 5), sharpe_n=range(10, 100, 10))
# wma_optimizer.save_results()

# ema_simulate = bind_SMA_simulator("exponential_moving_average", initial_cash=10000, max_active_positions=5)
# ema_optimizer = GridSearchOptimizer(ema_simulate)
# ema_optimizer.optimize(moveaverage_n=range(5, 100, 5), sharpe_n=range(10, 100, 10))
# ema_optimizer.save_results()

# dema_simulate = bind_SMA_simulator("DEMA", initial_cash=10000, max_active_positions=5)
# dema_optimizer = GridSearchOptimizer(dema_simulate)
# dema_optimizer.optimize(moveaverage_n=range(5, 100, 5), sharpe_n=range(10, 100, 10))
# dema_optimizer.save_results()

# tema_simulate = bind_SMA_simulator("TEMA", initial_cash=10000, max_active_positions=5)
# tema_optimizer = GridSearchOptimizer(tema_simulate)
# tema_optimizer.optimize(moveaverage_n=range(5, 100, 5), sharpe_n=range(10, 100, 10))
# tema_optimizer.save_results()
