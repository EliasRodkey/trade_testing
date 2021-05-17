#! contains simulation funcitons to run multiple optimizations concurrently

from multiprocessing import Process
from simulator_21LNG001 import BoundSimulators
from grid_search_optimizer import GridSearchOptimizer
from ToolKit.signal_generator import Signals
from ToolKit.data_loading import get_all_symbols, load_data_as_pd, load_eod_matrix
import pandas as pd
import os
from typing import Callable, List


# sims = [
#     BoundSimulators(    
#     Signals().create_momentum_signal,
#     Signals().calculate_rolling_sharpe_ratio,
#     initial_cash=10000, max_active_positions=20,
#     trade_fee=0
#     ),
#     BoundSimulators(
#     Signals().create_bollinger_band_signal,
#     Signals().calculate_rolling_sharpe_ratio,
#     initial_cash=10000, max_active_positions=20,
#     trade_fee=0
#     ),
#     BoundSimulators(
#     Signals().create_bollinger_band_signal,
#     Signals().calculate_aroon_oscillator,
#     initial_cash=10000, max_active_positions=20,
#     trade_fee=0
#     ),
#     BoundSimulators(
#     Signals().create_macd_signal,
#     Signals().calculate_rolling_sharpe_ratio,
#     initial_cash=10000, max_active_positions=20,
#     trade_fee=0
#     ),
#     BoundSimulators(
#     Signals().create_macd_signal,
#     Signals().calculate_aroon_oscillator,
#     initial_cash=10000, max_active_positions=20,
#     trade_fee=0
#     ),
# ]

# to_pd = []
# for sim in sims:
#     optimizer = GridSearchOptimizer(sim.simulate_lookback_only)
#     optimizer.optimize(signal_n=range(10, 20, 5), preference_n=range(20, 30, 5))
#     optimizer.save_results()
#     if sim == sims[2]:
#         break


boll_sim_class = BoundSimulators(
    Signals().create_bollinger_band_signal,
    Signals().calculate_rolling_sharpe_ratio,
    initial_cash=10000, max_active_positions=5,
    trade_fee=0
    )
boll_opt = GridSearchOptimizer(boll_sim_class.simulate_lookback_only)
boll_opt.optimize(signal_n=range(30, 40, 5), preference_n=range(30, 35, 5))
boll_opt.save_results()