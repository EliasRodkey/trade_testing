#! contains simulation funcitons to run multiple optimizations concurrently
from multiprocessing import Process
from simulator import BoundSimulators
from grid_search_optimizer import GridSearchOptimizer
from ToolKit.signal_generator import Signals
import pandas as pd
import os
from typing import Callable, List


for max_pos in range(5, 25, 5):
    sim_class = BoundSimulators( 
        Signals().create_macd_signals,
        Signals().calculate_rolling_sharpe_ratio,
        initial_cash=10000, max_active_positions=max_pos,
        trade_fee=0
        )
    optimizer = GridSearchOptimizer(sim_class.simulate_dual_lookback)
    optimizer.optimize(signal_n1=range(5, 30, 5), signal_n2=range(30, 150, 5), preference_n=range(5, 65, 5))
    optimizer.save_results()