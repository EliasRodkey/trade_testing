#! contains simulation funcitons to run multiple optimizations concurrently
from multiprocessing import Process
from simulator import BoundSimulators
from grid_search_optimizer import GridSearchOptimizer
from ToolKit.signal_generator import Signals

# untested signals
# KAMA
# VWAP not really sure how to do this one right now, maybe figure it out later
# Stochastic Oscillator
# RSI
# Stochastic RSI
# Williams %r

for max_pos in range(5, 25, 5):
    sim_class = BoundSimulators( 
        Signals().create_KAMA_signals,
        Signals().calculate_rolling_sharpe_ratio,
        initial_cash=10000, max_active_positions=max_pos,
        trade_fee=0
        )
    KAMA_optimizer = GridSearchOptimizer(sim_class.simulate_kama)
    KAMA_optimizer.optimize(signal_n=range(5, 65, 10), fast_lookback=range(5, 30, 5), slow_lookback=range(30, 150, 5), preference_n=range(5, 65, 5))
    KAMA_optimizer.save_results()

compatible = [
            "simple_moving_average",
            "triangular_moving_average",
            "weighted_moving_average",
            "exponential_moving_average",
            "DEMA", "TEMA"
        ]
    
signals = Signals()
signal_functions =[
    signals.create_stochastic_oscillator_signals,
    signals.create_relative_strength_index_signals,
    signals.create_stochastic_rsi_signals,
    signals.create_williams_r_signals
]

for ma_type in compatible:
    for signal_function in signal_functions:
        for max_pos in range(5, 25, 5):
            sim_class = BoundSimulators( 
                signal_function,
                signals.calculate_rolling_sharpe_ratio,
                initial_cash=10000, max_active_positions=max_pos,
                trade_fee=0
                )
            sim_class.ma_type = ma_type
            stoch_optimizer = GridSearchOptimizer(sim_class.simulate_static_bound_oscillator)
            stoch_optimizer.optimize(
                signal_n=range(5, 120, 10), 
                upper_bound=range(70, 100, 10), 
                lower_bound=range(10, 40, 10), 
                preference_n=range(5, 100, 5)
            )
            stoch_optimizer.save_results()