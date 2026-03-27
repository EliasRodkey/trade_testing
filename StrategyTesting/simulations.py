"""
simulations.py - Top-level runner that ties together BoundSimulators and GridSearchOptimizer
                 to execute full simulation sets across MA types and position sizes.

Functions:
    simulate_simset: Runs a complete grid search optimization over all MA types and
                     max_active_positions values for a given signal/preference function pair.

Module-level variables:
    signals: Shared Signals() instance used across simulation runs.
    compatible: List of MA type strings compatible with MA-based signal functions.
"""
from multiprocessing import Process
from typing import Callable

from simulator import BoundSimulators
from grid_search_optimizer import GridSearchOptimizer
from ToolKit.signal_generator import Signals


signals = Signals() 

compatible = [
            "SMA","TMA","WMA",
            "EMA","DEMA", "TEMA"
        ]

def simulate_simset(
    signal_function: Callable,
    signal_args: dict,
    preference_function: Callable,
    preference_args: dict,
    ma_types: list=compatible,
    maxpos_range: range=range(5, 25, 5),
    initial_cash: int=10000,
    trade_fee: float=0,
    contains_ma_type: bool=False,
    save_results: bool=True
) -> GridSearchOptimizer:
    """
    Runs a grid search optimization across all combinations of MA type and max_active_positions.

    For each (ma_type, maxpos) pair, constructs a BoundSimulators instance and runs a full
    GridSearchOptimizer.optimize() call. Returns the optimizer from the last run.

    Args:
        signal_function (Callable): Signal generation function passed to BoundSimulators.
        signal_args (dict): Parameter ranges for the signal function, passed to optimizer.optimize().
        preference_function (Callable): Preference scoring function passed to BoundSimulators.
        preference_args (dict): Parameter ranges for the preference function, passed to optimizer.optimize().
        ma_types (list): List of MA type strings to iterate over. Defaults to compatible.
        maxpos_range (range): Range of max_active_positions values to iterate over. Defaults to range(5, 25, 5).
        initial_cash (int): Starting portfolio cash passed to BoundSimulators. Defaults to 10000.
        trade_fee (float): Fixed trade fee per transaction passed to BoundSimulators. Defaults to 0.
        contains_ma_type (bool): Whether the signal function accepts ma_type as a kwarg. Defaults to False.
        save_results (bool): Whether to save results to CSV after each optimization run. Defaults to True.

    Returns:
        GridSearchOptimizer: The optimizer instance from the final simulation run.
    """
    optimizer = None
    for ma_type in ma_types:
        for maxpos in maxpos_range:
            simclass = BoundSimulators(
                signal_function,
                preference_function,
                contains_ma_type=contains_ma_type,
                initial_cash=initial_cash,
                max_active_positions=maxpos,
                trade_fee=trade_fee
            )
            simclass.ma_type=ma_type
            optimizer = GridSearchOptimizer(simclass.simulate, save_results=save_results)
            optimizer.optimize(
                signal_ranges=signal_args,
                preference_ranges=preference_args
            )
    return optimizer


if __name__ == "__main__":
    signal_args = {
        "signal_n" : range(5, 100, 10),
        "upper_bound" : range(70, 100, 10),
        "lower_bound" : range(10, 40, 10)
    }
    preference_args = {
        "preference_n" : range(5, 75, 10)
    }

    p = Process(
        target=simulate_simset, 
        args=(
            signals.create_stochastic_rsi_signals,
            signal_args, signals.calculate_rolling_sharpe_ratio,
            preference_args
            ),
        kwargs ={
            "contains_ma_type" : True,
            "ma_types" : [compatible[2]],
            "maxpos_range" : range(20, 25, 5),
            "save_results" : False
        }
    )
    p.start()

    # p = Process(
    #     target=simulate_simset, 
    #     args=(
    #         signals.create_stochastic_rsi_signals,
    #         signal_args, signals.calculate_rolling_sharpe_ratio,
    #         preference_args
    #         ),
    #     kwargs ={
    #         "contains_ma_type" : True,
    #         "ma_types" : compatible[3:],
    #         "save_results" : True
    #     }
    # )
    # p.start()
