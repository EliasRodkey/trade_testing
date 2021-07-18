#! contains simulation funcitons to run multiple optimizations concurrently
from multiprocessing import Process

from simulator import BoundSimulators
from grid_search_optimizer import GridSearchOptimizer
from ToolKit.signal_generator import Signals


signals = Signals() 

compatible = [
            "SMA","TMA","WMA",
            "EMA","DEMA", "TEMA"
        ]

def simulate_simset(
    signal_funciton, 
    signal_args: dict,
    preference_function,
    preference_args: dict,
    ma_types:list=compatible,
    maxpos_range: range=range(5, 25, 5),
    initial_cash: int=10000,
    trade_fee: float=0,
    contains_ma_type: bool=False,
    save_results: bool=True
):
    # makes running simsets easier by groupoing related functions into 1
    for ma_type in ma_types:
        for maxpos in maxpos_range:
            simclass = BoundSimulators(
                signal_funciton,
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
            "save_results" : True
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
