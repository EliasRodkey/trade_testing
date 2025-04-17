#! python3
# 
# optimizer_001.py - first simulation optimizer.
# GridSearOptomize given input of a simulation funciton. the .optomize function
# is called with a dicitonary of iterables which are tried in all combinations
# by the optimizer which then returns data on the most successful parameters

import os
import datetime
from time import perf_counter
from ToolKit.data_loading import RESULTS_PATH
import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Callable, Iterable, Any, NewType, Mapping

# Simulation function must take parameters as keyword arguments pointing to 
# iterables and return a performance metric dictionary
SimKwargs = NewType('Kwargs', Mapping[str, Iterable[Any]])


class OptimizationResult(object):
    """Simple container class for optimization data"""

    def __init__(self, parameters: Dict[str, int], performance: pd.DataFrame, metadata: Dict):

        # Make sure no collisions between performance metrics and params
        assert len(parameters.keys() & performance.keys()) == 0, \
            'parameter name matches performance metric name'

        self.parameters = self.as_pd(parameters)
        self.performance = performance
        self.metadata = self.as_pd(metadata)

    @staticmethod
    def as_pd(params: Dict[str, int]) -> pd.DataFrame:
        return pd.DataFrame(params, index=[0])

    @property
    def combined(self) -> Dict[str, float]:
        """Combines the dictionaries after we are sure of no collisions"""
        return pd.concat([self.metadata, self.parameters, self.performance], axis=1)
    

class GridSearchOptimizer(object):
    """
    A generic grid search optimizer that requires only a simulation function and
    a series of parameter ranges. Provides timing, summary, and plotting 
    utilities with return data.
    """

    def __init__(self, simulation_function: Callable, save_results: bool=True):

        self.simulate = simulation_function
        self._results_list: List[OptimizationResult] = list()
        self.time_df = pd.DataFrame()
        self.save = save_results
        # self.ui = Ui_MainWindow()
        # self.ui.show_ui()

        self._optimization_finished = False

    def add_results(self, parameters: Dict[str, int], performance: pd.DataFrame, metadata: Dict):
        _results = OptimizationResult(parameters, performance, metadata)
        self._results_list.append(_results.combined)
    
    def _add_to_time_df(self, times_to_add: pd.DataFrame):
        if self.time_df.empty:
            self.time_df = pd.DataFrame(columns=times_to_add.columns)
        _to_concat = [self.time_df, times_to_add]
        self.time_df = pd.concat(_to_concat, axis=0).reset_index(drop=True)

    def optimize(self, signal_ranges: dict, preference_ranges: dict):
        assert signal_ranges and preference_ranges, 'Must provide non-empty parameters.'

        # Convert all iterables to lranges
        signal_param_ranges = {k: list(v) for k, v in signal_ranges.items()}
        preference_param_ranges = {k: list(v) for k, v in preference_ranges.items()}
        param_ranges = signal_param_ranges | preference_param_ranges

        self.signal_param_names = signal_param_names = list(signal_param_ranges.keys())
        self.preference_param_names = preference_param_names = list(preference_param_ranges.keys())
        self.param_names = param_names = signal_param_names + preference_param_names

        # Count total simulation
        n = total_simulations = np.prod(
            [len(r) for r in signal_param_ranges.values()] + [len(r) for r in preference_param_ranges.values()]
        )

        total_time_elapsed = 0

        print(f'Starting simulation ...')
        returns = []
        wins = []
        trades = []
        for i, params in enumerate(product(*param_ranges.values())):
            start_time = perf_counter()
            
            if i > 0:
                s = f"""
                ____________________________________
                |########     SIM LOG    ######## 
                |   {self.ID}  
                |                                   
                |Simulating {i+1} / {total_simulations}...              
                |Expected Time Remaining : {round((n - (i + 1)) * self.time_df.total_time.mean(), 0)}s   
                |                                 
                |Avg Sim Time : {round(self.time_df.total_time.mean(), 2)}s       
                |Elapsed Time : {round(self.time_df.total_time_elapsed.mean(), 2)}s              
                |                                 
                |Avg Simulated Return : {round(100*np.average(returns), 2)}%   
                |Avg Simulated Win Percent : {100*round(np.average(wins), 2)}% 
                |Avg Number of Trades : {round(np.average(trades), 2)}    
                |___________________________________
                """
                print(s)
            else:
                print(f'Simulating: 1 / {total_simulations}...')

            parameters = {n: param for n, param in zip(param_names, params)}
            signal_params = params[:len(signal_param_names)]
            preference_params = params[len(signal_param_names):]

            sim, results = self.simulate(signal_params, preference_params)
            if i == 0:
                self.sim = sim
                self.ID
            self.make_metadata_dict(self.sim)
            self.add_results(parameters, results, self._metadata)
            returns.append(results.percent_return.iloc[0])
            wins.append(results.positive_trade_ratio.iloc[0])
            trades.append(results.number_of_trades.iloc[0])

            end_time = perf_counter()
            total_time = end_time - start_time
            total_time_elapsed += total_time
            _temp_time_data = {}
            _temp_time_data["total_time"] = [total_time]
            _temp_time_data["total_time_elapsed"] = [total_time_elapsed]
            self._add_to_time_df(pd.DataFrame(_temp_time_data))

        print(f'Simulated {total_simulations} / {total_simulations} ...')
        print(f'Elapsed time: {total_time_elapsed:.0f}s')
        print(f'Done')

        self._optimization_finished = True
        if self.save:
            self.save_results()

    def _assert_finished(self):
        assert self._optimization_finished, \
            'Run self.optimize before accessing this method.'

    @property
    def results(self) -> pd.DataFrame:
        self._assert_finished()
        self._results = pd.concat(self._results_list, axis=0).reset_index().drop(columns=["index"])

        _columns = set(list(self._results.columns.values))
        _params = set(self.param_names)
        self.metric_names = list(_columns - _params)

        return self._results

    def print_summary(self):
        df = self.results
        metric_names = self.metric_names

        print('Summary statistics')
        print(df[metric_names].describe().T)

    def make_metadata_dict(self, sim_class):
        # Creates a dictionary with all of the meta information about the optimization
        # this information is condensed in the simulation ID but is also included
        # in the results to easily reference simulation types etc
        signal, pref = list(sim_class.params)
        _dict = {
            "id" : self.ID,
            "signal" : signal,
            "preference" : pref,
            "max_positions" : sim_class.max_active_positions,
            "date" : datetime.datetime.now().strftime("%y%m%d")
        }
        self._metadata = _dict

    def _make_id(self, sim_class) -> str:
        """generates unique identifier for bound simulator"""
        signal, pref = list(sim_class.params)
        _params = f"{signal}_{pref}"
        _date = datetime.datetime.now().strftime("%y%m%d")
        iterator = 0
        base_ID = f"{_params}_{_date}"
        save_path = os.path.join(RESULTS_PATH, f"{base_ID}_{iterator}.csv")
        while os.path.exists(save_path):
            iterator += 1
            save_path = os.path.join(RESULTS_PATH, f"{base_ID}_{iterator}.csv")
        return f"{_params}_{_date}{iterator}"

    @property
    def ID(self) -> str:
        return self._make_id(self.sim)

    @property
    def winning_sim_ratio(self) -> int:
        # returns the ratio of winning (beating the market return) simualtions
        # as a percentage
        simset = self.results
        filt = simset["percent_return"] > simset["spy_percent_return"]
        number_winning = simset[filt].shape[0]
        ratio = number_winning / simset.shape[0]
        return ratio

    def compile_simset(self) -> pd.DataFrame:
        # returns the mean metrics along with information about how many sets beat
        # the spy returns
        meta = self.results.describe()
        meta.drop(columns=self.param_names, inplace=True)
        compiled = meta.loc["mean"]
        compiled["edge_score_stdev"] = self.results["edge_score"].std()
        compiled["simset_size"] = self.results.shape[0]
        compiled["percent_winning_sims"] = self.winning_sim_ratio
        compiled["id"] = self.ID
        return compiled.to_frame().T

    @property
    def compiled(self) -> pd.DataFrame:
        return self.compile_simset()        

    def add_to_compiled_results(self):
        # adds compilation of results to "compiled_results" file
        compiled_results_path = os.path.join(
            "StrategyTesting", 
            "optimization_results", 
            "compiled_results.csv"
        )
        try:
            old_results = pd.read_csv(compiled_results_path)
            new_results = pd.concat(
                [old_results, self.compiled], axis=0
            ).reset_index().drop(columns=["index", "Unnamed: 0"])
            new_results.to_csv(compiled_results_path)
        except:
            self.compiled.to_csv(compiled_results_path)

    def save_results(self):
        # saves the results of the grid search optomization to a CSV file 
        # self.add_to_compiled_results()
        results_dir = os.path.join("StrategyTesting", "optimization_results", f"{self.ID[:-2]}")
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        simset_path = os.path.join(results_dir, f"{self.ID}.csv")
        self.results.to_csv(simset_path)


# #Optimizer Usage
# if __name__ == '__main__':
#     # GridSearOptomizer example usage
#     from simulator import BoundSimulators
#     import cProfile

#     simulate =  BoundSimulators(
#         Signals().create_MA_signals,
#         Signals().calculate_rolling_sharpe_ratio,
#         initial_cash=10000, max_active_positions=5
#     )
#     optimizer = GridSearchOptimizer(simulate.simulate_single_ma_lookback)
#     simulate.ma_type = "SMA"
#     optimizer.optimize(signal_n=range(10, 15, 5),performance_n=range(20, 30, 5))
#     cProfile.run('optimizer.optimize(signal_n=range(10, 15, 5),performance_n=range(20, 30, 5))')
#     optimizer.save_results()
#     optimizer.print_summary()
#     print(optimizer.get_best('excess_cagr'))
#     optimizer.save_results() 
#     optimizer.plot('excess_cagr')
#     optimizer.plot('bollinger_n', 'excess_cagr')
#     optimizer.plot('bollinger_n', 'sharpe_n', 'excess_cagr')