#! python3
# 
# optimizer_001.py - first simulation optimizer.
# GridSearOptomize given input of a simulation funciton. the .optomize function
# is called with a dicitonary of iterables which are tried in all combinations
# by the optimizer which then returns data on the most successful parameters

import os
import sys
import datetime
from timeit import default_timer
from ToolKit.signal_generator import Signals 
from ToolKit.data_loading import load_eod_matrix, get_all_symbols
from ToolKit.log_window import Ui_MainWindow
import pandas as pd
import numpy as np
from collections import OrderedDict
from itertools import product
from reprint import output
from typing import Dict, Tuple, List, Callable, Iterable, Any, NewType, Mapping

import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D 

# Simulation function must take parameters as keyword arguments pointing to 
# iterables and return a performance metric dictionary
SimKwargs = NewType('Kwargs', Mapping[str, Iterable[Any]])


class OptimizationResult(object):
    """Simple container class for optimization data"""

    def __init__(self, parameters: Dict[str, int], performance: pd.DataFrame):

        # Make sure no collisions between performance metrics and params
        assert len(parameters.keys() & performance.keys()) == 0, \
            'parameter name matches performance metric name'

        self.parameters = self.as_pd(parameters)
        self.performance = performance

    @staticmethod
    def as_pd(params: Dict[str, int]) -> pd.DataFrame:
        return pd.DataFrame(params, index=[0])

    @property
    def combined(self) -> Dict[str, float]:
        """Combines the dictionaries after we are sure of no collisions"""
        return pd.concat([self.parameters, self.performance], axis=1)
    

class GridSearchOptimizer(object):
    """
    A generic grid search optimizer that requires only a simulation function and
    a series of parameter ranges. Provides timing, summary, and plotting 
    utilities with return data.
    """

    def __init__(self, simulation_function: Callable):

        self.simulate = simulation_function
        self._results_list: List[OptimizationResult] = list()
        self.time_df = pd.DataFrame()
        # self.ui = Ui_MainWindow()
        # self.ui.show_ui()

        self._optimization_finished = False

    def add_results(self, parameters: Dict[str, int], performance: pd.DataFrame):
        _results = OptimizationResult(parameters, performance)
        self._results_list.append(_results.combined)
    
    def _add_to_time_df(self, times_to_add: pd.DataFrame):
        if self.time_df.empty:
            self.time_df = pd.DataFrame(columns=times_to_add.columns)
        _to_concat = [self.time_df, times_to_add]
        self.time_df = pd.concat(_to_concat, axis=0).reset_index().drop(columns=["index"])

    def optimize(self, **optimization_ranges: SimKwargs):
        assert optimization_ranges, 'Must provide non-empty parameters.'

        # Convert all iterables to lists
        param_ranges = {k: list(v) for k, v in optimization_ranges.items()}
        self.param_names = param_names = list(param_ranges.keys())

        # Count total simulation
        n = total_simulations = np.prod([len(r) for r in param_ranges.values()])

        total_time_elapsed = 0

        print(f'Starting simulation ...')
        returns = []
        wins = []
        trades = []
        for i, params in enumerate(product(*param_ranges.values())):
            timer_start = default_timer()
            
            if i > 0:
                s = f"""
    ____________________________________
    |########     SIM LOG    ######## 
    |                                 
    |Simulating {i+1} / {total_simulations}...              
    |Expected Time Remaining : {round((n - (i + 1)) * self.time_df.total_time.mean(), 0)}s   
    |                                 
    |Setup Time : {round(self.time_df.setup_time.mean(), 4)}s              
    |Calculation Time : {round(self.time_df.calculation_time.mean(), 4)}s       
    |Transaction Time : {round(self.time_df.transaction_time.mean(), 4)}s        
    |Sim Iteration Time : {round(self.time_df.internal_sim_time.mean(), 2)}s          
    |Sim Funciton Time : {round(self.time_df.mid_sim_time.mean(), 2)}s             
    |Sim Optimization Time : {round(self.time_df.outside_sim_time.mean(), 2)}s       
    |Analysis Time : {round(self.time_df.finish_time.mean(), 4)}s           
    |Recording Time : {round(self.time_df.record_results_time.mean(), 4)}s         
    |Total Time : {round(self.time_df.total_time.mean(), 2)}s 
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

            print_time = default_timer()
            parameters = {n: param for n, param in zip(param_names, params)}
            params_time = default_timer()
            sim, results = self.simulate(*params)
            if i == 0:
                self.sim = sim
            timer_mid = default_timer()
            self.add_results(parameters, results)
            returns.append(results.percent_return.iloc[0])
            wins.append(results.positive_trade_ratio.iloc[0])
            trades.append(results.number_of_trades.iloc[0])

            timer_end = default_timer()
            total_time_elapsed += timer_end - timer_start 
            sim.time_data["print_time"] = print_time - timer_start
            sim.time_data["outside_sim_time"] = timer_mid - params_time 
            sim.time_data["record_results_time"] = timer_end - timer_mid
            sim.time_data["total_time"] = timer_end - timer_start
            sim.time_data["total_time_elapsed"] = total_time_elapsed
            self._add_to_time_df(sim.time_data)

        print(f'Simulated {total_simulations} / {total_simulations} ...')
        print(f'Elapsed time: {total_time_elapsed:.0f}s')
        print(f'Done')

        self._optimization_finished = True

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

    def get_best(self, metric_name: str) -> pd.DataFrame:
        """
        Sort the results by a specific performance metric
        """
        self._assert_finished()

        results = self.results
        param_names = self.param_names
        metric_names = self.metric_names

        assert metric_name in metric_names, 'Not a performance metric'
        partial_df = self.results[param_names+[metric_name]]

        return partial_df.sort_values(metric_name, ascending=False)

    def plot_1d_hist(self, x, show=True):
        self.results.hist(x)
        if show:
            plt.show()

    def plot_2d_line(self, x, y, show=True, **filter_kwargs):
        _results = self.results
        for k, v in filter_kwargs.items():
            _results = _results[getattr(_results, k) == v]

        ax = _results.plot(x, y)
        if filter_kwargs:
            k_str = ', '.join([f'{k}={v}' for k,v in filter_kwargs.items()])
            ax.legend([f'{x} ({k_str})'])

        if show:
            plt.show()

    def plot_2d_violin(self, x, y, show=True):
        """
        Group y along x then plot violin charts
        """
        x_values = self.results[x].unique()
        x_values.sort()

        y_by_x = OrderedDict([(v, []) for v in x_values])
        for _, row in self.results.iterrows():
            y_by_x[row[x]].append(row[y])

        fig, ax = plt.subplots()

        ax.violinplot(dataset=list(y_by_x.values()), showmedians=True)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_xticks(range(0, len(y_by_x)+1))
        ax.set_xticklabels([''] + list(y_by_x.keys()))
        if show:
            plt.show()

    def plot_3d_mesh(self, x, y, z, show=True, **filter_kwargs):
        """
        Plot interactive 3d mesh. z axis should typically be performance metric
        """
        _results = self.results
        fig = plt.figure()
        ax = Axes3D(fig)

        for k, v in filter_kwargs.items():
            _results = _results[getattr(_results, k) == v]

        X, Y, Z = [getattr(_results, attr) for attr in (x, y, z)]
        ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.2)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        if show:
            plt.show()

    def plot(self, *attrs: Tuple[str], show=True, 
        **filter_kwargs: Dict[str, Any]):
        """
        Attempt to intelligently dispatch plotting functions based on the number
        and type of attributes. Last argument should typically be the 
        performance metric.
        """
        self._assert_finished()
        param_names = self.param_names
        metric_names = self.metric_names

        if len(attrs) == 3:
            assert attrs[0] in param_names and attrs[1] in param_names, \
                'First two positional arguments must be parameter names.'

            assert attrs[2] in metric_names, \
                'Last positional argument must be a metric name.'

            assert len(filter_kwargs) + 2 == len(param_names), \
                'Must filter remaining parameters. e.g. p_three=some_number.'

            self.plot_3d_mesh(*attrs, show=show, **filter_kwargs)

        elif len(attrs) == 2:
            if len(param_names) == 1 or filter_kwargs:
                self.plot_2d_line(*attrs, show=show, **filter_kwargs)

            elif len(param_names) > 1:
                self.plot_2d_violin(*attrs, show=show)

        elif len(attrs) == 1:
            self.plot_1d_hist(*attrs, show=show)

        else:
            raise ValueError('Must pass between one and three column names.')
    
    def _make_id(self, sim_class) -> str:
        """generates unique identifier for bound simulator"""
        signal, pref = list(sim_class.params)
        _params = f"{signal}_{pref}"
        _date = datetime.datetime.now().strftime("%y%m%d")
        _kwarg = f"MAXPOS{sim_class.max_active_positions}"
        count = 0
        iterator = f"_{count}"
        while os.path.exists(f"StrategyTesting\\optimization_results\\{_params}_{_kwarg}_{_date}{iterator}.csv"):
            count += 1
            iterator = f"_{count}"
        return f"{_params}_{_kwarg}_{_date}{iterator}"

    @property
    def ID(self) -> str:
        return self._make_id(self.sim)

    def save_results(self):
        import os
        # saves the results of the grid search optomization to a CSV file 
        self.results.to_csv(f"StrategyTesting\\optimization_results\\{self.ID}.csv")


# Optimizer Usage
# if __name__ == '__main__':
#     # GridSearOptomizer example usage
#     from simulator_21LNG001 import BoundSimulators

#     simulate =  BoundSimulators(
#         Signals().create_bollinger_band_signal,
#         Signals().calculate_rolling_sharpe_ratio,
#         initial_cash=10000, max_active_positions=5
#     )
#     optimizer = GridSearchOptimizer(simulate.simulate_lookback_only)
#     optimizer.optimize(
#         signal_n=range(10, 15, 5),
#         performance_n=range(20, 30, 5),
#     )
#     optimizer.save_results()
    # optimizer.print_summary()
    # print(optimizer.get_best('excess_cagr'))
    # optimizer.save_results() 
    # optimizer.plot('excess_cagr')
    # optimizer.plot('bollinger_n', 'excess_cagr')
    # optimizer.plot('bollinger_n', 'sharpe_n', 'excess_cagr')


class OptimizationAnalysis():
    def __init__(self):
        self.results_path = os.path.join("StrategyTesting", "optimization_results")
        self.filename_df = self._load_filenames_df()
    
    def _load_filenames_df(self):
        _res = {
            "signal" : [],
            "preference" : [],
            "max_positions" : [],
            "date" : [],
            "iteration" : [],
            "id" : [],
            "filename" : []
        }
        for filename in os.listdir(self.results_path):
            _id = filename.replace(".csv", "")
            filename_list = _id.split("_")
            _res["signal"].append(filename_list[0])
            _res["preference"].append(filename_list[1])
            max_pos = int(filename_list[2].replace("MAXPOS", ""))
            _res["max_positions"].append(max_pos)
            date = datetime.datetime.strptime(filename_list[3], "%y%m%d")
            _res["date"].append(date)
            _res["iteration"].append(int(filename_list[4]))
            _res["id"].append(_id)
            _res["filename"].append(filename)
        return(pd.DataFrame(_res))
    
    def _results_by_signal(self, signal: str) -> pd.DataFrame:
        # searches the optimization results folder and filter by a given signal name
        # list of signal names given as class attribute
        _res = self.filename_df[self.filename_df.signal == signal]
        return _res
    
    def _results_by_preference(self, preference: str) -> pd.DataFrame:
        # searches the optimization results folder and filter by a given preference name
        # list of preference names given as class attribute
        _res = self.filename_df[self.filename_df.preference == preference]
        return _res
    
    def _results_by_max_positions(self, max_positions: str) -> pd.DataFrame:
        # searches the optimization results folder and filter by a given number of max positions
        _res = self.filename_df[self.filename_df.max_positions == max_positions]
        return _res
    
    def _results_by_date(self, start_date: str, end_date: str) -> pd.DataFrame:
        # searches the optimization results folder and filter by a given signal name
        # list of signal names given as class attribute
        _res = self.filename_df[self.filename_df.signal == signal]
        return _res
    
    def load_result(self, file_path: str) -> pd.DataFrame:
        pass
            
            


# OptimizationAnalysis usage
if __name__ == "__main__":
    results = OptimizationAnalysis()
    print(results.load_results_by_signal("BOLLIBANDSIGNA"))
