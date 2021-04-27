#! python3
# 
# optimizer_001.py - first simulation optimizer.
# GridSearOptomize given input of a simulation funciton. the .optomize function
# is called with a dicitonary of iterables which are tried in all combinations
# by the optimizer which then returns data on the most successful parameters

import datetime
from ToolKit.signal_generator import Signals 
from ToolKit.data_loading import load_eod_matrix, get_all_symbols
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
import inspect
from itertools import product
from timeit import default_timer
from reprint import output
from typing import Dict, Tuple, List, Callable, Iterable, Any, NewType, Mapping
import os
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
        print(f'Simulating: 1 / {total_simulations}', end='\r')
        with output(output_type='dict', interval=1) as output_lines:
            returns = []
            wins = []
            trades = []
            for i, params in enumerate(product(*param_ranges.values())):
                timer_start = default_timer()
                
                if i > 0:
                    s = f"""
                        Simulating {i+1} / {total_simulations}
                        Expected Time Remaining : {round((n - (i + 1)) * self.time_df.total_time.mean(), 0)}s

                        Setup Time : {round(self.time_df.setup_time.mean(), 4)}s
                        Calculation Time : {round(self.time_df.calculation_time.mean(), 4)}s
                        Transaction Time : {round(self.time_df.transaction_time.mean(), 4)}s
                        Sim Time : {round(self.time_df.total_sim_time.mean(), 2)}s
                        Analysis Time : {round(self.time_df.finish_time.mean(), 4)}s
                        Recording Time : {round(self.time_df.record_results_time.mean(), 2)}s
                        Total Time : {round(self.time_df.total_time.mean(), 2)}s

                        Avg Simulated Return : {round(100*np.average(returns), 2)}%
                        Avg Simulated Win Percent : {100*round(np.average(wins), 2)}%
                        Avg Number of Trades : {round(np.average(trades), 2)}
                    """
                    print(s, end='\r')
                    # output_lines['Simulating'] = f"{i+1} / {total_simulations}"
                    # output_lines['Expected Time Remaining'] = f"""{
                    #     round((n - (i + 1)) * self.time_df.total_time.mean(), 0)
                    # }s\n"""
                    # output_lines['Setup Time'] = f"{round(self.time_df.setup_time.mean(), 4)}s"
                    # output_lines['Calculation Time'] = f"{round(self.time_df.calculation_time.mean(), 4)}"
                    # output_lines['Transaction Time'] = f"{round(self.time_df.transaction_time.mean(), 4)}s"
                    # output_lines['Analysis Time'] = f"{round(self.time_df.finish_time.mean(), 4)}s"
                    # output_lines['Total Sim Time'] = f"{round(self.time_df.total_sim_time.mean(), 2)}s"
                    # output_lines['Recording Time'] = f"{round(self.time_df.record_results_time.mean(), 2)}s"
                    # output_lines['Total Time'] = f"{round(self.time_df.total_time.mean(), 2)}s"
                    # output_lines['Simulated Return'] = f"{round(100*np.average(returns), 2)}%"
                    # output_lines['Simulated Win Percent'] = f"{round(np.average(wins), 2)}%"
                    # output_lines['Number of Trades'] = f"{round(np.average(trades), 2)}"
                parameters = {n: param for n, param in zip(param_names, params)}
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
                sim.time_data["total_time"] = timer_end - timer_start
                sim.time_data["total_time_elapsed"] = total_time_elapsed
                sim.time_data["record_results_time"] = timer_end - timer_mid
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
        args = list(sim_class.params)
        for i, arg in enumerate(sim_class.params):
            arg.replace(" ", "")
            arg.replace("_", "")
            if len(arg) > 6:
                args[i] = arg[:6]
        _params = "".join(args).upper()
        _date = datetime.datetime.now().strftime("%d%m%Y")
        count = 0
        _kwarg = f"_MAXPOS{self.max_positions}"
        iterator = f"_{count}"
        while os.path.exists(f"optimization_results\\{_params}_{_kwarg}_{_date}{iterator}.csv"):
            count += 1
            iterator = f"_{count}"
        return f"{_params}_{_kwarg}_{_date}{iterator}"

    @property
    def ID(self) -> str:
        return self._make_id(self.sim)

    def save_results(self):
        import os
        # saves the results of the grid search optomization to a CSV file 
        self.results.to_csv(f"optimization_results\\{self.ID}.csv")



# Optimizer Usage
if __name__ == '__main__':
    # GridSearOptomizer example usage
    from simulator_21LNG001 import BoundSimulators

    simulate =  BoundSimulators(
        Signals().create_bollinger_band_signal,
        Signals().calculate_rolling_sharpe_ratio,
        initial_cash=10000, max_active_positions=5
    )
    optimizer = GridSearchOptimizer(simulate.simulate_lookback_only)
    optimizer.optimize(
        signal_n=range(10, 15, 1),
        performance_n=range(20, 30, 2),
    )
    optimizer.save_results()
    optimizer.print_summary()
    # print(optimizer.get_best('excess_cagr'))
    # optimizer.save_results() 
    # optimizer.plot('excess_cagr')
    # optimizer.plot('bollinger_n', 'excess_cagr')
    # optimizer.plot('bollinger_n', 'sharpe_n', 'excess_cagr')


class OptimizationAnalysis():
    def __init__(self):
        pass