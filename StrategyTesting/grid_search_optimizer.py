"""
grid_search_optimizer.py - Grid search optimization engine for backtesting strategy parameter tuning.

Classes:
    OptimizationResult: Container for a single simulation's parameters, metadata, and performance.
    GridSearchOptimizer: Iterates over all combinations of signal and preference parameter ranges,
                         collects results, and saves them to CSV.

Variables:
    SimKwargs: NewType alias for a mapping of parameter names to iterables of values.
"""

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
    """
    Simple container for a single simulation run's parameters, performance metrics, and metadata.

    Combines all three into a single row DataFrame via the combined property for easy aggregation.
    """

    def __init__(self, parameters: Dict[str, int], performance: pd.DataFrame, metadata: Dict):
        """
        Initializes an OptimizationResult and validates no name collisions exist.

        Args:
            parameters (Dict[str, int]): Parameter name-value pairs used in this simulation run.
            performance (pd.DataFrame): Single-row DataFrame of performance metrics from PortfolioHistory.
            metadata (Dict): Simulation metadata (e.g., signal type, date, max positions).

        Raises:
            AssertionError: If any key in parameters also appears in performance.
        """
        # Make sure no collisions between performance metrics and params
        assert len(parameters.keys() & performance.keys()) == 0, \
            'parameter name matches performance metric name'

        self.parameters = self.as_pd(parameters)
        self.performance = performance
        self.metadata = self.as_pd(metadata)

    @staticmethod
    def as_pd(params: Dict[str, int]) -> pd.DataFrame:
        """
        Wraps a dictionary as a single-row DataFrame.

        Args:
            params (Dict[str, int]): Key-value pairs to convert.

        Returns:
            pd.DataFrame: Single-row DataFrame with dict keys as columns.
        """
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
        """
        Initializes the optimizer with a simulation function and output settings.

        Args:
            simulation_function (Callable): A callable matching the BoundSimulators.simulate
                                            signature: (signal_args, preference_args) -> (sim, results).
            save_results (bool): If True, saves the results DataFrame to CSV after optimize() completes.
                                 Defaults to True.
        """
        self.simulate = simulation_function
        self._results_list: List[OptimizationResult] = list()
        self.time_df = pd.DataFrame()
        self.save = save_results
        # self.ui = Ui_MainWindow()
        # self.ui.show_ui()

        self._optimization_finished = False

    def add_results(self, parameters: Dict[str, int], performance: pd.DataFrame, metadata: Dict) -> None:
        """
        Wraps parameters, performance, and metadata into an OptimizationResult and appends it.

        Args:
            parameters (Dict[str, int]): Parameter name-value pairs for this simulation run.
            performance (pd.DataFrame): Performance metric DataFrame from PortfolioHistory.
            metadata (Dict): Simulation metadata (signal type, date, max positions, etc.).
        """
        _results = OptimizationResult(parameters, performance, metadata)
        self._results_list.append(_results.combined)
    
    def _add_to_time_df(self, times_to_add: pd.DataFrame) -> None:
        """
        Appends a timing row to the internal time tracking DataFrame.

        Args:
            times_to_add (pd.DataFrame): Single-row DataFrame with 'total_time' and
                                         'total_time_elapsed' columns.
        """
        if self.time_df.empty:
            self.time_df = pd.DataFrame(columns=times_to_add.columns)
        _to_concat = [self.time_df, times_to_add]
        self.time_df = pd.concat(_to_concat, axis=0).reset_index(drop=True)

    def optimize(self, signal_ranges: dict, preference_ranges: dict) -> None:
        """
        Runs the full grid search over all combinations of signal and preference parameters.

        Iterates the Cartesian product of all provided ranges, calls the simulation function
        for each combination, collects results, and (if save=True) writes them to CSV.
        Prints a progress log after each simulation beyond the first.

        Args:
            signal_ranges (dict): Mapping of signal parameter names to iterables of values.
            preference_ranges (dict): Mapping of preference parameter names to iterables of values.

        Raises:
            AssertionError: If either signal_ranges or preference_ranges is empty.
        """
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

    def _assert_finished(self) -> None:
        """
        Asserts that optimize() has been called before accessing results-dependent methods.

        Raises:
            AssertionError: If optimize() has not yet completed.
        """
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

    def print_summary(self) -> None:
        """Prints descriptive statistics for all performance metrics across all simulation runs."""
        df = self.results
        metric_names = self.metric_names

        print('Summary statistics')
        print(df[metric_names].describe().T)

    def make_metadata_dict(self, sim_class: object) -> None:
        """
        Populates self._metadata with identifying information about the simulation run.

        Stores signal/preference IDs, max_positions, and the current date. Called once
        per simulation run inside optimize().

        Args:
            sim_class (object): A BoundSimulators instance with a params attribute and
                                max_active_positions attribute.
        """
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
    def winning_sim_ratio(self) -> float:
        """Fraction of simulations whose percent_return exceeded the SPY benchmark return."""
        simset = self.results
        filt = simset["percent_return"] > simset["spy_percent_return"]
        number_winning = simset[filt].shape[0]
        ratio = number_winning / simset.shape[0]
        return ratio

    def compile_simset(self) -> pd.DataFrame:
        """
        Returns a single-row DataFrame summarizing the simulation set.

        Includes mean values for all performance metrics, edge score standard deviation,
        total simulation count, percent of simulations beating SPY, and the simset ID.

        Returns:
            pd.DataFrame: Single-row summary DataFrame.
        """
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
        """Single-row summary DataFrame for this simulation set. See compile_simset."""
        return self.compile_simset()

    def add_to_compiled_results(self) -> None:
        """
        Appends this simulation set's compiled summary row to the shared compiled_results.csv.

        Creates the file if it does not exist. Reads the existing file, concatenates the new
        row, and overwrites.
        """
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

    def save_results(self) -> None:
        """
        Saves the full results DataFrame to a CSV file in the optimization_results directory.

        Creates a subdirectory named after the simulation set ID if it does not already exist,
        then writes the results DataFrame to a CSV file within it.
        """
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