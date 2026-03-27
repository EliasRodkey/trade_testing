#! python3
"""
Visualization module for algorithmic trading optimization results.

Reads simulation result CSVs from the optimization_results directory and
generates plots to help identify effective trading strategies.

Classes:
    Graph: Loads a simset CSV and exposes methods to generate histograms,
           line charts, violin plots, and interactive 3D surface plots.
"""

import os
from typing import Tuple, Dict, Any

from ToolKit import data_loading
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


class Graph():
    """
    Loads optimization result data for a single simset and provides plotting methods.

    Attributes:
        GRAPH_TYPES (list[str]): Supported graph type identifiers used as subdirectory names.
        ID (str): The simset identifier string (e.g. 'BOLLIBANDSIGNA_ROLLISHARPRATIO_210516').
        simset_folder_path (str): Path to the simset's result folder.
        simset_file_path (str): Path to the simset's result CSV file.
        dir_paths (dict[str, str]): Maps graph type to its output directory path, populated on first use.
        df (pd.DataFrame): The loaded simset results DataFrame.

    Properties:
        controlled_variables: Column names for the parameters varied during optimization.
        metrics: Column names for all performance metrics (from percent_return onwards).

    Methods:
        create_graph_directory: Creates and registers an output directory for a graph type.
        create_1d_hist: Generates and optionally saves/shows a histogram for one metric.
        generate_hists: Batch-generates and saves histograms for all metrics.
        plot_2d_line: Line chart of y vs x with optional equality filters.
        plot_2d_violin: Violin chart grouping y values by unique x values.
        plot_3d_mesh: Interactive 3D surface plot of z over (x, y).
        plot: Smart dispatcher — selects chart type based on number of column args.
    """

    def __init__(self, simset_ID: str):
        """
        Args:
            simset_ID (str): The folder and file name of the simset results
                             (e.g. 'BOLLIBANDSIGNA_ROLLISHARPRATIO_210516').
        """
        self.GRAPH_TYPES = ["histogram", "line", "violin", "mesh"]

        self.ID = simset_ID

        self.simset_folder_path = os.path.join(data_loading.RESULTS_PATH, simset_ID)
        self.simset_file_path = os.path.join(self.simset_folder_path, f"{simset_ID}.csv")
        self.dir_paths = {}

        self.df = pd.read_csv(self.simset_file_path)

    @property
    def controlled_variables(self) -> list:
        """
        Column names for the parameters that were varied during optimization.

        Includes max_positions plus any parameter columns that appear between
        the 'date' and 'percent_return' columns in the CSV.

        Returns:
            list[str]: Ordered list of controlled variable column names.
        """
        ctrl_var_list = ["max_positions"]
        columns = list(self.df.columns)
        lower_bound_idx = columns.index("date")
        upper_bound_idx = columns.index("percent_return")
        for header in columns[lower_bound_idx + 1:upper_bound_idx]:
            ctrl_var_list.append(header)
        return ctrl_var_list

    @property
    def metrics(self) -> list:
        """
        Column names for all performance metrics in the simset results.

        Returns:
            list[str]: Column names from 'percent_return' to the end of the DataFrame.
        """
        columns = list(self.df.columns)
        lower_bound_idx = columns.index("percent_return")
        return columns[lower_bound_idx:]

    def create_graph_directory(self, graph_type: str) -> str:
        """
        Creates a subdirectory for storing a specific type of graph image.

        Registers the path in self.dir_paths[graph_type] for later use.
        Does nothing if the directory already exists.

        Args:
            graph_type (str): The type of graph. Must be one of self.GRAPH_TYPES.

        Returns:
            str: Absolute path to the graph output directory.

        Raises:
            ValueError: If graph_type is not in self.GRAPH_TYPES.
        """
        if graph_type not in self.GRAPH_TYPES:
            raise ValueError(f"Invalid graph type '{graph_type}'. Must be one of: {self.GRAPH_TYPES}")
        dir_path = os.path.join(self.simset_folder_path, f"{graph_type}_graphs")
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
            print(f"directory created: {dir_path}")
        self.dir_paths[graph_type] = dir_path
        return dir_path

    def create_1d_hist(self, metric: str, save: bool = False, show: bool = False) -> None:
        """
        Generates a 1D frequency histogram for a single metric column.

        Args:
            metric (str): Column name from self.df to plot.
            save (bool): If True, saves the figure as a PNG to the histogram graph directory.
            show (bool): If True, displays the figure interactively.

        Returns:
            None
        """
        plt_inst = plt
        plt_inst.hist(self.df[metric])
        plt_inst.title(f"{self.ID}:\n {metric} Histogram")
        plt_inst.xlabel(metric)
        plt_inst.ylabel("frequency")
        if show:
            plt_inst.show()
        if save:
            self.create_graph_directory("histogram")
            plt_inst.savefig(os.path.join(self.dir_paths["histogram"], f"{metric}_histogram.png"))
            print(f"histogram ID: {self.ID}, metric: {metric}, SAVED\n")
            plt_inst.clf()

    def generate_hists(self) -> None:
        """
        Generates and saves a frequency histogram for every metric in the simset.

        Calls create_1d_hist(metric, save=True) for each column in self.metrics.

        Returns:
            None
        """
        self.create_graph_directory("histogram")
        for metric in self.metrics:
            print(f"generating frequency histogram for {self.ID} {metric}")
            self.create_1d_hist(metric, save=True)

    def plot_2d_line(self, x: str, y: str, show: bool = True, **filter_kwargs) -> None:
        """
        Plots a 2D line chart of y vs x, with optional equality filters on the dataset.

        Args:
            x (str): Column name to use as the x-axis.
            y (str): Column name to use as the y-axis.
            show (bool): If True, displays the figure interactively.
            **filter_kwargs: Column equality filters applied before plotting
                             (e.g. max_positions=10 keeps only rows where max_positions == 10).

        Returns:
            None
        """
        _results = self.df
        for k, v in filter_kwargs.items():
            _results = _results[_results[k] == v]
        ax = _results.plot(x, y)
        if filter_kwargs:
            k_str = ', '.join([f'{k}={v}' for k, v in filter_kwargs.items()])
            ax.legend([f'{x} ({k_str})'])
        if show:
            plt.show()

    def plot_2d_violin(self, x: str, y: str, show: bool = True) -> None:
        """
        Plots a violin chart showing the distribution of y grouped by each unique value of x.

        Args:
            x (str): Column name to group by (x-axis categories).
            y (str): Column name whose values are distributed across groups (y-axis).
            show (bool): If True, displays the figure interactively.

        Returns:
            None
        """
        groups = self.df.groupby(x)[y].apply(list)
        fig, ax = plt.subplots()
        ax.violinplot(dataset=list(groups.values), showmedians=True)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_xticks(range(0, len(groups) + 1))
        ax.set_xticklabels([''] + list(groups.index))
        if show:
            plt.show()

    def plot_3d_mesh(self, x: str, y: str, z: str, show: bool = True, **filter_kwargs) -> None:
        """
        Plots an interactive 3D triangulated surface of z over the (x, y) parameter space.

        Args:
            x (str): Column name for the x-axis parameter.
            y (str): Column name for the y-axis parameter.
            z (str): Column name for the z-axis value (typically a performance metric).
            show (bool): If True, displays the figure interactively.
            **filter_kwargs: Column equality filters applied before plotting.

        Returns:
            None
        """
        _results = self.df
        for k, v in filter_kwargs.items():
            _results = _results[_results[k] == v]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y, Z = [_results[attr] for attr in (x, y, z)]
        ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.2)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        if show:
            plt.show()

    def plot(self, *attrs: Tuple[str], show: bool = True, **filter_kwargs: Dict[str, Any]) -> None:
        """
        Dispatches to the appropriate plot method based on the number of column arguments.

        - 1 attr: histogram of that column via create_1d_hist
        - 2 attrs: line chart (if 1 controlled variable or filters given) or violin chart
        - 3 attrs: 3D surface mesh (first two must be parameter names, last must be a metric)

        Args:
            *attrs (str): Column names to plot. For 3-attr calls, the last must be a metric
                          name and the first two must be parameter names.
            show (bool): If True, displays the figure interactively.
            **filter_kwargs: Column equality filters passed through to the underlying plot method.

        Returns:
            None

        Raises:
            AssertionError: If 3 attrs are given and they do not satisfy the parameter/metric constraints.
            ValueError: If fewer than 1 or more than 3 attrs are provided.
        """
        param_names = self.controlled_variables
        metric_names = self.metrics

        if len(attrs) == 3:
            assert attrs[0] in param_names and attrs[1] in param_names, \
                'First two positional arguments must be parameter names.'
            assert attrs[2] in metric_names, \
                'Last positional argument must be a metric name.'
            self.plot_3d_mesh(*attrs, show=show, **filter_kwargs)

        elif len(attrs) == 2:
            if len(param_names) == 1 or filter_kwargs:
                self.plot_2d_line(*attrs, show=show, **filter_kwargs)
            elif len(param_names) > 1:
                self.plot_2d_violin(*attrs, show=show)

        elif len(attrs) == 1:
            self.create_1d_hist(*attrs, show=show)

        else:
            raise ValueError('Must pass between one and three column names.')


if __name__ == "__main__":
    simset_ID = "BOLLIBANDSIGNA_ROLLISHARPRATIO_210516"
    grph = Graph(simset_ID)
    # grph.generate_hists()
    grph.plot_2d_line("signal_n", "edge_score")
    grph.plot_2d_violin("max_positions", "sharpe_ratio")
    grph.plot_3d_mesh("signal_n", "preference_n", "edge_score")
    grph.plot("signal_n", "edge_score")
