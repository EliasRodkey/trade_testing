#! python3
# graph_genreator.py takes data from the "optimization_results" and creates a series of graphs
# to analyze for feasible trading strategys

import matplotlib as plt
import os

from numpy.lib.npyio import save
from ToolKit import data_loading
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd


class Graph():
    def __init__(self, simset_ID: str):
        self.GRAPH_TYPES = ["histogram", "line", "violin", "mesh"]

        self.ID = simset_ID

        self.simset_folder_path = os.path.join(data_loading.RESULTS_PATH, simset_ID)
        self.simset_file_path = os.path.join(self.simset_folder_path, f"{simset_ID}.csv")
        self.dir_paths = {}

        self.df = pd.read_csv(self.simset_file_path)
        
    @property
    def controlled_variables(self) -> list:
        # returns the controlled variables column headers for the given simset
        ctrl_var_list = ["max_positions"]
        columns = list(self.df.columns)
        maxpos_idx = columns.index("max_positions")
        lower_bound_idx = columns.index("date")
        upper_bound_idx = columns.index("percent_return")
        for header in columns[lower_bound_idx + 1:upper_bound_idx]:
            ctrl_var_list.append(header)
        return ctrl_var_list
    
    @property
    def metrics(self) -> list:
        # returns a list of the names of the metrics used in the simset
        columns = list(self.df.columns)
        lower_bound_idx = columns.index("percent_return")
        return columns[lower_bound_idx:]

    def create_graph_directory(self, graph_type: str) -> str:
        # creates a direcotry to contain a series of graph images
        if graph_type not in self.GRAPH_TYPES:
            print("invalid graph type provided, directory not created")
            return
        dir_path = os.path.join(self.simset_folder_path, f"{graph_type}_graphs")
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
            print(f"directory created: {dir_path}")
        self.dir_paths[graph_type] = dir_path
        return dir_path
    
    def create_1d_hist(self, metric: str, save=False, show=False):
        # generates a single 1 dimensional histogram with distributed by ferquency
        plt_inst = plt
        plt_inst.hist(self.df[metric])
        plt_inst.title(f"{self.ID}:\n {metric} Histogram")
        plt_inst.xlabel(metric)
        plt_inst.ylabel("frequency")
        if show:
            plt_inst.show()
        if save:
            self.create_graph_directory("histogram")
            plt_inst.savefig(os.path.join(self.dir_paths["histogram"], f"{metric}_hisotgram.png"))
            print(f"histogram ID: {self.ID}, metric: {metric}, SAVED\n")
    
    def generate_hists(self):
        # generates a series of histograms, 1 for each metric
        self.create_graph_directory("histogram")
        for metric in self.metrics[:5]:
            print(f"generating frequency histogram for {self.ID} {metric}")
            self.create_1d_hist(metric, save=True)


if __name__ == "__main__":
    simset_ID = "BOLLIBANDSIGNA_ROLLISHARPRATIO_210516"
    grph = Graph(simset_ID)
    grph.generate_hists()

#NOTE: For some reason when the histograms are shown they look fine but when they are saved 
# they save over eachtother in different colors making them unreadable
#TODO: make a to omitt list so single value graphs are not generated
#IDEA: for optimization, I dont have to calculate CAGR / return 
# for SPY every time since its the same but that probably saves like no time
#TODO: make all graphs for all controlled variables
#TODO: Add graphs to a report template
#TODO: Profit

# def plot_1d_hist(x, save=False, show=False):
#     plt.hist(x)
#     if show:
#         plt.show()

# def plot_2d_line(self, x, y, show=True, **filter_kwargs):
#     _results = self.results
#     for k, v in filter_kwargs.items():
#         _results = _results[getattr(_results, k) == v]

#     ax = _results.plot(x, y)
#     if filter_kwargs:
#         k_str = ', '.join([f'{k}={v}' for k,v in filter_kwargs.items()])
#         ax.legend([f'{x} ({k_str})'])

#     if show:
#         plt.show()

# def plot_2d_violin(self, x, y, show=True):
#     """
#     Group y along x then plot violin charts
#     """
#     x_values = self.results[x].unique()
#     x_values.sort()

#     y_by_x = OrderedDict([(v, []) for v in x_values])
#     for _, row in self.results.iterrows():
#         y_by_x[row[x]].append(row[y])

#     fig, ax = plt.subplots()

#     ax.violinplot(dataset=list(y_by_x.values()), showmedians=True)
#     ax.set_xlabel(x)
#     ax.set_ylabel(y)
#     ax.set_xticks(range(0, len(y_by_x)+1))
#     ax.set_xticklabels([''] + list(y_by_x.keys()))
#     if show:
#         plt.show()

# def plot_3d_mesh(self, x, y, z, show=True, **filter_kwargs):
#     """
#     Plot interactive 3d mesh. z axis should typically be performance metric
#     """
#     _results = self.results
#     fig = plt.figure()
#     ax = Axes3D(fig)

#     for k, v in filter_kwargs.items():
#         _results = _results[getattr(_results, k) == v]

#     X, Y, Z = [getattr(_results, attr) for attr in (x, y, z)]
#     ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.2)
#     ax.set_xlabel(x)
#     ax.set_ylabel(y)
#     ax.set_zlabel(z)
#     if show:
#         plt.show()

# def plot(self, *attrs: Tuple[str], show=True, 
#     **filter_kwargs: Dict[str, Any]):
#     """
#     Attempt to intelligently dispatch plotting functions based on the number
#     and type of attributes. Last argument should typically be the 
#     performance metric.
#     """
#     self._assert_finished()
#     param_names = self.param_names
#     metric_names = self.metric_names

#     if len(attrs) == 3:
#         assert attrs[0] in param_names and attrs[1] in param_names, \
#             'First two positional arguments must be parameter names.'

#         assert attrs[2] in metric_names, \
#             'Last positional argument must be a metric name.'

#         assert len(filter_kwargs) + 2 == len(param_names), \
#             'Must filter remaining parameters. e.g. p_three=some_number.'

#         self.plot_3d_mesh(*attrs, show=show, **filter_kwargs)

#     elif len(attrs) == 2:
#         if len(param_names) == 1 or filter_kwargs:
#             self.plot_2d_line(*attrs, show=show, **filter_kwargs)

#         elif len(param_names) > 1:
#             self.plot_2d_violin(*attrs, show=show)

#     elif len(attrs) == 1:
#         self.plot_1d_hist(*attrs, show=show)

#     else:
#         raise ValueError('Must pass between one and three column names.')
