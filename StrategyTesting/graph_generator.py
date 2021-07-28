#! python3
# graph_genreator.py takes data from the "optimization_results" and creates a series of graphs
# to analyze for feasible trading strategys

import matplotlib as plt
from ToolKit import data_loading
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D 


class grapher():
    def __init__(self, results_path: str):
        pass


def plot_1d_hist(x, save=False, show=False):
    plt.hist(x)
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
