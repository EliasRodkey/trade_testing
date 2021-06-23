#! python3
# 
# tool_kit.py contains functions associated with calculating
# metrics and indicators of a set of chronical financial price data
# as well as classes used in creating a financial market simulation 
# to test trading strategies (also called backtesting) and evaluate
# performance 

from typing import Dict, NewType, Any, List, Set
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ToolKit.signal_generator import Metrics
from ToolKit.data_loading import load_SPY_data, load_data_as_pd


# Position contains the class
# which monitors and records raw data about
# a single transactions during the simulation
#
# data recorded:
# entry and exit date
# entry and exit price
# equity value on each day during the trade
# percent change
# $ increase/decrease
# summary display

Symbol = NewType('Symbol', str)
Dollars = NewType('Dollars', float)

DATE_FORMAT_STR = '%a %b %d, %Y'
def _pdate(date: pd.Timestamp):
    """Pretty-print a datetime with just the date"""
    return date.strftime(DATE_FORMAT_STR)


class Position(object):
    """
    A simple object to hold and manipulate data related to long stock trades.

    Allows a single buy and sell operation on an asset for a constant number of 
    shares.

    The __init__ method is equivelant to a buy operation. The exit
    method is a sell operation.
    """

    def __init__(self, symbol: Symbol, entry_date: pd.Timestamp, 
        entry_price: Dollars, shares: int):
        """
        Equivelent to buying a certain number of shares of the asset
        """

        # Recorded on initialization
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.symbol = symbol

        # Recorded on position exit
        self.exit_date: pd.Timestamp = None
        self.exit_price: Dollars = None

        # For easily getting current portolio value
        self.last_date: pd.Timestamp = None
        self.last_price: Dollars = None

        # Updated intermediately
        self._dict_series: Dict[pd.Timestamp, Dollars] = OrderedDict()
        self.record_price_update(entry_date, entry_price)

        # Cache control for pd.Series representation
        self._price_series: pd.Series = None
        self._needs_update_pd_series: bool = True

    def exit(self, exit_date, exit_price):
        """
        Equivelent to selling a stock holding
        """
        assert self.entry_date != exit_date, 'Churned a position same-day.'
        assert not self.exit_date, 'Position already closed.'
        self.record_price_update(exit_date, exit_price)
        self.exit_date = exit_date
        self.exit_price = exit_price

    def record_price_update(self, date, price):
        """
        Stateless function to record intermediate prices of existing positions
        """
        self.last_date = date
        self.last_price = price
        self._dict_series[date] = price

        # Invalidate cache on self.price_series
        self._needs_update_pd_series = True

    @property
    def price_series(self) -> pd.Series:
        """
        Returns cached readonly pd.Series 
        """
        if self._needs_update_pd_series or self._price_series is None:
            self._price_series = pd.Series(self._dict_series)
            self._needs_update_pd_series = False
        return self._price_series

    @property
    def last_value(self) -> Dollars:
        return self.last_price * self.shares

    @property
    def is_active(self) -> bool:
        return self.exit_date is None

    @property
    def is_closed(self) -> bool:
        return not self.is_active
    
    @property
    def value_series(self) -> pd.Series:
        """
        Returns the value of the position over time. Ignores self.exit_date.
        Used in calculating the equity curve.
        """
        assert self.is_closed, 'Position must be closed to access this property'
        return self.shares * self.price_series[:-1]

    @property
    def percent_return(self) -> float:
        return (self.exit_price / self.entry_price) - 1
    
    @property
    def entry_value(self) -> Dollars:
        return self.shares * self.entry_price

    @property
    def exit_value(self) -> Dollars:
        return self.shares * self.exit_price

    @property
    def change_in_value(self) -> Dollars:
        return self.exit_value - self.entry_value
    
    @property
    def change_in_price(self) -> Dollars:
        return self.exit_price - self.entry_price

    @property
    def trade_length(self):
        return len(self._dict_series) - 1
    
    @property
    def trade_summary(self):
        to_df = {
            "symbol": [self.symbol],
            "entry_date": [self.entry_date],
            "exit_date" : [self.exit_date],
            "trade_length": [self.trade_length],
            "entry_price": [self.entry_price],
            "exit_price": [self.exit_price],
            "price_change": [self.change_in_price],
            "entry_vlaue": [self.entry_value],
            "exit_value": [self.exit_value],
            "value_change": [self.change_in_value],
            "percent_return": [self.percent_return],
            "hash": [self.__hash__()]
        }
        return pd.DataFrame(to_df)
    
    def print_position_summary(self):
        _entry_date = _pdate(self.entry_date)
        _exit_date = _pdate(self.exit_date)
        _days = self.trade_length

        _entry_price = round(self.entry_price, 2)
        _exit_price = round(self.exit_price, 2)

        _entry_value = round(self.entry_value, 2)
        _exit_value = round(self.exit_value, 2)

        _return = round(100 * self.percent_return, 1)
        _diff = round(self.change_in_value, 2)

        print(f'{self.symbol:<5}     Trade summary')
        print(f'Date:     {_entry_date} -> {_exit_date} [{_days} days]')
        print(f'Price:    ${_entry_price} -> ${_exit_price} [{_return}%]')
        print(f'Value:    ${_entry_value} -> ${_exit_value} [${_diff}]\n')
        print()

    def __hash__(self):
        """
        A unique position will be defined by a unique combination of an 
        entry_date and symbol, in accordance with our constraints regarding 
        duplicate, variable, and compound positions
        """
        return hash((self.entry_date, self.symbol))


# PortfolioHistory supports fully finished position objects being added to it
# once the .finish() method is called it uses position data to calculate an
# equity curve and a series of metrics associated with the data
# using the metrics class.


class PortfolioHistory(object):
    """
    Holds Position objects and keeps track of portfolio variables.
    Produces summary statistics.
    """

    def __init__(self):
        # initialize metrics instance
        self.metrics = Metrics()
        # Keep track of positions, recorded in this list after close
        self.position_history: List[Position] = []
        self._logged_positions: Set[Position] = set()

        # Keep track of the last seen date
        self.last_date: pd.Timestamp = pd.Timestamp.min

        # Readonly fields
        self._cash_history: Dict[pd.Timestamp, Dollars] = dict()
        self._simulation_finished = False
        self._spy: pd.DataFrame = pd.DataFrame()
        self._spy_log_returns: pd.Series = pd.Series(dtype=float)

    def add_to_history(self, position: Position):
        _log = self._logged_positions
        assert not position in _log, 'Recorded the same position twice.'
        assert position.is_closed, 'Position is not closed.'
        self._logged_positions.add(position)
        self.position_history.append(position)
        self.last_date = max(self.last_date, position.last_date)

    def record_cash(self, date, cash):
        self._cash_history[date] = cash
        self.last_date = max(self.last_date, date)

    @staticmethod
    def _as_oseries(d: Dict[pd.Timestamp, Any]) -> pd.Series:
        return pd.Series(d).sort_index()

    def _generate_trade_summary_df(self) -> pd.DataFrame:
        self._assert_finished()
        summaries = []
        for position in self.position_history:
            summaries.append(position.trade_summary)
        try:
            self.trade_summary_df = pd.concat(summaries, axis=0).reset_index().drop(columns=["index"])
        except ValueError:
            cols = [
                "trade_length", "price_change", 
                "value_change", "percent_return"
            ]
            self.trade_summary_df = pd.DataFrame(columns=cols)
            
    def _compute_cash_series(self):
        self._cash_series = self._as_oseries(self._cash_history)

    @property
    def cash_series(self) -> pd.Series:
        return self._cash_series

    def _compute_portfolio_value_series(self):
        value_by_date = defaultdict(float)
        last_date = self.last_date

        # Add up value of assets
        for position in self.position_history:
            for date, value in position.value_series.items():
                value_by_date[date] += value

        # Make sure all dates in cash_series are present
        for date in self.cash_series.index:
            value_by_date[date] += 0
        
        self._portfolio_value_series = self._as_oseries(value_by_date)

    @property
    def portfolio_value_series(self):
        return self._portfolio_value_series

    def _compute_equity_series(self):
        c_series = self.cash_series
        p_series = self.portfolio_value_series
        assert all(c_series.index == p_series.index), \
            'portfolio_series has dates not in cash_series'
        raw_series = c_series + p_series
        self._equity_series = raw_series[raw_series > 0]

    @property
    def equity_series(self):
        return self._equity_series
    
    @property
    def days_traded(self):
        return self.equity_series.shape[0]

    def _compute_log_return_series(self):
        self._log_return_series = \
            Metrics.calculate_log_return_series(self.equity_series)

    @property
    def log_return_series(self):
        return self._log_return_series

    def _assert_finished(self):
        assert self._simulation_finished, \
            'Simuation must be finished by running self.finish() in order ' + \
            'to access this method or property.'

    def finish(self):
        """
        Notate that the simulation is finished and compute readonly values
        """
        self._simulation_finished = True
        self._generate_trade_summary_df()
        self._compute_cash_series()
        self.cash_series.name = "cash series"
        self._compute_portfolio_value_series()
        self.portfolio_value_series.name = "porfolio return series"
        self._compute_equity_series()
        self.equity_series.name = "equity series"
        self._compute_log_return_series()
        self.log_return_series.name = "log return series"
        self._assert_finished()

    def compute_portfolio_size_series(self) -> pd.Series:
        size_by_date = defaultdict(int)
        for position in self.position_history:
            for date in position.value_series.index:
                size_by_date[date] += 1
        return self._as_oseries(size_by_date)

    @property
    def spy(self) -> pd.DataFrame:
        if self._spy.empty:
            self._spy = load_SPY_data()
        return self._spy

    @property
    def spy_log_returns(self) -> pd.Series:
        if self._spy_log_returns.empty:
            close = self.spy['close']
            self._spy_log_returns =  self.metrics.calculate_log_return_series(close)
        self._spy_log_returns.name = "spy log returns"
        return self._spy_log_returns

    @property
    def percent_return(self) -> float:
        return self.metrics.calculate_percent_return(self.equity_series)

    @property
    def spy_percent_return(self) -> float:
        return self.metrics.calculate_percent_return(self.spy['close'])

    @property
    def cagr(self) -> float:
        return self.metrics.calculate_cagr(self.equity_series)

    @property
    def volatility(self) -> float:
        return self.metrics.calculate_annualized_volatility(self.log_return_series)

    @property
    def sharpe_ratio(self) -> float:
        return self.metrics.calculate_sharpe_ratio(self.equity_series)

    @property
    def sortino_ratio(self) -> float:
        return self.metrics.calculate_sortino_ratio(self.equity_series)

    @property
    def calmar_ratio(self) -> float:
        return self.metrics.calculate_calmar_ratio(self.equity_series)
    
    @property
    def pure_profit_score(self) -> float:
        return self.metrics.calculate_pure_profit_score(self.equity_series)

    @property
    def spy_cagr(self) -> float:
        return self.metrics.calculate_cagr(self.spy['close'])
    
    @property
    def excess_cagr(self) -> float:
        return self.cagr - self.spy_cagr

    @property
    def jensens_alpha(self) -> float:
        return self.metrics.calculate_jensens_alpha(
            self.log_return_series,
            self.spy_log_returns,
        )

    @property
    def alpha(self) -> float:
        return self.metrics.calculate_alpha(
            self.log_return_series, 
            self.spy_log_returns
        )

    @property
    def beta(self) -> float:
        return self.metrics.calculate_beta(
            self.log_return_series, 
            self.spy_log_returns
        )
    
    @property
    def r_squared(self) -> float:
        return self.metrics.calculate_r_squared(
            self.log_return_series, 
            self.spy_log_returns
        )

    @property
    def dollar_max_drawdown(self):
        return self.metrics.calculate_max_drawdown(self.equity_series, 'dollar')

    @property
    def percent_max_drawdown(self):
        return self.metrics.calculate_max_drawdown(self.equity_series, 'percent')

    @property
    def log_max_drawdown_ratio(self):
        return self.metrics.calculate_log_max_drawdown_ratio(self.equity_series)
    
    @property
    def number_of_trades(self):
        return len(self.position_history)
    
    @property
    def avg_trades_per_day(self):
        return self.number_of_trades / self.days_traded

    @property
    def average_active_trades(self):
        return self.compute_portfolio_size_series().mean()

    @property
    def final_cash(self):
        self._assert_finished()
        return self.cash_series[-1]
    
    @property
    def final_equity(self):
        self._assert_finished()
        return self.equity_series[-1]

    @property
    def average_trade_length(self) -> float:
        return self.trade_summary_df.trade_length.mean()
    
    @property
    def winning_trades(self) -> pd.DataFrame:
        return self.trade_summary_df[self.trade_summary_df["percent_return"] > 0]

    @property
    def losing_trades(self) -> pd.DataFrame:
        return  self.trade_summary_df[self.trade_summary_df["percent_return"] < 0]
        
    @property
    def winning_trade_quant(self) -> int:
        return self.winning_trades.shape[0]
    
    @property
    def losing_trade_quant(self) -> int:
        return self.losing_trades.shape[0]

    @property
    def positive_trade_ratio(self) -> float:
        try:
            return (self.winning_trade_quant / (self.losing_trade_quant + self.winning_trade_quant))
        except ZeroDivisionError:
            return 0

    @property
    def average_winning_trade_return(self) -> float:
        return self.winning_trades["percent_return"].mean()

    @property
    def average_losing_trade_return(self) -> float:
        return self.losing_trades["percent_return"].mean()

    @property
    def average_price_change(self) -> float:
        return self.trade_summary_df.price_change.mean()

    @property
    def average_value_change(self) -> float:
        return self.trade_summary_df.value_change.mean()

    @property
    def average_return_per_trade(self) -> float:
        return self.trade_summary_df.percent_return.mean()
    
    @property
    def average_return_per_day(self) -> float:
        return self.log_return_series.mean()
    
    @property
    def A_score(self) -> float:
        # A score or alpha score: 10(alpha + 0.005) / 0.01
        # essentially just takes small value snd stretches it roughly
        # from 1-10
        return 10 * (self.alpha + 0.005) / 0.01

    @property
    def W_score(self) -> float:
        # W score or win score: log10(avg_win_return/avg_lose_return)
        if self.average_losing_trade_return == 0:
            return 0
        else:
            return np.log10(self.average_winning_trade_return/(abs(self.average_losing_trade_return)**2))
    
    @property
    def P_score(self) -> float:
        # P score or profit score: 100*CAGR*(win_ratio + r^2) / 2 
        # averages the percent win rate and r^2 (linearity) and penalizes
        # CAGR accordingly. only score term that can be negative
        return 100 * self.cagr * ((self.positive_trade_ratio + self.r_squared) / 2)

    @property
    def V_score(self) -> float:
        # V score or volatility score: (beta^2 + max_drawdown_% * annualized_volatility) * 10
        # attempt at a consolidated volatility score (higher is more volatile)
        return (self.beta**2 + (self.percent_max_drawdown*self.volatility)) * 10

    @property
    def edge_score(self) -> float:
        # edge score: (A_score+2W_score+P_score)^2/V_score
        # consolidates the other * scores to give overall portfolio performance at a glance 
        return (2*self.W_score + self.A_score + self.P_score)**2 / self.V_score

    _PERFORMANCE_METRICS_PROPS = [
        'percent_return',
        'spy_percent_return',
        'spy_cagr',
        'cagr',
        'excess_cagr',
        'volatility',
        'sharpe_ratio',
        'sortino_ratio',
        'calmar_ratio',
        'r_squared',
        'pure_profit_score',
        'jensens_alpha',
        'alpha',
        'beta',
        'dollar_max_drawdown',
        'percent_max_drawdown',
        'log_max_drawdown_ratio',
        'number_of_trades',
        'days_traded',
        'avg_trades_per_day',
        'average_trade_length',
        'average_active_trades',
        'winning_trade_quant',
        'losing_trade_quant',
        'positive_trade_ratio',
        'average_winning_trade_return',
        'average_losing_trade_return',
        'average_return_per_trade',
        'average_return_per_day',
        'final_equity',
        'A_score',
        'W_score',
        'P_score',
        'V_score',
        'edge_score'
    ]

    PerformancePayload = NewType('PerformancePayload', Dict[str, float])

    @property
    def performance_metric_data(self) -> PerformancePayload:
        self._assert_finished()
        props = self._PERFORMANCE_METRICS_PROPS
        to_df = {prop: [getattr(self, prop)] for prop in props}
        # to_df = {prop: [self.__dict__[prop]] for prop in props} #doesn't work because properties havent been called yet
        return pd.DataFrame(to_df)

    def print_position_summaries(self):
        for position in self.position_history:
            position.print_position_summary()

    def print_summary(self):
        self._assert_finished()
        s = f'Equity: ${self.final_equity:.2f}\n' \
            f'Percent Return: {100*self.percent_return:.2f}%\n' \
            f'S&P 500 Return: {100*self.spy_percent_return:.2f}%\n\n' \
            f'Number of trades: {self.number_of_trades}\n' \
            f'Average length of trades: {round(self.average_trade_length,1)} days\n' \
            f'Days traded: {self.days_traded} days\n' \
            f'Average active trades: {self.average_active_trades:.2f}\n' \
            f'Positive trade ratio: {round(self.positive_trade_ratio, 1)}%\n' \
            f'Average return of winning trades: {100*round(self.average_winning_trade_return, 2)}%\n' \
            f'Average return of losing trades: {100*round(self.average_losing_trade_return, 2)}%\n\n' \
            f'CAGR: {100*self.cagr:.2f}%\n' \
            f'S&P 500 CAGR: {100*self.spy_cagr:.2f}%\n' \
            f'Excess CAGR: {100*self.excess_cagr:.2f}%\n\n' \
            f'Annualized Volatility: {100*self.volatility:.2f}%\n' \
            f'Sharpe Ratio: {self.sharpe_ratio:.2f}\n' \
            f'Jensen\'s Alpha: {self.jensens_alpha:.6f}\n' \
            f'Alpha: {round(self.alpha, 6)}\n' \
            f'Beta: {round(self.beta, 4)}\n\n' \
            f'Dollar Max Drawdown: ${self.dollar_max_drawdown:.2f}\n' \
            f'Percent Max Drawdown: {100*self.percent_max_drawdown:.2f}%\n' \
            f'Log Max Drawdown Ratio: {self.log_max_drawdown_ratio:.2f}\n'

        print(s)

    def plot(self, show=True):
        """
        Plots equity, cash and portfolio value curves.
        """
        self._assert_finished()

        figure, axes = plt.subplots(nrows=3, ncols=1)
        figure.tight_layout(pad=3.0)
        axes[0].plot(self.equity_series)
        axes[0].set_title('Equity')
        axes[0].grid()

        axes[1].plot(self.cash_series)
        axes[1].set_title('Cash')
        axes[1].grid()

        axes[2].plot(self.portfolio_value_series)
        axes[2].set_title('Portfolio Value')
        axes[2].grid()

        if show:
            plt.show()

        return figure

    def plot_benchmark_comparison(self, show=True) -> plt.Figure:
        """
        Plot comparable investment in the S&P 500.
        """
        self._assert_finished()

        equity_curve = self.equity_series
        ax = equity_curve.plot()

        spy_closes = self.spy['close']
        initial_cash = self.cash_series[0]
        initial_spy = spy_closes[0]

        scaled_spy = spy_closes * (initial_cash / initial_spy)
        scaled_spy.plot()

        baseline = pd.Series(initial_cash, index=equity_curve.index)
        ax = baseline.plot(color='black')
        ax.grid()

        ax.legend(['Equity curve', 'S&P 500 portflio'])

        if show:
            plt.show()

# PortfolioHistory Usage
if __name__ == "__main__":
    symbol = 'AWU'
    df = load_data_as_pd(symbol)

    portfolio_history = PortfolioHistory()
    initial_cash = cash = 10000

    for i, row in enumerate(df.itertuples()):
        date = row.Index
        price = row.close

        if i == 123:
            # Figure out how many shares to buy
            shares_to_buy = initial_cash / price 

            # Record the position
            position = Position(symbol, date, price, shares_to_buy)

            # Spend all of your cash
            cash -= initial_cash

        elif 123 < i < 2345:
            position.record_price_update(date, price)

        elif i == 2345:
            # Sell the asset
            position.exit(date, price)

            # Get your cash back
            cash += price * shares_to_buy

            # Record the position
            portfolio_history.add_to_history(position)

        # Record cash at every step
        portfolio_history.record_cash(date, cash)

    portfolio_history.finish()

    # portfolio_history.print_position_summaries()

    portfolio_history.print_summary()

    # print(portfolio_history.performance_metric_data)

    # portfolio_history.plot_benchmark_comparison()