"""
simulator.py - Trading simulation engine for backtesting strategies against historical price data.

Classes:
    SimpleSimulator: Executes a trading strategy loop over price, signal, and preference DataFrames.
    BoundSimulators: Wraps SimpleSimulator with signal/preference functions for grid search optimization.
"""
from typing import Tuple, List, Dict, Callable, Iterable
import pandas as pd
import numpy as np
from portfolio import PortfolioHistory, Position, Symbol
from ToolKit.data_loading import concatenate_metrics, get_all_symbols, load_eod_matrix
from ToolKit.signal_generator import Signals
from collections import OrderedDict


class SimpleSimulator(object):
    """
    Executes a trading strategy against historical price data.

    Manages cash, open positions, slippage, and trade fees over the simulation period.
    Results are stored in a PortfolioHistory instance accessible after simulate() completes.
    """

    def __init__(self, initial_cash: float=10000, max_active_positions: int=5,
                 percent_slippage: float=0.0005, trade_fee: float=1):
        """
        Initializes the simulator with trading parameters.

        Args:
            initial_cash (float): Starting capital in dollars. Defaults to 10000.
            max_active_positions (int): Maximum number of simultaneously held positions. Defaults to 5.
            percent_slippage (float): Fraction of price added/subtracted on buy/sell to simulate
                                      market order slippage. Defaults to 0.0005 (0.05%).
            trade_fee (float): Fixed dollar fee per trade. Defaults to 1.
        """

        ### Set simulation parameters

        # Initial cash in porfolio
        # self.cash will fluctuate
        self.initial_cash = self.cash = initial_cash

        # Maximum number of different assets that can be help simultaneously
        self.max_active_positions: int = max_active_positions

        # The percentage difference between closing price and fill price for the
        # position, to simulate adverse effects of market orders
        self.percent_slippage = percent_slippage

        # The fixed fee in order to open a position in dollar terms
        self.trade_fee = trade_fee

        # Keep track of live trades
        self.active_positions_by_symbol: Dict[Symbol, Position] = OrderedDict()

        # Keep track of portfolio history like cash, equity, and positions
        self.portfolio_history = PortfolioHistory()

        # Ends simulation early if portfolio value approaches 0
        self.simulation_finished = False

    @property
    def active_positions_count(self) -> int:
        """Number of currently open positions."""
        return len(self.active_positions_by_symbol)

    @property
    def free_position_slots(self) -> int:
        """Number of additional positions that can be opened before hitting the max."""
        return self.max_active_positions - self.active_positions_count

    @property
    def active_symbols(self) -> List[Symbol]:
        return list(self.active_positions_by_symbol.keys())

    def print_initial_parameters(self) -> str:
        """
        Prints and returns a formatted string of the simulator's configuration parameters.

        Returns:
            str: Formatted string with initial cash, max positions, slippage, and trade fee.
        """
        s = f'Initial Cash: ${self.initial_cash} \n' \
            f'Maximum Number of Assets: {self.max_active_positions}\n' \
            f'Slipage: {self.percent_slippage}%\n' \
            f'Trade Fee: ${self.trade_fee}\n'
        print(s)
        return s

    @staticmethod
    def make_tuple_lookup(columns) -> Callable[[str, str], int]:
        """
        Map a multi-index dataframe to an itertuples-like object.

        The index of the dateframe is always the zero-th element.
        """

        # col is a hierarchical column index represented by a tuple of strings
        tuple_lookup: Dict[Tuple[str, str], int] = { 
            col: i + 1 for i, col in enumerate(columns) 
        }

        return lambda symbol, metric: tuple_lookup[(symbol, metric)]

    @staticmethod
    def make_all_valid_lookup(_idx: Callable) -> Callable:
        """
        Returns a function that checks whether all required data columns are valid for a symbol.

        Args:
            _idx (Callable): Column index lookup function from make_tuple_lookup.

        Returns:
            Callable: Function (row, symbol) -> bool, True if price, signal, and pref are all non-NaN.
        """
        return lambda row, symbol: (
            not pd.isna(row[_idx(symbol, 'pref')]) and \
            not pd.isna(row[_idx(symbol, 'signal')]) and \
            not pd.isna(row[_idx(symbol, 'price')])
        )

    def buy_to_open(self, symbol: Symbol, date: pd.Timestamp, price: float) -> None:
        """
        Opens a new long position, spending an equal share of available cash.

        Applies slippage and trade fee. Ends simulation early if cash is insufficient.

        Args:
            symbol (Symbol): Ticker symbol to buy.
            date (pd.Timestamp): Date of the transaction.
            price (float): Current market price per share.

        Raises:
            AssertionError: If the symbol is already an active position.
            AssertionError: If spending would result in negative cash.
        """
        if self.simulation_finished:
            return
        # Figure out how much we are willing to spend
        cash_to_spend = self.cash / self.free_position_slots

        # Make sure transaction won't bankrupt us. if so, initiate
        # finishing of the simulation early.
        if self.cash <= self.trade_fee: 
            self.simulation_finished = True
            return 
        
        # Dont include trade fee in share purchase
        cash_to_spend -= self.trade_fee

        # Calculate buy_price and number of shares. Fractional shares allowed.
        purchase_price = (1 + self.percent_slippage) * price
        shares = cash_to_spend / purchase_price

        # Spend the cash
        self.cash -= cash_to_spend + self.trade_fee
        assert self.cash >= 0, 'Spent cash you do not have.'
        self.portfolio_history.record_cash(date, self.cash)   

        # Record the position
        positions_by_symbol = self.active_positions_by_symbol
        assert not symbol in positions_by_symbol, 'Symbol already in portfolio.'        
        position = Position(symbol, date, purchase_price, shares)
        positions_by_symbol[symbol] = position

    def sell_to_close(self, symbol: Symbol, date: pd.Timestamp, price: float) -> None:
        """
        Closes an active position, recovers cash, and records it in portfolio history.

        Args:
            symbol (Symbol): Ticker symbol to sell.
            date (pd.Timestamp): Date of the transaction.
            price (float): Current market price per share.

        Raises:
            KeyError: If symbol is not currently an active position.
        """

        # Exit the position
        positions_by_symbol = self.active_positions_by_symbol
        position = positions_by_symbol[symbol]
        position.exit(date, price)
        # position.print_position_summary()

        # Receive the cash
        sale_value = position.last_value * (1 - self.percent_slippage)
        self.cash += sale_value
        self.portfolio_history.record_cash(date, self.cash)

        # Record in portfolio history
        self.portfolio_history.add_to_history(position)
        del positions_by_symbol[symbol]
    
    @staticmethod
    def _assert_equal_columns(*args: Iterable[pd.DataFrame]) -> None:
        """
        Asserts that all provided DataFrames have identical column sets.

        Args:
            *args (Iterable[pd.DataFrame]): Two or more DataFrames to compare.

        Raises:
            AssertionError: If any DataFrame has a different set of column names.
        """
        column_names = set(args[0].columns.values)
        for arg in args[1:]:
            assert set(arg.columns.values) == column_names, \
                'Found unequal column names in input data frames.'

    def simulate(self, price: pd.DataFrame, signal: pd.DataFrame,
                 preference: pd.DataFrame) -> None:
        """
        Runs the full trading simulation over the provided DataFrames.

        Iterates day-by-day over all symbols, executing sells on sell signals,
        then buys on buy signals ranked by preference. Uses itertuples for performance.

        Args:
            price (pd.DataFrame): Date-indexed DataFrame of close prices, one column per symbol.
            signal (pd.DataFrame): Date-indexed DataFrame of signals (1=buy, -1=sell, 0=hold).
            preference (pd.DataFrame): Date-indexed DataFrame of preference scores for ranking buys.

        Raises:
            AssertionError: If price, signal, and preference do not share the same column set.
        """
        _sale_times = []
        _buy_times = []
        _record_times = []
        # Create a hierarchical data frame to loop through
        self._assert_equal_columns(price, signal, preference)
        df = concatenate_metrics({
            'price': price,
            'signal': signal,
            'pref': preference,
        })

        # Get list of symbols
        all_symbols = list(set(price.columns.values))

        # Get lookup functions
        _idx = self.make_tuple_lookup(df.columns)
        _all_valid = self.make_all_valid_lookup(_idx)

        # Store some variables
        active_positions_by_symbol = self.active_positions_by_symbol
        max_active_positions = self.max_active_positions

        # Iterating over all dates.
        # itertuples() is significantly faster than iterrows(), it however comes
        # at the cost of being able index easily. In order to get around this
        # we use an tuple lookup function: "_idx"
        for i, row in enumerate(df.itertuples()):

            # date index is always first element of tuple row
            date = row[0]

            # Get symbols with valid and tradable data
            symbols: List[str] = [s for s in all_symbols if _all_valid(row, s)]


            # Iterate over active positions and sell stocks with a sell signal.
            _active = self.active_symbols
            to_exit = [s for s in _active if row[_idx(s, 'signal')] == -1]
            for s in to_exit:
                sell_price = row[_idx(s, 'price')]
                self.sell_to_close(s, date, sell_price)

            # Get up to max_active_positions symbols with a buy signal in 
            # decreasing order of preference
            to_buy = [
                s for s in symbols if \
                    row[_idx(s, 'signal')] == 1 and \
                    not s in active_positions_by_symbol
            ]
            to_buy.sort(key=lambda s: row[_idx(s, 'pref')], reverse=True)
            to_buy = to_buy[:max_active_positions]
            # makes sure we don't churn posiitons on the last day of trading
            if date == df.index[-1]:
                to_buy = []

            trade_made = False
            for s in to_buy:
                buy_price = row[_idx(s, 'price')]
                buy_preference = row[_idx(s, 'pref')]

                # If we have some empty slots, just buy the asset outright
                if self.active_positions_count < max_active_positions:
                    self.buy_to_open(s, date, buy_price)
                    trade_made = True
                    continue

                # If are holding max_active_positions, evaluate a swap based on
                # preference
                _active = self.active_symbols
                active_prefs = [(s, row[_idx(s, 'pref')]) for s in _active]

                _min = min(active_prefs, key=lambda k: k[1])
                min_active_symbol, min_active_preference = _min

                # If a more preferable symbol exists, then sell an old one
                if min_active_preference < buy_preference:
                    sell_price = row[_idx(min_active_symbol, 'price')]
                    self.sell_to_close(min_active_symbol, date, sell_price)
                    self.buy_to_open(s, date, buy_price)
                    trade_made = True

            if len(to_exit) == 0 and not trade_made:
                self.portfolio_history.record_cash(date, self.cash)

            # Update price data everywhere
            for s in self.active_symbols:
                price = row[_idx(s, 'price')]
                position = active_positions_by_symbol[s]
                position.record_price_update(date, price)

            portfolio_value = 0
            for s in self.active_symbols:
                current_price = row[_idx(s, 'price')] 
                portfolio_value += current_price * self.active_positions_by_symbol[s].shares
                self.current_portfolio_value = portfolio_value

            # Checks to see if simulation ended early do to
            # attempted account overdraft
            if self.simulation_finished:
                wait_1 = False
                for s in self.active_symbols:
                    if self.active_positions_by_symbol[s].entry_date == date:
                        wait_1 = True
                if wait_1 == True:
                    continue
                # final cash still negative? need to make sure not
                # rogue buy operations after self.simulation_finished = True
                # and track cash closely through last 2 cycles
                for s in self.active_symbols: 
                    self.sell_to_close(s, date, row[_idx(s, 'price')])
                self.portfolio_history.finish() 
                return
            
        # Sell all positions and mark simulation as complete
        for s in self.active_symbols:
            self.sell_to_close(s, date, row[_idx(s, 'price')])
        self.portfolio_history.finish()


class BoundSimulators():
    """
    Binds signal and preference functions to a SimpleSimulator for use in grid search optimization.

    Loads all available price data on construction and applies the provided signal and preference
    functions via DataFrame.apply() each time simulate() is called. Supports MA-type variation
    by injecting ma_type as a keyword argument to the signal function.
    """

    def __init__(
        self, signal_func: Callable,
        preference_func: Callable,
        contains_ma_type: bool=False,
        **sim_kwargs
    ):
        """
        Initializes BoundSimulators, loads all symbol data, and stores function references.

        Args:
            signal_func (Callable): Function applied column-wise to the price DataFrame to
                                    generate buy/sell signals.
            preference_func (Callable): Function applied column-wise to generate preference scores.
            contains_ma_type (bool): If True, passes ma_type as a kwarg to signal_func. Defaults to False.
            **sim_kwargs: Keyword arguments forwarded to SimpleSimulator (e.g., max_active_positions).
        """
        symbols = get_all_symbols()
        self.prices = load_eod_matrix(symbols)
        self.max_posiitons = sim_kwargs["max_active_positions"]
        self.contains_ma_type = contains_ma_type
        self.sim_kwargs = sim_kwargs

        self.signal_func = signal_func
        self.pref_func = preference_func
        self.signal_id = self._make_funciton_id(signal_func.__name__)
        self.pref_id = self._make_funciton_id(preference_func.__name__)

    @staticmethod
    def _make_funciton_id(function_name: str, letters_per_word: int=5) -> str:
        """
        Makes a unique string for each function used as signal / preference 
        splits name on underscores and removes 'calculate' and 'create' keywords
        concat first 5 letters of each word in funciton
        """
        listed = function_name.split("_")[1:]
        _id = ""
        for word in listed:
            to_concat = letters_per_word
            if len(word) < letters_per_word:
                to_concat = len(word)
            _id = _id + word[:to_concat]
        return _id.upper()

    def simulate(
        self, signal_args: tuple, preference_args: tuple
    ) -> pd.DataFrame:
        """
        Runs a single simulation with the given signal and preference arguments.

        Args:
            signal_args (tuple): Positional arguments passed to signal_func (excluding the series).
            preference_args (tuple): Positional arguments passed to preference_func.

        Returns:
            pd.DataFrame: Performance metric DataFrame from PortfolioHistory.performance_metric_data.
        """
        if self.contains_ma_type:
            signal = self.prices.apply(self.signal_func, args=signal_args, ma_type=self.ma_type, axis=0)
            if not self.signal_id == self.ma_type + self.signal_id:
                self.signal_id = self.ma_type + self.signal_id
        else:
            signal = self.prices.apply(self.signal_func, args=signal_args, axis=0)
        preference = self.prices.apply(self.pref_func, args=preference_args, axis=0)
        simulator = SimpleSimulator(**self.sim_kwargs)
        simulator.simulate(self.prices, signal, preference)
        simulator.params = (self.signal_id, self.pref_id)

        return simulator, simulator.portfolio_history.performance_metric_data



# Example Usage
# the strategy given in the book. signals are bollinger band crossings
# and preference is calculated using a rolling sharpe ratio

# if __name__ == '__main__':
    # def simulate_portfolio():

    #     bollinger_n = 20
    #     sharpe_n = 20

    #     # Load in data
    #     symbols: List[str] = get_all_symbols()
    #     prices: pd.DataFrame = load_eod_matrix(symbols)

    #     # Use the bollinger band outer band crossorver as a signal
    #     _bollinger = Signals().create_bollinger_band_signal
    #     signal = prices.apply(_bollinger, args=(bollinger_n,), axis=0)
    #     # Use a rolling sharpe ratio approximation as a preference matrix
    #     # cagr = Metrics().calculate_cagr(prices)
    #     # print(cagr)
    #     _sharpe = Signals().calculate_rolling_sharpe_ratio
    #     preference = prices.apply(_sharpe, args=(sharpe_n,), axis=0)
    #     # Run the simulator
    #     simulator = SimpleSimulator(
    #         initial_cash=10000,
    #         max_active_positions=5,
    #         percent_slippage=0.0005,
    #         trade_fee=1,
    #     )
    #     simulator.simulate(prices, signal, preference)

    #     # Print results
    #     # simulator.portfolio_history.print_position_summaries()
    #     simulator.print_initial_parameters()
    #     simulator.portfolio_history.print_summary()
    #     print(simulator.portfolio_history.performance_metric_data)
    #     simulator.portfolio_history.plot_benchmark_comparison()

#     simulate_portfolio()