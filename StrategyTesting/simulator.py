from typing import Tuple, List, Dict, Callable, Iterable
import pandas as pd
import numpy as np
from timeit import default_timer
from portfolio import PortfolioHistory, Position, Symbol
from ToolKit.data_loading import concatenate_metrics, get_all_symbols, load_eod_matrix
from ToolKit.signal_generator import Signals
from collections import OrderedDict


class SimpleSimulator(object):
    """
    A simple trading simulator to work with the PortfolioHistory class
    """

    def __init__(self, initial_cash: float=10000, max_active_positions: int=5,
        percent_slippage: float=0.0005, trade_fee: float=1):

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
    def active_positions_count(self):
        return len(self.active_positions_by_symbol)

    @property
    def free_position_slots(self):
        return self.max_active_positions - self.active_positions_count

    @property
    def active_symbols(self) -> List[Symbol]:
        return list(self.active_positions_by_symbol.keys())

    def print_initial_parameters(self):
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
    def make_all_valid_lookup(_idx: Callable):
        """
        Return a function that checks for valid data, given a lookup function
        """
        return lambda row, symbol: (
            not pd.isna(row[_idx(symbol, 'pref')]) and \
            not pd.isna(row[_idx(symbol, 'signal')]) and \
            not pd.isna(row[_idx(symbol, 'price')])
        )

    def buy_to_open(self, symbol, date, price):
        """
        Keep track of new position, make sure it isn't an existing position. 
        Verify you have cash.
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

    def sell_to_close(self, symbol, date, price):
        """
        Keep track of exit price, recover cash, close position, and record it in
        portfolio history.

        Will raise a KeyError if symbol isn't an active position
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
    def _assert_equal_columns(*args: Iterable[pd.DataFrame]):
        column_names = set(args[0].columns.values)
        for arg in args[1:]:
            assert set(arg.columns.values) == column_names, \
                'Found unequal column names in input data frames.'

    def simulate(self, price: pd.DataFrame, signal: pd.DataFrame, 
        preference: pd.DataFrame):
        """
        Runs the simulation.

        price, signal, and preference are data frames with the column names 
        represented by the same set of stock symbols.
        """
        _sale_times = []
        _buy_times = []
        _record_times = []
        start_time = default_timer()
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
        iteration_start = default_timer()
        for i, row in enumerate(df.itertuples()):

            # date index is always first element of tuple row
            date = row[0]

            # Get symbols with valid and tradable data
            symbols: List[str] = [s for s in all_symbols if _all_valid(row, s)]

            sale_start = default_timer()

            # Iterate over active positions and sell stocks with a sell signal.
            _active = self.active_symbols
            to_exit = [s for s in _active if row[_idx(s, 'signal')] == -1]
            for s in to_exit:
                sell_price = row[_idx(s, 'price')]
                self.sell_to_close(s, date, sell_price)

            sale_end = default_timer()
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

            buy_end = default_timer()

            if len(to_exit) == 0 and not trade_made:
                self.portfolio_history.record_cash(date, self.cash)

            # Update price data everywhere
            for s in self.active_symbols:
                price = row[_idx(s, 'price')]
                position = active_positions_by_symbol[s]
                position.record_price_update(date, price)

            record_cash_end = default_timer()

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
                finish_time = default_timer()
                return
        
        _sale_times.append(sale_end - sale_start)
        _buy_times.append(buy_end - sale_end)
        _record_times.append(record_cash_end - buy_end)
            
        # Sell all positions and mark simulation as complete
        for s in self.active_symbols:
            self.sell_to_close(s, date, row[_idx(s, 'price')])
        self.portfolio_history.finish()        

        finish_time = default_timer()

        self.time_data = pd.DataFrame({
            "setup_time" : [iteration_start - start_time],
            "transaction_time": np.average(_sale_times) 
                                + np.average(_buy_times) 
                                + np.average(_record_times),
            "finish_time": [finish_time - record_cash_end],
            "internal_sim_time": [finish_time - iteration_start]
        })


class BoundSimulators():
    def __init__(
        self, signal_func: Callable, 
        preference_func: Callable,
        **sim_kwargs
    ):
        symbols = get_all_symbols()
        self.prices = load_eod_matrix(symbols)
        self.max_posiitons = sim_kwargs["max_active_positions"]
        self.sim_kwargs = sim_kwargs

        self.signal_func = signal_func
        self.pref_func = preference_func
        self.signal_id = self._make_funciton_id(signal_func.__name__)
        self.pref_id = self._make_funciton_id(preference_func.__name__)

    @staticmethod
    def _make_funciton_id(function_name: str) -> str:
        """
        Makes a unique string for each function used as signal / preference 
        splits name on underscores and removes 'calculate' and 'create' keywords
        concat first 5 letters of each word in funciton
        """
        listed = function_name.split("_")[1:]
        _id = ""
        for word in listed:
            to_concat = 5
            if len(word) < 5:
                to_concat = len(word)
            _id = _id + word[:to_concat]
        return _id.upper()

    def simulate_lookback_only(self, signal_n: int, preference_n: int) -> pd.DataFrame:
        """
        Simulation that takes 1 signal and 1 preference and the only editable 
        parameters are the lookback window for the buy signal and transaction
        preference
        """
        calc_start = default_timer()
        signal = self.prices.apply(self.signal_func, args=(signal_n,), axis=0)
        preference = self.prices.apply(self.pref_func, args=(preference_n,), axis=0)
        calc_end = default_timer()
        simulator = SimpleSimulator(**self.sim_kwargs)
        simulator.simulate(self.prices, signal, preference)
        simulator.params = (self.signal_id, self.pref_id)
        sim_end_time = default_timer()
        simulator.time_data["calculation_time"] = calc_end - calc_start
        simulator.time_data["mid_sim_time"] = sim_end_time - calc_end

        return simulator, simulator.portfolio_history.performance_metric_data

    def simulate_single_ma_lookback(
        self, signal_n: int, preference_n: int
    ) -> pd.DataFrame:
        """
        Simulation that takes 1 signal and 1 preference and the only editable 
        parameters are the lookback window for the buy signal and transaction
        preference and the type of moving average signal 
        """
        calc_start = default_timer()
        signal = self.prices.apply(self.signal_func, args=(signal_n,), ma_type=self.ma_type, axis=0)
        preference = self.prices.apply(self.pref_func, args=(preference_n,), axis=0)
        calc_end = default_timer()
        simulator = SimpleSimulator(**self.sim_kwargs)
        simulator.simulate(self.prices, signal, preference)
        simulator.params = (self.signal_id, self.pref_id)
        sim_end_time = default_timer()
        simulator.time_data["calculation_time"] = calc_end - calc_start
        simulator.time_data["mid_sim_time"] = sim_end_time - calc_end

        return simulator, simulator.portfolio_history.performance_metric_data
    
    def simulate_dual_lookback(
        self, signal_n1: int, signal_n2: int, preference_n: int
    ) -> pd.DataFrame:
        """
        Simulation that takes 1 signal and 1 preference and the only editable 
        parameters are the lookback window for the buy signal and transaction
        preference and the type of moving average signal 
        """
        if signal_n2 < signal_n1:
            return None, None
        calc_start = default_timer()
        signal = self.prices.apply(self.signal_func, args=(signal_n1, signal_n2), axis=0)
        preference = self.prices.apply(self.pref_func, args=(preference_n,), axis=0)
        calc_end = default_timer()
        simulator = SimpleSimulator(**self.sim_kwargs)
        simulator.simulate(self.prices, signal, preference)
        simulator.params = (self.signal_id, self.pref_id)
        sim_end_time = default_timer()
        simulator.time_data["calculation_time"] = calc_end - calc_start
        simulator.time_data["mid_sim_time"] = sim_end_time - calc_end

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

    #     start1 = default_timer()
    #     # Use the bollinger band outer band crossorver as a signal
    #     _bollinger = Signals().create_bollinger_band_signal
    #     signal = prices.apply(_bollinger, args=(bollinger_n,), axis=0)
    #     print(f"bollinger belts time: {default_timer() - start1}")
    #     # Use a rolling sharpe ratio approximation as a preference matrix
    #     start2 = default_timer()
    #     # cagr = Metrics().calculate_cagr(prices)
    #     # print(cagr)
    #     _sharpe = Signals().calculate_rolling_sharpe_ratio
    #     preference = prices.apply(_sharpe, args=(sharpe_n,), axis=0)
    #     print(f"rolling sharpe time: {default_timer() - start2}")
    #     # Run the simulator
    #     simulator = SimpleSimulator(
    #         initial_cash=10000,
    #         max_active_positions=5,
    #         percent_slippage=0.0005,
    #         trade_fee=1,
    #     )
    #     start3 = default_timer()
    #     simulator.simulate(prices, signal, preference)
    #     print(f"simulator time: {default_timer() - start3}")

    #     # Print results
    #     # simulator.portfolio_history.print_position_summaries()
    #     simulator.print_initial_parameters()
    #     simulator.portfolio_history.print_summary()
    #     print(simulator.portfolio_history.performance_metric_data)
    #     simulator.portfolio_history.plot_benchmark_comparison()

#     simulate_portfolio()