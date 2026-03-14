#!python3
"""
signal_generator.py - Technical indicators, financial metrics, and trading signal generators.

Classes:
    Metrics: Financial performance metrics (returns, volatility, drawdown, Sharpe, alpha/beta).
    Indicators: Technical indicators (moving averages, oscillators, volume metrics).
                Inherits from Metrics.
    Signals: Buy/sell signal generators derived from technical indicators.
             Inherits from Indicators.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable
# import swifter
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class Metrics():
    """
    Financial performance metrics for price and return series.

    Calculates return series, volatility, CAGR, Sharpe ratio, rolling Sharpe ratio,
    annualized downside deviation, Sortino ratio, drawdown series, max drawdown (with
    metadata), log max drawdown ratio, Calmar ratio, pure profit score, Jensen's alpha,
    beta, alpha, and R-squared relative to a benchmark.
    """
    def __init__(self):
        """
        Initializes Metrics with available drawdown evaluator methods mapped by name.
        """
        self.return_series_types = ["log", "percent"]
        # self.DRAWDOWN_EVALUATORS: Dict[str, Callable] = {
        #     'dollar': lambda price, peak: peak - price,
        #     'percent': lambda price, peak: -((price / peak) - 1),
        #     'log': lambda price, peak: np.log(peak) - np.log(price),
        # }

        self.DRAWDOWN_EVALUATORS: Dict[str, Callable] = {
            'dollar': self.dollar_drawdown,
            'percent': self.percent_drawdown,
            'log': self.log_drawdown,
        }

    @staticmethod
    def dollar_drawdown(price: float, peak: float) -> float:
        """
        Calculates the dollar drawdown from a peak price.

        Args:
            price (float): Current price.
            peak (float): Peak price reached prior to the current price.

        Returns:
            float: Dollar difference between peak and current price.
        """
        return peak - price

    @staticmethod
    def percent_drawdown(price: float, peak: float) -> float:
        """
        Calculates the percent drawdown from a peak price.

        Args:
            price (float): Current price.
            peak (float): Peak price reached prior to the current price.

        Returns:
            float: Percent drawdown as a positive decimal (e.g., 0.1 for 10% drawdown).
        """
        return -((price / peak) - 1)

    @staticmethod
    def log_drawdown(price: float, peak: float) -> float:
        """
        Calculates the log drawdown from a peak price.

        Args:
            price (float): Current price.
            peak (float): Peak price reached prior to the current price.

        Returns:
            float: Log drawdown (log(peak) - log(price)).
        """
        return np.log(peak) - np.log(price)

    @staticmethod
    def calculate_return_series(series: pd.Series) -> pd.Series:
        """
        Calculates the period-over-period return series of a price series.

        The first value will always be NaN. Output series retains the index of the input.

        Args:
            series (pd.Series): Date-indexed price series.

        Returns:
            pd.Series: Return series (e.g., 0.01 for 1% gain in a period).
        """
        shifted_series = series.shift(1, axis=0)
        return series / shifted_series - 1
    
    @staticmethod
    def calculate_percent_return(series: pd.Series) -> float:
        """
        Calculates the total percent return from the first to the last value of a series.

        Args:
            series (pd.Series): Date-indexed price series.

        Returns:
            float: Percent return as a decimal (e.g., 0.25 for 25%).
        """
        return series.iloc[-1] / series.iloc[0] - 1
    
    @staticmethod
    def calculate_log_return_series(series: pd.Series) -> pd.Series:
        """Same as calculate_return_series but with log returns"""
        shifted_series = series.shift(1, axis=0)
        return pd.Series(np.log(series / shifted_series))
    
    @staticmethod
    def get_years_past(series: pd.Series) -> float:
        """
        Calculate the years past according to the index of the series for use with
        functions that require annualization   
        """
        start_date = series.index[0]
        end_date = series.index[-1]
        return (end_date - start_date).days / 365.25
    

    def calculate_annualized_volatility(self, return_series: pd.Series) -> float:
        """
        Calculates annualized volatility for a date-indexed return series. 
        Works for any interval of date-indexed prices and returns.
        """
        years_past = self.get_years_past(return_series)
        entries_per_year = return_series.shape[0] / years_past
        return return_series.std() * np.sqrt(entries_per_year)
    

    def calculate_cagr(self, series: pd.Series) -> float:
        """
        Calculate compounded annual growth rate
        """
        value_factor = series.iloc[-1] / series.iloc[0]
        year_past = self.get_years_past(series)
        res = (value_factor ** (1 / year_past)) - 1
        if type(res) != np.float64:
            print(f"value factor: {value_factor}\nyears past: {year_past}\nres: {res}")
        return res


    def calculate_sharpe_ratio(
        self, price_series: pd.Series, 
        benchmark_rate: float=0
    ) -> float:
        """
        Calculates the sharpe ratio given a price series. Defaults to benchmark_rate
        of zero.
        """
        cagr = self.calculate_cagr(price_series)
        return_series = self.calculate_return_series(price_series)
        volatility = self.calculate_annualized_volatility(return_series)
        if volatility == 0.0:
            return None
        return (cagr - benchmark_rate) / volatility
    

    def calculate_rolling_sharpe_ratio(
        self, price_series: pd.Series,n: float=20
    ) -> pd.Series:
        """
        Compute an approximation of the Sharpe ratio on a rolling basis. 
        Intended for use as a preference value.
        """
        rolling_return_series = self.calculate_return_series(price_series).rolling(n)
        return rolling_return_series.mean() / rolling_return_series.std()


    def calculate_annualized_downside_deviation(
        self, return_series: pd.Series, 
        benchmark_rate: float=0
    ) -> float:
        """
        Calculates the downside deviation for use in the sortino ratio.

        Benchmark rate is assumed to be annualized. It will be adjusted according 
        to the number of periods per year seen in the data.
        """

        # For both de-annualizing the benchmark rate and annualizing result
        years_past = self.get_years_past(return_series)
        entries_per_year = return_series.shape[0] / years_past

        adjusted_benchmark_rate = ((1+benchmark_rate) ** (1/entries_per_year)) - 1

        downside_series = adjusted_benchmark_rate - return_series
        downside_sum_of_squares = (downside_series[downside_series > 0] ** 2).sum()
        denominator = return_series.shape[0] - 1
        downside_deviation = np.sqrt(downside_sum_of_squares / denominator)

        return downside_deviation * np.sqrt(entries_per_year)


    def calculate_sortino_ratio(
        self, price_series: pd.Series, 
        benchmark_rate: float=0
    ) -> float:
        """Calculates the sortino ratio."""

        cagr = self.calculate_cagr(price_series)
        return_series = self.calculate_return_series(price_series)
        downside_deviation = self.calculate_annualized_downside_deviation(return_series)
        return (cagr - benchmark_rate) / downside_deviation


    def calculate_drawdown_series(self, series: pd.Series, method: str='log') -> pd.Series:
        """Returns the drawdown series"""
        assert method in self.DRAWDOWN_EVALUATORS, \
            f'Method "{method}" must by one of {list(self.DRAWDOWN_EVALUATORS.keys())}'

        evaluator = self.DRAWDOWN_EVALUATORS[method]
        return evaluator(series, series.cummax())


    def calculate_max_drawdown(self, series: pd.Series, method: str='log') -> float:
        """Returns the max drawdown as a float"""
        return self.calculate_drawdown_series(series, method).max()


    def calculate_max_drawdown_with_metadata(
        self, series: pd.Series, 
        method: str='log'
    ) -> Dict[str, Any]:
        """
        Calculates max_drawdown and stores metadata about when and where. Returns 
        a dictionary of the form 
            {
                'max_drawdown': float,
                'peak_date': pd.Timestamp,
                'peak_price': float,
                'trough_date': pd.Timestamp,
                'trough_price': float,
            }
        """

        assert method in self.DRAWDOWN_EVALUATORS, \
            f'Method "{method}" must by one of {list(self.DRAWDOWN_EVALUATORS.keys())}'

        evaluator = self.DRAWDOWN_EVALUATORS[method]

        max_drawdown = 0
        local_peak_date = peak_date = trough_date = series.index[0]
        local_peak_price = peak_price = trough_price = series.iloc[0]

        for date, price in series.iteritems():

            # Keep track of the rolling max
            if price > local_peak_price:
                local_peak_date = date
                local_peak_price = price

            # Compute the drawdown
            drawdown = evaluator(price, local_peak_price)

            # Store new max drawdown values
            if drawdown > max_drawdown:
                max_drawdown = drawdown

                peak_date = local_peak_date
                peak_price = local_peak_price

                trough_date = date
                trough_price = price

        return {
            'max_drawdown': max_drawdown,
            'peak_date': peak_date,
            'peak_price': peak_price,
            'trough_date': trough_date,
            'trough_price': trough_price
        }

    def calculate_log_max_drawdown_ratio(self, series: pd.Series) -> float:
        """
        Calculates the log return minus the log max drawdown ratio.

        A higher value indicates better risk-adjusted return on a log scale.

        Args:
            series (pd.Series): Date-indexed price series.

        Returns:
            float: Log return minus log max drawdown.
        """
        log_drawdown = self.calculate_max_drawdown(series, method='log')
        log_return = np.log(series.iloc[-1]) - np.log(series.iloc[0])
        return log_return - log_drawdown

    def calculate_calmar_ratio(self, series: pd.Series) -> float:
        """
        Return the percent max drawdown ratio over the past three years using
        CAGR as the numerator, otherwise known as the Calmar Ratio
        """
        # Compute annualized percent max drawdown ratio
        percent_drawdown = self.calculate_max_drawdown(series, method='percent')
        cagr = self.calculate_cagr(series)
        return cagr / percent_drawdown

    def calculate_pure_profit_score(self, price_series: pd.Series) -> float:
        """
        Calculates the pure profit score
        """
        cagr = self.calculate_cagr(price_series)

        # Build a single column for a predictor, t
        t: np.ndarray = np.arange(0, price_series.shape[0]).reshape(-1, 1)

        # Fit the regression
        regression = LinearRegression().fit(t, price_series)

        # Get the r-squared value
        r_squared = regression.score(t, price_series)

        return cagr * r_squared

    @staticmethod
    def calculate_jensens_alpha(
        return_series: pd.Series, 
        benchmark_return_series: pd.Series
    ) -> float: 
        """
        Calculates jensens alpha. Prefers input series have the same index. Handles
        NAs.
        """

        # Join series along date index and purge NAs
        df = pd.concat([return_series, benchmark_return_series], sort=True, axis=1)
        df = df.dropna()

        # Get the appropriate data structure for scikit learn
        clean_returns: pd.Series = df[return_series.name]
        clean_benchmarks = pd.DataFrame(df[benchmark_return_series.name])

        # Fit a linear regression and return the alpha
        regression = LinearRegression().fit(clean_benchmarks, y=clean_returns)
        return regression.intercept_
    
    @staticmethod
    def _get_linreg(
        return_series: pd.Series,
        benchmark_return_series: pd.Series
    ) -> LinearRegression:
        """
        Fits a linear regression of return_series against benchmark_return_series.

        Used internally by calculate_beta, calculate_alpha, and calculate_r_squared.
        Joins both series on their date index and drops NaN rows before fitting.

        Args:
            return_series (pd.Series): Named return series for the portfolio or stock.
            benchmark_return_series (pd.Series): Named return series for the benchmark.

        Returns:
            LinearRegression: Fitted sklearn LinearRegression model.
        """
        df = pd.concat([return_series, benchmark_return_series], sort=True, axis=1)
        df = df.dropna()
        clean_returns: pd.Series = df[return_series.name]
        clean_benchmarks = pd.DataFrame(df[benchmark_return_series.name])
        return LinearRegression().fit(clean_benchmarks, y=clean_returns)

    def calculate_beta(
        self, return_series: pd.Series,
        benchmark_return_series: pd.Series
    ) -> float:
        """
        Calculates the beta of a portfolio or stock return series
        versus a benchmark
        """
        return self._get_linreg(return_series, benchmark_return_series).coef_[0]
    
    def calculate_alpha(
        self, return_series: pd.Series,
        benchmark_return_series: pd.Series
    ) -> float:
        """
        Calculates the alpha of a portfolio or stock return series
        versus a benchmark
        """
        return self._get_linreg(return_series, benchmark_return_series).intercept_
    
    def calculate_r_squared(
        self, return_series: pd.Series,
        benchmark_return_series: pd.Series
    ) -> float:
        """
        Calculates the r^2 of a portfolio or stock return series
        versus a benchmark
        """
        reshaped_return = np.arange(0, return_series.shape[0]).reshape(-1, 1)
        reshaped_bench = np.arange(0, benchmark_return_series.shape[0]).reshape(-1, 1)
        reg = self._get_linreg(return_series, benchmark_return_series)
        return reg.score(reshaped_return, reshaped_bench)


class Indicators(Metrics):
    """
    Technical indicator calculator. Inherits all performance metrics from Metrics.

    Supported indicators:
        Moving averages: SMA, TMA, WMA, EMA, DEMA, TEMA, KAMA
        Oscillators: MACD, Stochastic, RSI, Stochastic RSI, Williams %R, CCI,
                     Chande, ROC, Aroon (up/down/oscillator), Ultimate Oscillator
        Volatility: Bollinger Bands, ATR
        Directional: Plus/Minus DM, Plus/Minus DI, DX, ADX, ADXR
        Volume-based: VWAP, MFI, CMF, OBV, Money Flow Volume
        Other: Momentum, Typical Price, PPO, Balance of Power, Midpoint, TRIX
    """

    def __init__(self):
        """
        Initializes Indicators, calling Metrics.__init__ via super().
        """
        super().__init__()

    def get_ma_from_string(self, string: str) -> Callable:
        """
        Returns the moving average method corresponding to the given string identifier.

        Defaults to SMA if the provided string is not recognized.

        Args:
            string (str): Moving average type. One of: "SMA", "TMA", "WMA", "EMA", "DEMA", "TEMA".

        Returns:
            Callable: The bound calculate_* method for the specified moving average type.
        """
        compatible = [
            "SMA","TMA","WMA",
            "EMA","DEMA", "TEMA"
        ]
        if string not in compatible:
            string = compatible[0]
        moving_average = getattr(self, f"calculate_{string}")
        return moving_average

    @staticmethod
    def calculate_momentum(series: pd.Series, n: int=10) -> pd.Series:
        """
        Calculates the momentum (price - price[-n])
        """
        mom = series.rolling(n).apply(lambda x: x[-1] - x[0])
        mom.name = "Momentum"
        return mom

    @staticmethod
    def calculate_SMA(series: pd.Series, n: int=20) -> pd.Series:
        """
        Calculates the simple moving average
        """
        sma = series.rolling(n).mean()
        sma.name = "Simple Moving Average"
        return sma
    
    def calculate_TMA(self, series: pd.Series, n: int=20) -> pd.Series:
        """
        Calculates the triangular moving average
        """
        sma = self.calculate_SMA(series, n)
        tma = self.calculate_SMA(sma, n)
        tma.name = "Triangular Moving Average"
        return tma
    
    @staticmethod
    def calculate_WMA(series: pd.Series, n: int=20) -> pd.Series:
        """
        Calculates weighted moving average of a data set
        """
        denominator = int((n * (n+1)) / 2)
        weights = [(i + 1)/denominator for i in range(n)]
        wma = series.rolling(n).apply(lambda x: np.sum(weights * x))
        wma.name = "Weighted Moving Average"
        return wma
    
    @staticmethod
    def calculate_EMA(series: pd.Series, n: int=20) -> pd.Series:
        """
        Calculates exponential moving average
        """
        ewm = series.ewm(span=n, adjust=False).mean()   
        ewm.name = "Exponential Moving Average"
        return ewm
    
    def calculate_DEMA(self, series: pd.Series, n: int=20) -> pd.Series:
        """
        Calculates the double exponential moving average
        """
        ema = self.calculate_EMA(series, n)
        ema_of_ema = self.calculate_EMA(ema, n)
        dema = (2 * ema) - ema_of_ema
        dema.name = "Double Exponential Moving Average"
        return dema
    
    def calculate_TEMA(self, series: pd.Series, n: int=20) -> pd.Series:
        """
        Calculates the triple exponential moving average
        """
        ema_1 = self.calculate_EMA(series, n)
        ema_2 = self.calculate_EMA(ema_1, n)
        ema_3 = self.calculate_EMA(ema_2, n)
        tema = (3 * ema_1) - (3 * ema_2) + ema_3
        tema.name = "Triple Exponential Moving Average"
        return tema

    def calculate_KAMA(
        self, series: pd.Series, n: int=10,
        fast_lookback: int=5, slow_lookback: int=30
    ) -> pd.Series:
        """
        Calculates Kaufmans adaptive moving average
        """
        # Calculate the efficiency ratio
        change = self.calculate_momentum(series, n)
        abs_dollar_retrun = abs(series - series.shift(1))
        volatility = abs_dollar_retrun.rolling(n).sum()
        er = change / volatility

        # Calculate the smoothing constant
        fast_sc = 2/(fast_lookback + 1)
        slow_sc = 2/(slow_lookback + 1)
        sc = (er * (fast_sc - slow_sc) + fast_sc) ** 2

        # Calculate kaufmans adaptive moving average
        kama = [self.calculate_SMA(series, n).dropna()[0]]
        last_value = kama[0]
        for i, value in enumerate(sc):
            if i == (n - 1):
                continue
            if value > 0:
                kama_1 = last_value + value * (series[i] - last_value)
                kama.append(kama_1)
                last_value = kama_1
            else:
                kama.insert(0, value)
        kama = pd.Series(kama, index=sc.index)
        kama.name = "Kaufman Adaptive Moving Average"
        return kama
    
    # def calculate_MAMA(
    #     self, series: pd.Series, n: int=10, 
    #     fast_lookback:int=5, slow_lookback: int=30
    # ):
    #     """
    #     Calculates the MESA adaptive moving average
    #     """
    #     ema = self.calculate_exponential_moving_average(series, n)
    #     pass

    @staticmethod
    def calculate_typical_price(df: pd.DataFrame) -> pd.Series:
        """
        Calculates the tyicpal (average) price over a given period
        """
        typical_price = (df.high + df.low + df.close) / 3
        typical_price.name = "Typical Price"
        return typical_price

    def calculate_vwap(self, price_df: pd.DataFrame, n: int=20) -> pd.Series:
        """
        Calculates the volume weighted average price
        """
        typical_price = self.calculate_typical_price(price_df)
        numerator = typical_price * price_df.volume
        rolling_numerator = numerator.rolling(n).sum()
        rolling_denominator = price_df.volume.rolling(n).sum()
        vwap = rolling_numerator / rolling_denominator
        vwap.name = "Volume Weighted Average Price"
        return vwap

    @staticmethod
    def calculate_simple_moving_sample_stdev(price_series: pd.Series, n: int=20) -> pd.Series:
        """
        Calculates the sample moving standard deviation
        """
        stdev = price_series.rolling(n).std()
        stdev.name = "Moving Standard Deviation"
        return stdev

    def calculate_macd_oscillator(
        self, series: pd.Series,
        n1: int=5, n2: int=34,
        ma1: str="SMA",
        ma2: str="SMA"
    ) -> pd.Series:
        """
        Calculate the moving average convergence divergence oscillator, given a 
        short moving average of length n1 and a long moving average of length n2
        """
        assert n1 < n2, f'n1 must be less than n2'
        moving_average_1 = self.get_ma_from_string(ma1)
        moving_average_2 = self.get_ma_from_string(ma2)
        macd = moving_average_1(series, n=n1) - \
            moving_average_2(series, n=n2)
        macd.name = "MACD Oscillator"
        return macd

    def calculate_stochastic_oscillator(
        self, series: pd.Series, n: int=14,
        ma: str="simple_moving_average"
    ) -> pd.Series:
        """
        Calculates the stochastic oscillator
        """
        high = series.rolling(n).max()
        low = series.rolling(n).min()
        ratio = (series - low) / (high - low)
        moving_average = self.get_ma_from_string(ma)
        stoch = moving_average(ratio, n=n)
        stoch.name = "Stochastic Oscillator"
        return stoch
    
    def calculate_relative_strength_index(
        self, series: pd.Series, n: int=20,
        ma: str="EMA"
    ) -> pd.Series:
        """
        Calculates the relative strength index
        """
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        moving_average = self.get_ma_from_string(ma)
        ma_up = moving_average(up, n=n)
        ma_down = moving_average(down, n=n)
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        rsi.name = "Relative Strength Index"
        return rsi

    def calculate_stochastic_rsi(
        self, series: pd.Series, n: int=20,
        ma: str="EMA"
    ) -> pd.Series:
        """
        Calculates the stochastic oscillation of the RSI
        """
        rsi = self.calculate_relative_strength_index(series, n=n, ma=ma)
        stoch_rsi = self.calculate_stochastic_oscillator(rsi, n=n, ma=ma)
        stoch_rsi.name = "Stochastic RSI Oscillator"
        return stoch_rsi

    @staticmethod
    def calculate_williams_r(series: pd.Series, n: int=14) -> pd.Series:
        """
        Calculates Williams %R
        """
        high = series.rolling(n).max()
        low = series.rolling(n).min()
        will = (high - series) / (high - low)
        will.name = "Williams %R"
        return will * 100
    
    @staticmethod
    def calculate_average_true_range(df: pd.DataFrame, n: int=14) -> pd.Series:
        """
        Calculates the average true range of a security
        TR is the greater of:
        the current high - current low, 
        current high - previous close, 
        or current low - previous close.
        """
        tr1 = df.high - df.low
        tr2 = df.high - df.close.shift(1)
        tr3 = df.low - df.close.shift(1)
        atr_df = pd.concat([tr1, tr2, tr3], axis=1)
        atr = atr_df.max(axis=1)
        window_sum = atr.rolling(n).sum()
        atr = window_sum - (window_sum / n) + atr
        atr.name = "Average True Range"
        return atr

    @staticmethod
    def calculate_plus_dm(df: pd.DataFrame) -> pd.Series:
        """
        Calculates the positive directional movement
        """
        plus_dm = df["high"].diff()
        plus_dm.name = "Positive Directional Movement"
        return plus_dm
    
    def calculate_smoothed_plus_dm(self, df: pd.DataFrame, n: int=14) -> pd.Series:
        """
        Calculates the smoothed positive directional movement
        """
        plus_dm = self.calculate_plus_dm(df)
        window_sum = plus_dm.rolling(n).sum()
        smoothed = window_sum - (window_sum / n) + plus_dm 
        smoothed.name = "Positive Smoothed Directional Movement"
        return smoothed
    
    def calculate_plus_di(self, df: pd.DataFrame, n: int=14) -> pd.Series:
        """
        Calculates the positive directional movement index (or indicator)
        """
        smoothed = self.calculate_smoothed_plus_dm(df, n)
        atr = self.calculate_average_true_range(df, n)
        plus_di = (smoothed / atr) * 100
        plus_di.name = "Positive Directional Movement Index"
        return plus_di

    @staticmethod
    def calculate_minus_dm(df: pd.DataFrame) -> pd.Series:
        """
        Calculates the negative directional movement
        """
        minus_dm = df["low"].diff()
        minus_dm.name = "Negative Directional Movement"
        return minus_dm
    
    def calculate_smoothed_minus_dm(self, df: pd.DataFrame, n: int=14) -> pd.Series:
        """
        Calculates the smoothed positive directional movement
        """
        minus_dm = self.calculate_minus_dm(df)
        window_sum = minus_dm.rolling(n).sum()
        smoothed = window_sum - (window_sum / n) + minus_dm 
        smoothed.name = "Negative Smoothed Directional Movement"
        return smoothed
    
    def calculate_minus_di(self, df: pd.DataFrame, n: int=14) -> pd.Series:
        """
        Calculates the Negative directional movement index (or indicator)
        """
        smoothed = self.calculate_smoothed_minus_dm(df, n)
        atr = self.calculate_average_true_range(df, n)
        minus_di = (smoothed / atr) * 100
        minus_di.name = "Negative Directional Movement Index"
        return minus_di

    def calculate_directional_index(self, df: pd.DataFrame, n: int=14) -> pd.Series:
        """
        Calculates the directional movement index
        """
        plus_di = self.calculate_plus_di(df, n)
        minus_di = self.calculate_minus_di(df, n)
        di = ((plus_di - minus_di) / (plus_di + minus_di)) * 100
        di.name = "Directional Index"
        return di

    def calculate_average_directional_index(self, df: pd.DataFrame, n: int=14) -> pd.Series:
        """
        Calculates the average directional movement index
        """
        adx = self.calculate_directional_index(df, n).rolling(n).mean()
        adx.name = "Average Directional Index"
        return adx

    def calculate_adxr(self, df: pd.DataFrame, adx_n: int=14, adxr_n: int=20) -> pd.Series:
        """
        Calculates the average directional movement index rating
        this metric serves to smooth the ADX
        """
        adx = self.calculate_average_directional_index(df, adx_n)
        adxr = self.calculate_momentum(adx, adxr_n) / 2
        adxr.name = "Average Directional Index Rating"
        return adxr

    def calculate_percentage_price_oscillator(self, series: pd.Series) -> pd.Series:
        """
        Calculates the percentage price oscillator of a series
        9 day ema - 26 day ema
        divided by 26 day ema
        """
        nine_day = self.calculate_exponential_moving_average(series, 9)
        long_ema = self.calculate_exponential_moving_average(series, 26)
        ppo = (nine_day - long_ema) / long_ema
        ppo.name = "Percentage Price Oscillator"
        return ppo

    def calculate_balance_of_power(self, df: pd.DataFrame, n: int=20) -> pd.DataFrame:
        """
        Calculates the "Balance of Power" between buyers and sellers
        sma of (close - open) / (high - low)
        """
        temp = (df.close - df.open) / (df.high - df.low)
        bop = self.calculate_SMA(temp, n)
        bop.name = "Balance of Power"
        return bop

    def calculate_commodity_channel_index(
        self, df: pd.DataFrame, n: int=20,
        ma: str="simple_moving_average"    
    ) -> pd.Series:
        """
        Calculates the commodity channel index of a single security
        (estimation)
        """
        typ = self.calculate_typical_price(df)
        sma = self.get_ma_from_string(ma)
        std = self.calculate_simple_moving_sample_stdev(typ, n)
        cci = (typ - sma(typ, n)) / (0.015 * std)
        cci.name = "Commodity Channel Index"
        return cci
    
    @staticmethod
    def calculate_chande_oscillator(df: pd.DataFrame, n: int=10) -> pd.Series:
        """
        Calculates the Chande oscillator
        """
        diff = df.close - df.open
        higher = diff.rolling(n).apply(lambda x: x[x > 0].sum())
        lower = diff.rolling(n).apply(lambda x: x[x < 0].sum())
        chande = ((higher - lower) / (higher + lower)) * 100
        chande.name = "Chande Momentum Oscillator"
        return chande

    def calculate_rate_of_change(self, series: pd.Series, n: int=5) -> pd.Series:
        """
        Calculates the rate of change of a security
        (price today - price n days ago) / price n days ago
        """
        mom = self.calculate_momentum(series, n)
        roc = (mom / series.shift(n)) * 100
        roc.name = "Rate of Change"
        return roc 
    

    class AroonHelper():
        """
        Helper class for Aroon indicator calculation.

        Handles rolling-window extrema tracking, which is awkward to express
        cleanly with standard pandas Series operations.
        """

        def __init__(self, series: pd.Series, n: int=25):
            """
            Initializes AroonHelper with a price series and lookback period.

            Args:
                series (pd.Series): Date-indexed price series.
                n (int): Lookback period in days. Defaults to 25.
            """
            self.series = series
            self._series_length = series.shape[0]
            self.index = series.index
            self.n = n

        def get_maxidx(self) -> float:
            """
            Calculates the max in rolling period and updates
            distance_from_max_attribute
            """
            self.max_to_df = {"max" : [], "distance_to_max" : []}
            roll = self.series.rolling(self.n).apply(self._get_extremes, args=("max",))
            self.max_df = self._make_df(self.max_to_df)
            return 0
        
        def get_minidx(self) -> float:
            """
            Calculates the min in rolling period and updates
            distance_from_max_attribute
            """
            self.min_to_df = {"min" : [], "distance_to_min" : []}
            roll = self.series.rolling(self.n).apply(self._get_extremes, args=("min",))
            self.min_df = self._make_df(self.min_to_df)
            return 0
        
        def _get_extremes(self, sub_series: pd.Series, extreme_type: str) -> tuple:
            """
            Calculates the max and distance back to max and
            returns it as a data frame
            """
            length = sub_series.shape[0]
            extreme_func = getattr(pd.Series, extreme_type)
            extreme = extreme_func(sub_series)
            getattr(self, f"{extreme_type}_to_df")[extreme_type].append(extreme)
            for i, value in enumerate(sub_series):
                if value == extreme:
                    distance = length - i
                    getattr(self, f"{extreme_type}_to_df")[f"distance_to_{extreme_type}"].append(distance)
                    break
            return 0
        
        def _make_df(self, dictionary: dict) -> pd.DataFrame:
            """
            Converts the dictionary made in min and maxidx functions into pandas dataframe
            """
            current = len(dictionary[list(dictionary.keys())[0]])
            to_add = self._series_length -  current
            if to_add > 0:
                for i in range(to_add):
                    for key in list(dictionary.keys()):
                        dictionary[key].insert(0, np.nan)
            return pd.DataFrame(dictionary, index = self.index)


    def calculate_aroon_up(self, series: pd.Series, n: int=25) -> pd.Series:
        """
        Calculates the Aroon up indicator
        """
        helper = self.AroonHelper(series, n)
        helper.get_maxidx()
        aroon_values = helper.max_df
        aroon_up = ((n - aroon_values.distance_to_max) / n) * 100
        aroon_up.name = "Aroon Up"
        return aroon_up
    
    def calculate_aroon_down(self, series: pd.Series, n: int=25) -> pd.Series:
        """
        Calculates the Aroon down indicator
        """
        helper = self.AroonHelper(series, n)
        helper.get_minidx()
        aroon_values = helper.min_df
        aroon_down = ((n - aroon_values.distance_to_min) / n) * 100
        aroon_down.name = "Aroon Down"
        return aroon_down
    
    def calculate_aroon_oscillator(self, series: pd.Series, n: int=25) -> pd.Series:
        """
        Calculates the aroon oscilator (the difference between the aroon up and aroon down)
        """
        aroon_up = self.calculate_aroon_up(series, n)
        aroon_down = self.calculate_aroon_down(series, n)
        aroon_osc = aroon_up - aroon_down
        aroon_osc.name = "Aroon Oscillator"
        return aroon_osc

    def calculate_bollinger_bands(
        self, series: pd.Series, 
        n: int=20
    ) -> pd.DataFrame:
        """
        Calculates the bollinger bands and returns them as a dataframe
        """
        sma = self.calculate_SMA(series, n=n)
        stdev = self.calculate_simple_moving_sample_stdev(series, n=n)

        return pd.DataFrame({
            'middle': sma,
            'upper': sma + 2 * stdev,
            'lower': sma - 2 * stdev
        })

    def calculate_money_flow(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates the money flow for the money flow index
        """
        typical = self.calculate_typical_price(df)
        money_flow = typical * df.volume
        money_flow.name = "Money FLow"
        return money_flow

    def calculate_pos_neg_money_flows(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates the series contining the positive and negative money flows
        """
        index = df.index
        money_flow = self.calculate_money_flow(df)
        diff = money_flow.diff()
        negative = []
        positive = []
        for value in diff:
            if value > 0:
                negative.append(0)
                positive.append(value)
            elif value < 0:
                negative.append(abs(value))
                positive.append(0)
            else:
                negative.append(np.nan)
                positive.append(np.nan)
        return pd.Series(positive, index=index), pd.Series(negative, index=index)
            
    def calculate_money_flow_ratio(self, df: pd.DataFrame, n: int=10) -> pd.Series:
        """
        Calculates the money flow ratio
        pos money flow / negative money flow
        """
        positive, negative = self.calculate_pos_neg_money_flows(df)
        pmf = positive.rolling(n).sum()
        nmf = negative.rolling(n).sum()
        money_ratio = pmf / nmf
        money_ratio.name = "Money Ratio"
        return money_ratio
    
    def calculate_money_flow_index(self, df: pd.DataFrame, n: int=10) -> pd.Series:
        """
        Calculate the money flow index
        100 - (100 / 1 + money flow ratio)
        """
        ratio = self.calculate_money_flow_ratio(df, n)
        mfi = 100 - (100 / (1 + ratio))
        mfi.name = "Money Flow Index"
        return mfi

    @staticmethod
    def calculate_money_flow_volume_series(df: pd.DataFrame) -> pd.Series:
        """
        Calculates money flow series for the chaikin money flow
        """
        mfv = df['volume'] * (2*df['close'] - df['high'] - df['low']) / \
                                        (df['high'] - df['low'])
        mfv.name = "Money Flow Volume"
        return mfv

    def calculate_money_flow_volume(self, df: pd.DataFrame, n: int=20) -> pd.Series:
        """
        Calculates money flow volume, or q_t in our formula
        """
        return self.calculate_money_flow_volume_series(df).rolling(n).sum()

    def calculate_chaikin_money_flow(self, df: pd.DataFrame, n: int=20) -> pd.Series:
        """
        Calculates the Chaikin money flow
        """
        chaikin = self.calculate_money_flow_volume(df, n=n) / df['volume'].rolling(n).sum()
        chaikin.name = "Chaikin Money Flow"
        return chaikin
    
    def calculate_trix(self, series: pd.Series, n: int=10) -> pd.Series:
        """
        Calculates the rate of change of the triple exponential moving average (TRIX)
        """
        tema = self.calculate_TEMA(series, n)
        trix = self.calculate_rate_of_change(tema, n)
        trix.name = "ROC of TEMA"
        return trix
    
    class UltOscHelper():
        """
        Helper class for Ultimate Oscillator calculation.

        Encapsulates the per-period 'a' (buying pressure) and 'b' (true range)
        computations used in the Ultimate Oscillator formula.
        """

        def __init__(self, ma_function: Callable):
            """
            Initializes UltOscHelper with a moving average function.

            Args:
                ma_function (Callable): Moving average function used in a/b calculations
                                        (e.g., Indicators.calculate_SMA).
            """
            self.ma_func = ma_function
        
        def find_next_a(self) -> str:
            """
            uses the sma_to_weigh dictionary to label next dictionary item
            a1, a2, a3 etc
            """
            try:
                last_key = list(self.sma_to_weigh.keys())[-1]
            except KeyError:
                return "a1"
            last_key.replace("n", "")
            num = int(last_key)
            num += 1
            new_key = "a" + num
            return new_key

        def calculate_a(self, df: pd.DataFrame, n: int) -> pd.Series:
            """
            Calculates the buying pressure component (a) for the Ultimate Oscillator.

            a = n-period moving average of (close - true_low)

            Args:
                df (pd.DataFrame): OHLCV price DataFrame.
                n (int): Lookback period.

            Returns:
                pd.Series: Buying pressure series.
            """
            low1 = df.close.shift(1)
            low2 = df.low 
            low_df = pd.concat([low1, low2], axis=1)
            interior = df.close - low_df.min(axis=1)
            a = self.ma_func(interior, n)
            return a
            
        def calculate_b(self, df: pd.DataFrame, n: int) -> pd.Series:
            """
            Calculates the true range component (b) for the Ultimate Oscillator.

            b = n-period moving average of the true range

            Args:
                df (pd.DataFrame): OHLCV price DataFrame.
                n (int): Lookback period.

            Returns:
                pd.Series: True range moving average series.
            """
            trmax = pd.concat([df.high, df.close.shift(1)], axis=1)
            trmin = pd.concat([df.low, df.close.shift(1)], axis=1)
            tr = trmax.max(axis=1) - trmin.min(axis=1)
            b = self.ma_func(tr, n)
            return b

    def calculate_ultimate_oscillator(
        self, df: pd.DataFrame, 
        n1: int=7, n2: int=14, n3: int=28
    ) -> pd.Series:
        """
        Calculates the ultimate oscillator (google it lol)
        not sure if this is right also (it should be between 0 and 100)
        """
        helper = self.UltOscHelper(self.calculate_SMA)
        a1 = helper.calculate_a(df, n1)
        a2 = helper.calculate_a(df, n2)
        a3 = helper.calculate_a(df, n3)
        b1 = helper.calculate_b(df, n1)
        b2 = helper.calculate_b(df, n2)
        b3 = helper.calculate_b(df, n3)
        w1 = (a1 / b1) * ((4 / 7) * n1)
        w2 = (a2 / b2) * ((2 / 7) * n1)
        w3 = (a3 / b3) * ((1 / 7) * n1)
        numerator = w1 + w2 + w3
        osc = (numerator / n1) * 100
        osc.name = "Ultimate Oscillator"
        return osc
    
    @staticmethod
    def calculate_midpoint(df: pd.DataFrame) -> pd.Series:
        """
        Calculates midpoint (like typical value) high - low / 2
        """
        midpoint = (df.high + df.low) / 2
        return midpoint
    
    @staticmethod
    def calculate_on_balance_volume(df: pd.DataFrame) -> pd.Series:
        """
        Calculates the on-balance volume (cumulative running volume)
        volume added if price higher and subtracted if price lower
        """
        index = df.index
        obv = []
        last_close_series = df.close.shift(1)
        close_series = df.close
        for i, value in enumerate(df.volume):
            close = close_series.at[index[i]]
            last_close = last_close_series.at[index[i]]
            if i == 0:
                obv.append(value)
                continue
            if close < last_close:
                to_add = -value
            elif close > last_close:
                to_add = value
            else:
                to_add = 0
            obv.append(obv[i-1] + to_add)
        obv.name = "On Balance Volume"
        return pd.Series(obv, index=index)


class Signals(Indicators):
    """
    Trading signal generators built on all Indicators.

    Produces buy/sell signal Series where 1 = buy, -1 = sell, 0 = hold.
    Signals are derived from indicator crossovers, zero-crossings, and
    range-exceeding logic. All indicators from Indicators are available
    as signal sources.
    """

    def __init__(self):
        """
        Initializes Signals, calling Indicators.__init__ via super().
        """
        super().__init__()
    
    @staticmethod
    def create_zero_crossing_signals(series: pd.Series) -> pd.Series:
        """
        Creates buy and sell signals for an oscillator where changing from positive to negative
        or vice versa generates a signal
        """
        sign = np.sign(series) 
        # Create a copy shifted by some amount.
        last_sign = np.sign(series).shift(1, axis=0)
        # Multiply by the sign by the boolean. This will have the effect of casting
        # the boolean to an integer (either 0 or 1) and then multiply by the sign
        # (either -1, 0 or 1).
        signals = sign * (sign != last_sign)
        return signals
    
    @staticmethod
    def create_range_exceeding_signals(
        price_series: pd.Series,
        range_df: pd.DataFrame,
        mean_reversion:bool=True
    ) -> pd.Series:
        """
        Creates signals of price series crossing above or below a given range
        data frame with columns named ["upper", "lower"] in range_df arg    
        """
        sell = price_series > range_df['upper']
        buy = price_series < range_df['lower']
        signals = (1*buy - 1*sell)
        if not mean_reversion:
            signals = -1 * signals
        signals.name = "Range Exceeding Signals"
        return signals
    
    @staticmethod
    def create_static_range_exceeding_signals(
        indicator_series: pd.Series,
        upper_bound: int, lower_bound: float,
        mean_reversion:bool=True
    ) -> pd.Series:
        """
        Creates signals of price series crossing above or below a given range
        data frame with columns named ["upper", "lower"] in range_df arg    
        """
        sell = indicator_series > upper_bound
        buy = indicator_series < lower_bound
        signals = (1*buy - 1*sell)
        if not mean_reversion:
            signals = -1 * signals
        signals.name = "Range Exceeding Signals"
        return signals

    def create_indicator_crossover_signals(
        self, price_series: pd.Series, 
        indicator_series: pd.Series,
        inverse: bool=False
    ) -> pd.Series:
        """
        Creates signals for when a price series crosses over a indicator series
        specifically, generates buy signal when price crosses from below to above indicator
        and sells when price crosses from above to below. can be reversed by changing inverse arg to True
        """
        difference = price_series - indicator_series
        signals = self.create_zero_crossing_signals(difference)
        if inverse:
            signals = -1*signals
        return signals
    
    def create_MA_signals(
        self, series: pd.Series, n: int=14,
        ma_type: str="simple_moving_average"
    ) -> pd.Series:
        """
        Creates signals based on crossover of one of the MA and the current price
        supported moving averages 
            "simple_moving_average",
            "triangular_moving_average",
            "weighted_moving_average",
            "exponential_moving_average",
            "DEMA", "TEMA"
        """
        ma_func = self.get_ma_from_string(ma_type)
        ma = ma_func(series, n)
        signals = self.create_indicator_crossover_signals(series, ma)
        signals.name = "Moving Average Signals"
        return signals
    
    def create__signals(self, series: pd.Series, n: int=14) -> pd.Series:
        """
        Creates signals based on the 
        """
        mom = self.calculate_momentum(series, n)
        signals = self.create_zero_crossing_signals(mom)
        signals.name = " Signals"
        return signals
        
    # MESA adaptive moving average (MAMA)
    # average true range (ATR)
    # plus, minus directional movement
    # plus, minus directional index
    # average directional indes (ADX)
    # directional index (DI)
    # ADX ratio (ADXR)
    # percentage price index (PPI)
    # balance of power (BOP)
    # commodity channel index (CCI)
    # Chande oscillator
    # rate of change (ROC)
    # Aroon up, down, oscillator 
    # Bollinger bands (BBANDS)
    # money flow index (MFI)
    # money flow volume (MFV)
    # Chaikin money flow (CMF)
    # ROC of TEMA (TRIX)
    # ultimate oscillator
    # midpoint (MID)
    # on-balance volume (OBV)

    def create_momentum_signals(self, series: pd.Series, n: int=14) -> pd.Series:
        """
        Creates signals based on the momentum crossover principle
        if momentum > 0, price accelerating upwards == buy
        if momentum < 0, price accelerating downards == self
        """
        mom = self.calculate_momentum(series, n)
        signals = self.create_zero_crossing_signals(mom)
        signals.name = "Momentum Signals"
        return signals

    def create_KAMA_signals(self, price_series: pd.Series, n: int=10, 
        fast_lookback:int=5, slow_lookback: int=30
    ) -> pd.Series:
        """
        Creates buy and sell signals using simple crossover of Kaufmans adaptive
        moving average
        """
        kama = self.calculate_KAMA(price_series, n, fast_lookback, slow_lookback)
        signals = self.create_indicator_crossover_signals(price_series, kama)
        signals.name = "Kaufmans Adaptive Moving Average Signals"
        return signals

    def create_vwap_signals(self, price_df: pd.DataFrame) -> pd.Series:
        """
        Creates signals for vwap indicator, when price goes above vwap triggers sell signal
        and vice versa
        """
        vwap = self.calculate_vwap(price_df)
        signals = self.create_indicator_crossover_signals(price_df["close"], vwap, inverse=True)
        signals.name = "Volume Weighted Average Price Crossover Signals"
        return signals

    def create_macd_signals(self, price_series: pd.Series, n1: int=5, n2: int=34) -> pd.Series:
        """ 
        Create a momentum-based signal based on the MACD crossover principle. 
        Generate a buy signal when the MACD cross above zero, and a sell signal when
        it crosses below zero.
        """
        # Calculate the macd and get the signs of the values.
        macd = self.calculate_macd_oscillator(price_series, n1, n2)
        macd_signals = self.create_zero_crossing_signals(macd)
        macd_signals.name = "MACD Signals"
        return macd_signals

    def create_stochastic_oscillator_signals(
        self, price_series: pd.Series, 
        n: int=14, upper_bound: int=80, lower_bound: int=20,
        ma_type: str="simple_moving_average"
     ) -> pd.Series:
        """
        Creates range exceeding signals for stochasitc oscillator
        """
        oscillator = self.calculate_stochastic_oscillator(price_series, n=n, ma=ma_type)
        signals = self.create_static_range_exceeding_signals(oscillator*100, upper_bound, lower_bound)
        signals.name = "Stochasitc Oscillator Signals"
        return signals

    def create_relative_strength_index_signals(
        self, price_series: pd.Series, 
        n: int=14, upper_bound: int=70, lower_bound: int=30,
        ma_type: str="simple_moving_average"
     ) -> pd.Series:
        """
        Creates range exceeding signals for relative strength index
        """
        oscillator = self.calculate_relative_strength_index(price_series, n=n, ma=ma_type)
        signals = self.create_static_range_exceeding_signals(oscillator, upper_bound, lower_bound)
        signals.name = "Relative Strength Index Signals"
        return signals
    
    def create_stochastic_rsi_signals(
        self, price_series: pd.Series, 
        n: int=14, upper_bound: int=80, lower_bound: int=20,
        ma_type: str="simple_moving_average"
     ) -> pd.Series:
        """
        Creates range exceeding signals for stochasitc rsi
        """
        oscillator = self.calculate_stochastic_rsi(price_series, n=n, ma=ma_type)
        signals = self.create_static_range_exceeding_signals(oscillator*100, upper_bound, lower_bound)
        signals.name = "Relative Strength Index Signals"
        return signals
    
    def create_williams_r_signals(
        self, price_series: pd.Series, 
        n: int=14, upper_bound: int=80, lower_bound: int=20,
     ) -> pd.Series:
        """
        Creates range exceeding signals for williams %r
        """
        oscillator = self.calculate_williams_r(price_series, n=n)
        signals = self.create_static_range_exceeding_signals(oscillator, upper_bound, lower_bound)
        signals.name = "Relative Strength Index Signals"
        return signals

    def create_bollinger_band_signals(
        self, price_series: pd.Series, 
        n: int=20
    ) -> pd.Series:
        """
        Create a reversal-based signal based on the upper and lower bands of the 
        Bollinger bands. Generate a buy signal when the price is below the lower 
        band, and a sell signal when the price is above the upper band.
        """
        bollinger_bands = self.calculate_bollinger_bands(price_series, n=n)
        boll = self.create_range_exceeding_signals(price_series, bollinger_bands)
        boll.name = "Bollinger Band Signals"
        return boll
    
    @staticmethod
    def show_indicator(
        price_series: pd.Series=pd.Series(),
        indicator_series: pd.Series=pd.Series(),
        signal_series: pd.Series=pd.Series(),
        combine_price_and_indicator: bool=True
    ) -> None:
        """
        Displays a pyplot graph of a price series, indicator, and/or signal series.

        Args:
            price_series (pd.Series): Date-indexed price series.
            indicator_series (pd.Series): Date-indexed indicator series.
            signal_series (pd.Series): Date-indexed signal series (1/-1/0).
            combine_price_and_indicator (bool): If True, overlays price and indicator
                on one subplot with signals below. Raises AssertionError if price and
                indicator are not both provided when True.
        """
        sets = [price_series, indicator_series, signal_series]
        empty_series = [not series.empty for series in sets]
        empty_series_count = empty_series.count(True)
        
        if combine_price_and_indicator:
            no_signals = [True, True, False]
            with_signals = [True, True, True]
            assert empty_series == no_signals or empty_series == with_signals,\
                "Must include price and indicator series to enable combine_price_and_indicator"
            if empty_series == no_signals:
                plt.plot(price_series)
                plt.plot(indicator_series)
                plt.show()
                return
            else:
                fig, axs = plt.subplots(2, sharex=True)
                axs[0].plot(price_series)
                axs[0].plot(indicator_series)
                axs[1].plot(signal_series)
                plt.show()
                return
        elif empty_series_count == 1:
            for series in sets:
                if not series.empty:
                    plt.plot(series)
            plt.show()
            return
        else:
            fig, axs = plt.subplots(empty_series_count, sharex=True)
            i = 0
            for series in sets:
                if not series.empty:
                    axs[i].plot(series)
                    i += 1
            if i >= 1:
                plt.show()
                return


# signals testing
if __name__ == "__main__":
    import pprint
    import cProfile
    from data_loading import load_SPY_data, load_data_as_pd
    SPY = load_SPY_data()
    AWU = load_data_as_pd("AWU")["close"]
    signals = Signals()
    cProfile.run("signals.calculate_WMA(AWU)")
    # cProfile.run("signals.create_MA_signals(AWU, n=20, ma_type='WMA')")
    # pprint.pprint(signal_series)
    # pprint.pprint(indicator_series)
    # signals.show_indicator(AWU, indicator_series, signal_series, combine_price_and_indicator=False)