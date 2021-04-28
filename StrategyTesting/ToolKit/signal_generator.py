#!python3


from ToolKit.data_loading import load_SPY_data, load_data_as_pd
import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Dict, Any, Callable
from sklearn.linear_model import LinearRegression


# Metrics - carries class with various stock performance matrics
# based on price variations that attempt to quantify performance
# in relation to risk or to a baseline market.
#
# supported metrics:
# log or percent return series,
# annualized volatility (%)
# compounded annual growth rate (CAGR)
# annualized downside deviation
# Sharpe ratio
# Rolling sharpe ratio
# Sortino ratio
# drawdown series
# max drawdown (with metadata and as log if desired)
# Calmar ratio
# pure profit score
# alpha


class Metrics():
    def __init__(self):
        self.return_series_types = ["log", "percent"]
        self.DRAWDOWN_EVALUATORS: Dict[str, Callable] = {
            'dollar': lambda price, peak: peak - price,
            'percent': lambda price, peak: -((price / peak) - 1),
            'log': lambda price, peak: np.log(peak) - np.log(price),
        }

    @staticmethod
    def calculate_return_series(series: pd.Series) -> pd.Series:
        """
        Calculates the return series of a time series.
        The first value will always be NaN.
        Output series retains the index of the input series.
        """
        shifted_series = series.shift(1, axis=0)
        return series / shifted_series - 1
    
    @staticmethod
    def calculate_percent_return(series: pd.Series) -> float:
        return series.iloc[-1] / series.iloc[0] - 1
    
    @staticmethod
    def calculate_log_return_series(series: pd.Series) -> pd.Series:
        """
        Same as calculate_return_series but with log returns
        """
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
        """
        Calculates the sortino ratio.
        """
        cagr = self.calculate_cagr(price_series)
        return_series = self.calculate_return_series(price_series)
        downside_deviation = self.calculate_annualized_downside_deviation(return_series)
        return (cagr - benchmark_rate) / downside_deviation

    def calculate_drawdown_series(self, series: pd.Series, method: str='log') -> pd.Series:
        """
        Returns the drawdown series
        """
        assert method in self.DRAWDOWN_EVALUATORS, \
            f'Method "{method}" must by one of {list(self.DRAWDOWN_EVALUATORS.keys())}'

        evaluator = self.DRAWDOWN_EVALUATORS[method]
        return evaluator(series, series.cummax())

    def calculate_max_drawdown(self, series: pd.Series, method: str='log') -> float:
        """
        Simply returns the max drawdown as a float
        """
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
        log_drawdown = self.calculate_max_drawdown(series, method='log')
        log_return = np.log(series.iloc[-1]) - np.log(series.iloc[0])
        return log_return - log_drawdown

    def calculate_calmar_ratio(self, series: pd.Series, years_past: int=3) -> float:
        """
        Return the percent max drawdown ratio over the past three years using
        CAGR as the numerator, otherwise known as the Calmar Ratio
        """

        # Filter series on past three years
        last_date = series.index[-1]
        three_years_ago = last_date - pd.Timedelta(days=years_past*365.25)
        series = series[series.index > three_years_ago]

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
    ) -> float:
        mask = ~np.isnan(return_series) & ~np.isnan(benchmark_return_series)
        return stats.linregress(benchmark_return_series[mask], return_series[mask])

    def calculate_beta(
        self, return_series: pd.Series,
        benchmark_return_series: pd.Series
    ) -> float:
        """
        Calculates the beta of a portfolio or stock return series
        versus a benchmark
        """
        return self._get_linreg(return_series, benchmark_return_series).slope
    
    def calculate_alpha(
        self, return_series: pd.Series,
        benchmark_return_series: pd.Series
    ) -> float:
        """
        Calculates the alpha of a portfolio or stock return series
        versus a benchmark
        """
        return self._get_linreg(return_series, benchmark_return_series).intercept


# Indicators - creates class that calculates various 
# market technical indicators that indicate at what stage in
# a cycle the security may be in or how it is performing relative
# its past or average performance.
#
# supported inidcators:
#
# momentum (MOM)
# simple moving average (SMA)
# triangular moving average (TMA)
# weighted moving average (WMA)
# exponential, double exponential, triple exponential moving average (EMA, DEM, TEMA)
# Kaufman adaptive moving average (KAMA)
# MESA adaptive moving average (MAMA)
# typical price 
# volume weighted average price (VWAP)
# simple moving standard deviation 
# moving average convergence divergence oscillator (MACD)
# stochastic oscillator (STOCH)
# relative strength index (RSI)
# stochastic RSI
# williams %r 
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



class Indicators(Metrics):
    def __init__(self):
        super().__init__()
    
    def get_ma_from_string(self, string):
        compatible = [
            "simple_moving_average",
            "triangular_moving_average",
            "weighted_moving_average",
            "exponential_moving_average",
            "DEMA", "TEMA"
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
    def calculate_simple_moving_average(series: pd.Series, n: int=20) -> pd.Series:
        """
        Calculates the simple moving average
        """
        sma = series.rolling(n).mean()
        sma.name = "Simple Moving Average"
        return sma
    
    def calculate_triangular_moving_average(self, series: pd.Series, n: int=20) -> pd.Series:
        """
        Calculates the triangular moving average
        """
        sma = self.calculate_simple_moving_average(series, n)
        tma = self.calculate_simple_moving_average(sma, n)
        tma.name = "Triangular Moving Average"
        return tma
    
    @staticmethod
    def calculate_weighted_moving_average(series: pd.Series, n: int=20) -> pd.Series:
        """
        Calculates weighted moving average of a data set
        """
        denominator = int((n * (n+1)) / 2)
        weights = [i + 1 for i in range(n)]
        wma = series.rolling(n).apply(lambda x: np.sum(weights * x) / denominator)
        wma.name = "Weighted Moving Average"
        return wma
    
    @staticmethod
    def calculate_exponential_moving_average(series: pd.Series, n: int=20) -> pd.Series:
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
        ema = self.calculate_exponential_moving_average(series, n)
        ema_of_ema = self.calculate_exponential_moving_average(ema, n)
        dema = (2 * ema) - ema_of_ema
        dema.name = "Double Exponential Moving Average"
        return dema
    
    def calculate_TEMA(self, series: pd.Series, n: int=20) -> pd.Series:
        """
        Calculates the triple exponential moving average
        """
        ema_1 = self.calculate_exponential_moving_average(series, n)
        ema_2 = self.calculate_exponential_moving_average(ema_1, n)
        ema_3 = self.calculate_exponential_moving_average(ema_2, n)
        tema = (3 * ema_1) - (3 * ema_2) + ema_3
        tema.name = "Triple Exponential Moving Average"
        return tema

    def calculate_KAMA(
        self, series: pd.Series, n: int=10, 
        fast_lookback:int=5, slow_lookback: int=30
    ):
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
        kama = [self.calculate_simple_moving_average(series, n).dropna()[0]]
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

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates the volume weighted average price
        """
        typical_price = self.calculate_typical_price(df)
        numerator = typical_price * df.volume
        vwap = numerator.cumsum() / df.volume.cumsum()
        vwap.name = "Volume Weighted Average Price"
        return vwap

    @staticmethod
    def calculate_simple_moving_sample_stdev(series: pd.Series, n: int=20) -> pd.Series:
        """
        Calculates the sample moving standard deviation
        """
        stdev = series.rolling(n).std()
        stdev.name = "Moving Standard Deviation"
        return stdev

    def calculate_macd_oscillator(
        self, series: pd.Series,
        n1: int=5, n2: int=34,
        ma1: str="simple_moving_average",
        ma2: str="simple_moving_average"
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
        ma: str="exponential_moving_average"
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
        ma: str="exponential_moving_average"
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
        bop = self.calculate_simple_moving_average(temp, n)
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
        Class to help calculate aroon indicator due to issues
        with pandas.series
        """
        def __init__(self, series: pd.Series, n: int=25):
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
        Calculates the aroon oscilator ( the difference between the aroon up and aroon down)
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
        sma = self.calculate_simple_moving_average(series, n=n)
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
        def __init__(self, ma_function):
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

        def calculate_a(self, df: pd.DataFrame, n: int):
            """
            a = n period sma of (close - true_low) * n
            """
            low1 = df.close.shift(1)
            low2 = df.low 
            low_df = pd.concat([low1, low2], axis=1)
            interior = df.close - low_df.min(axis=1)
            a = self.ma_func(interior, n)
            return a
            
        def calculate_b(self, df: pd.DataFrame, n: int):
            """
            b = n period sma of true range
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
        helper = self.UltOscHelper(self.calculate_simple_moving_average)
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


# Signals - class that inherits from indicators and creates buy and sell
# signals based on the indicators
# all indicators listed in comment Indicators class are supported


class Signals(Indicators):
    def __init__(self):
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
        series: pd.Series, 
        upper_range: pd.Series, 
        lower_range: pd.Series
    ) -> pd.Series:
        """

        """
    
    def create_indicator_crossover_signals(
        self, price_series: pd.Series, 
        indicator_series: pd.Series
    ) -> pd.Series:
        difference = price_series - indicator_series
        signals = self.create_zero_crossing_signals(difference)
        return signals

    def create_momentum_signal(self, series: pd.Series, n: int=14) -> pd.Series:
        """
        Creates signals based on the momentum crossover principle
        if momentum > 0, price accelerating upwards == buy
        if momentum < 0, price accelerating downards == self
        """
        mom = self.calculate_momentum(series, n)
        signals = self.create_zero_crossing_signals(mom)
        signals.name = "Momentum Signals"
        return signals
    
    def create_MA_signal(
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
        
    # Kaufman adaptive moving average (KAMA)
    # MESA adaptive moving average (MAMA)
    # typical price 
    # volume weighted average price (VWAP)
    # simple moving standard deviation 
    # moving average convergence divergence oscillator (MACD)
    # stochastic oscillator (STOCH)
    # relative strength index (RSI)
    # stochastic RSI
    # williams %r 
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

    def create_macd_signal(self, series: pd.Series, n1: int=5, n2: int=34) -> pd.Series:
        """ 
        Create a momentum-based signal based on the MACD crossover principle. 
        Generate a buy signal when the MACD cross above zero, and a sell signal when
        it crosses below zero.
        """
        # Calculate the macd and get the signs of the values.
        macd = self.calculate_macd_oscillator(series, n1, n2)
        macd_signals = self.create_zero_crossing_signal(macd)
        macd_signals.name = "MACD Signals"
        return macd

    def create_bollinger_band_signal(
        self, series: pd.Series, 
        n: int=20
    ) -> pd.Series:
        """
        Create a reversal-based signal based on the upper and lower bands of the 
        Bollinger bands. Generate a buy signal when the price is below the lower 
        band, and a sell signal when the price is above the upper band.
        """
        bollinger_bands = self.calculate_bollinger_bands(series, n=n)
        sell = series > bollinger_bands['upper']
        buy = series < bollinger_bands['lower']
        boll = (1*buy - 1*sell)
        boll.name = "Bollinger Band Signals"
        return boll


# signals testing
if __name__ == "__main__":
    import matplotlib as plt
    SPY = load_SPY_data()["close"]
    AWU = load_data_as_pd("AWU")["close"]
    signals = Signals()
    print(list(signals.create_bollinger_band_signal(AWU, n=5)))