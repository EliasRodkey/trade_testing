#! python3
#
# api_tool_kit.py - hold classes dictating interactions with 
# information gathering APIs
#
# used APIs:
# Alpha Vantage (5 calls per minute maximum, 500 calls per day maximum)
# Alpaca

# # gets Alpha Vantage API Key
# import subprocess, locale, os
# remote_path = os.path.abspath(os.path.join(
#     os.sep, "runable_scripts", 
#     "Applications", "Pandle", 
#     "remote_pandle.py"
# ))
# proccess_obj = subprocess.run(["python", remote_path, "alpha_vantage"], stdout=subprocess.PIPE)
# my_key = proccess_obj.stdout.decode(locale.getdefaultlocale()[1])

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.cryptocurrencies import CryptoCurrencies

time_series = TimeSeries(output_format="pandas")
technical_indicators = TechIndicators(output_format="pandas")
fundamentals = FundamentalData(output_format="pandas")
foreign_exchange = ForeignExchange(output_format="pandas")
crypto = CryptoCurrencies(output_format="pandas")