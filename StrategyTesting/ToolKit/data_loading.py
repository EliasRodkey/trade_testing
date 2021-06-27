#! python3
# 
# data_loading.py contains functions associated with calculating
# metrics and indicators of a set of chronical financial price data
# as well as classes used in creating a financial market simulation 
# to test trading strategies (also called backtesting) and evaluate
# performance 

import os
import pandas as pd
from typing import Dict, List


EOD_DATA_DIR = os.path.join("StrategyTesting", "test_data", "eod")
SPY_PATH  = os.path.join("StrategyTesting", "test_data", "SPY.csv")
RESULTS_PATH = os.path.join("StrategyTesting", "optimization_results")


def load_data_as_pd(symbol: str) -> pd.DataFrame:
    """Loads data from test_data\\eod stock with input symbol"""
    df = pd.read_csv(os.path.join(EOD_DATA_DIR, f"{symbol}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df

def load_SPY_data():
    """Loads data for SPY (S&P 500) in test_data"""
    df = pd.read_csv(SPY_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df

def get_all_symbols():
    files = os.listdir(EOD_DATA_DIR)
    s = [item.strip(".csv") for item in files]
    return s

def _combine_columns(filepaths_by_symbol: Dict[str, str], 
    column: str='close') -> pd.DataFrame:

    data_frames = [
        pd.read_csv(
            filepath, 
            index_col='date', 
            usecols=['date', column], 
            parse_dates=['date'],
        ).rename(
            columns={
                'date': 'date', 
                column: symbol,
            }
        ) for symbol, filepath in filepaths_by_symbol.items()
    ]
    return pd.concat(data_frames, sort=True, axis=1)    

def load_eod_matrix(tickers: List[str], column: str='close') -> pd.DataFrame:
    filepaths_by_symbol = {
        t: os.path.join(EOD_DATA_DIR, f'{t}.csv') for t in tickers
    }
    return _combine_columns(filepaths_by_symbol, column)
    
def concatenate_metrics(df_by_metric: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates different dataframes that have the same columns into a
    hierarchical dataframe.

    The input df_by_metric should of the form

    {
        'metric_1': pd.DataFrame()
        'metric_2: pd.DataFrame()
    }
    where each dataframe should have the same columns, i.e. symbols.
    """

    to_concatenate = []
    tuples = []
    for key, df in df_by_metric.items():
        to_concatenate.append(df)
        tuples += [(s, key) for s in df.columns.values]

    df = pd.concat(to_concatenate, sort=True, axis=1)
    df.columns = pd.MultiIndex.from_tuples(tuples, names=['symbol', 'metric'])

    return df
    
def list_results_files() -> list:
    # walks the optimization_results directory and returns the simset ids in a list
    for _, _, filenames in os.walk(RESULTS_PATH):
        return filenames
    
def find_unmerged_files(results_files: list, as_filename: bool=True, as_path: bool=False) -> list:
    # iterates through results files names list and returns those containing 
    # maxpos element
    if as_filename == False:
        as_path=False
    unmerged = []
    for filename in results_files:
        if "maxpos".upper() in filename:
            unmerged.append(filename)
    
    if not as_filename:
        for i, filename in enumerate(unmerged):
            unmerged[i] = filename.replace(".csv", "")
    if as_path:
        for i, filename in enumerate(unmerged):
            file_path = os.path.join(RESULTS_PATH, filename)
            unmerged[i] = file_path
    return unmerged

def extract_id(unmerged_simset_id: str) -> str:
    pass

def group_simsets():
    pass


# laoding data usage
if __name__ == "__main__":
    simset_ids = list_results_files()
    to_merge = find_unmerged_files(simset_ids, as_path=True)
    print(to_merge)
