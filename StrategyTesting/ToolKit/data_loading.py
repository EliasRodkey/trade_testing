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
from send2trash import send2trash



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
    

def list_results_dirs() -> list:
    # creates a list of all of the relative paths to the subfolders
    # in "opimization_results" that house results
    for folder, subfolders, filenames in os.walk(RESULTS_PATH):
        return subfolders

def unmerged_files_dict() -> dict:
    # created a dictionary of the results folders as keys and the unmerged filenames as a list
    # of values
    dictionary = {}
    for folder, subfolder, filenames in os.walk(RESULTS_PATH):
        if folder == RESULTS_PATH:
            continue
        else:
            dictionary[folder] = [
                i for i in filenames if i.replace(".csv", "") != os.path.basename(folder) and i[-4:] == ".csv"
            ]
    return dictionary

def display_files_to_merge(file_list: list):
    # shows a list of files nicely that are about to be merged
    s = "files to merge:\n"
    for filename in file_list:
        ID = os.path.basename(filename)
        s = f"{s}{ID}\n"
    print(s)

def verify_merge() -> bool:
    # asks for input of y/n to confirm file merge
    verify = input("Are you sure you would like to merge these files? y/n\n")
    if verify == "y":
        return True
    elif verify == "n":
        return False
    else:
        return "merge not verified, files not merged\n\n"

def execute_file_merge(filename_path: str, file_list: list, delete_old:bool=False):
    # executes the merging of files verified by user
    dfs_to_merge = []
    print(filename_path)
    for file_path in file_list:
        dfs_to_merge.append(pd.read_csv(os.path.join(filename_path, file_path)))
    merged_df = pd.concat(dfs_to_merge, axis=0).reset_index(drop=True)
    new_file = os.path.join(filename_path, f"{os.path.basename(filename_path)}.csv")
    iterator = 0
    while os.path.exists(new_file):
        new_file = os.path.join(filename_path, f"{os.path.basename(filename_path)}_{iterator}.csv")
        iterator += 1
    merged_df.to_csv(new_file)
    if delete_old:
        for filepath in [os.path.join(filename_path, filename) for filename in file_list]:
            send2trash(filepath)
    print("files merged\n\n")

def merge_ungrouped_files(delete_old:bool=False):
    # creates user dialogue which clarifies which files will be merged and
    # verifies each set of files to ensure correct execution
    for filename_path, file_list in unmerged_files_dict().items():
        display_files_to_merge(file_list)
        if not file_list == []:
            verified = verify_merge() 
        if verified:
            execute_file_merge(filename_path, file_list, delete_old)
        else:
            print("files not merged\n\n")
    print("done")

    
# laoding data usage
if __name__ == "__main__":
    merge_ungrouped_files()
    