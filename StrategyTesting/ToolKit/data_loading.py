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
    
def list_results_files(as_id: bool=False, as_filename: bool=False, as_path: bool=False) -> list:
    # walks the optimization_results directory and returns the simset ids in a list
    for _, _, filenames in os.walk(RESULTS_PATH):
        res = filenames
    if as_filename:
        return res
    elif as_id:
        return [filename.replace(".csv", "") for filename in res]
    elif as_path:
        return [os.path.join(RESULTS_PATH, filename) for filename in filenames]


def list_unmerged_files(as_id: bool=False, as_filename: bool=False, as_path: bool=False)-> List:
    # lists all files in results that contain iterator element indicating they are unmerged
    simset_ids = list_results_files(as_id=True)
    unmerged = []
    for simset_id in simset_ids:
        if len(simset_id.split("_")) == 4:
            unmerged.append(simset_id)
    if as_id:
        return unmerged
    elif as_filename:
        return [f"{simset_id}.csv" for simset_id in unmerged]
    elif as_path:
        return [os.path.join(RESULTS_PATH, f"{simset_id}.csv") for simset_id in unmerged]
        
def remove_iterator(simset_id: str, include_date: bool=False) -> str:
    # takes a simset_id as an input and returns the same ID without 
    # the iterator element
    x = -1 if include_date else -2
    res = simset_id.split("_")[:x]
    return "_".join(res)

def date_hash_from_ID(ID: str, from_path: bool=False) -> str:
    # takes simset ID and returns the date portion as a string
    if not from_path:
        components = ID.split("_")
        return components[2]
    else:
        ID = os.path.basename(ID)
        components = ID.replace(".csv", "").split("_")
        return components[2]

def group_unmerged_files(include_date: bool=False):
    # groups together files that have been unmerged by their simset_id
    unmerged = list_unmerged_files(as_id=True)
    groups = {}
    for simset_id in unmerged:
        keys = list(groups.keys())
        real_id = remove_iterator(simset_id, include_date=include_date)
        if not real_id in keys:
            groups[real_id] = [os.path.join(RESULTS_PATH, f"{simset_id}.csv")]
        else:
            groups[real_id].append(os.path.join(RESULTS_PATH, f"{simset_id}.csv"))
    return groups

def add_date_to_grouped_files(grouped_files_dict: dict) -> dict:
    # takes the date portion from the first item in the list of each
    # generalized dictionary entry and adds it to the key
    for key in list(grouped_files_dict.keys()):
        date_hash = date_hash_from_ID(grouped_files_dict[key][0], from_path=True)
        grouped_files_dict[f"{key}_{date_hash}"] = grouped_files_dict.pop(key)
    return grouped_files_dict

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

def execute_file_merge(filename, file_list: list, delete_old:bool=False):
    # executes the merging of files verified by user
    dfs_to_merge = []
    for file_path in file_list:
        dfs_to_merge.append(pd.read_csv(file_path))
    merged_df = pd.concat(dfs_to_merge, axis=0).reset_index(drop=True)
    merged_df.to_csv(os.path.join(RESULTS_PATH, filename))
    if delete_old:
        for filepath in file_list:
            send2trash(filepath)
    print("files merged\n\n")

def merge_ungrouped_files(delete_old:bool=False):
    # creates user dialogue which clarifies which files will be merged and
    # verifies each set of files to ensure correct execution
    groups = group_unmerged_files()
    with_date = add_date_to_grouped_files(groups)
    for real_id in list(with_date.keys()):
        file_list = with_date[real_id]
        display_files_to_merge(file_list)
        verified = verify_merge() 
        if verified:
            execute_file_merge(f"{real_id}.csv", file_list, delete_old)
        else:
            print("files not merged\n\n")
    print("done")

    
# laoding data usage
if __name__ == "__main__":
    merge_ungrouped_files(delete_old=True)
    