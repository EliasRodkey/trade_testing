"""
data_loading.py - Data loading and file management utilities for the backtesting framework.

Functions:
    load_data_as_pd: Loads a single ticker's EOD CSV into a date-indexed DataFrame.
    load_SPY_data: Loads the S&P 500 (SPY) benchmark CSV.
    get_all_symbols: Returns a list of all ticker symbols available in the EOD data directory.
    _combine_columns: Internal helper that joins a single column across multiple ticker CSVs.
    load_eod_matrix: Loads a matrix of close prices (or another column) for a list of tickers.
    concatenate_metrics: Stacks multiple same-shaped DataFrames into a hierarchical multi-index DataFrame.
    list_results_dir: Returns a list of subdirectory names inside the optimization results folder.
    unmerged_files_dict: Returns a dict mapping result folder paths to lists of unmerged CSV filenames.
    display_files_to_merge: Prints the list of files about to be merged.
    verify_merge: Prompts the user to confirm a file merge via stdin.
    execute_file_merge: Merges a list of CSVs into one file, optionally deleting the originals.
    merge_ungrouped_files: Interactive workflow that merges all unmerged result CSVs with user confirmation.

Module-level variables:
    EOD_DATA_DIR  -- Path to the directory containing per-ticker EOD CSV files.
    SPY_PATH      -- Path to the SPY benchmark CSV file.
    RESULTS_PATH  -- Path to the optimization results output directory.
"""

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

def load_SPY_data() -> pd.DataFrame:
    """
    Loads the S&P 500 (SPY) benchmark data from test_data/SPY.csv.

    Returns:
        pd.DataFrame: Date-indexed DataFrame with OHLCV columns.
    """
    df = pd.read_csv(SPY_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df

def get_all_symbols() -> List[str]:
    """
    Returns a list of all ticker symbols available in the EOD data directory.

    Returns:
        List[str]: Ticker symbols derived from the filenames in EOD_DATA_DIR (filenames without .csv).
    """
    files = os.listdir(EOD_DATA_DIR)
    s = [item.strip(".csv") for item in files]
    return s

def _combine_columns(filepaths_by_symbol: Dict[str, str],
    column: str='close') -> pd.DataFrame:
    """
    Joins a single named column from multiple ticker CSV files into one wide DataFrame.

    Each ticker becomes a column; the shared date index is used to align rows.

    Args:
        filepaths_by_symbol (Dict[str, str]): Mapping of ticker symbol to its CSV file path.
        column (str): The column name to extract from each CSV. Defaults to 'close'.

    Returns:
        pd.DataFrame: Date-indexed DataFrame with one column per ticker containing the
                      requested data column.
    """

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
    """
    Loads end-of-day price data for a list of tickers into a single wide DataFrame.

    Args:
        tickers (List[str]): List of ticker symbols to load.
        column (str): The column to extract from each ticker's CSV. Defaults to 'close'.

    Returns:
        pd.DataFrame: Date-indexed DataFrame with one column per ticker.
    """
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
    """
    Returns a list of subdirectory names directly inside the optimization results folder.

    Returns:
        list: Names of immediate subdirectories in RESULTS_PATH.
    """
    for folder, subfolders, filenames in os.walk(RESULTS_PATH):
        return subfolders

def unmerged_files_dict() -> dict:
    """
    Returns a mapping of result folder paths to lists of unmerged CSV filenames within them.

    A file is considered "unmerged" if its name (without extension) does not match the folder
    name — i.e., it has not yet been combined into the canonical merged file for that simset.

    Returns:
        dict: Mapping of folder path (str) to list of unmerged CSV filenames (List[str]).
    """
    dictionary = {}
    for folder, subfolder, filenames in os.walk(RESULTS_PATH):
        if folder == RESULTS_PATH:
            continue
        else:
            dictionary[folder] = [
                i for i in filenames if i.replace(".csv", "") != os.path.basename(folder) and i[-4:] == ".csv"
            ]
    return dictionary

def display_files_to_merge(file_list: list) -> None:
    """
    Prints a formatted list of filenames that are about to be merged.

    Args:
        file_list (list): List of file paths or filenames to display.
    """
    s = "files to merge:\n"
    for filename in file_list:
        ID = os.path.basename(filename)
        s = f"{s}{ID}\n"
    print(s)

def verify_merge() -> bool:
    """
    Prompts the user via stdin to confirm or cancel a file merge.

    Returns:
        bool: True if the user confirms with 'y', False if the user declines with 'n'.
              Returns an error message string if input is neither 'y' nor 'n'.
    """
    verify = input("Are you sure you would like to merge these files? y/n\n")
    if verify == "y":
        return True
    elif verify == "n":
        return False
    else:
        return "merge not verified, files not merged\n\n"

def execute_file_merge(filename_path: str, file_list: list, delete_old: bool=False) -> None:
    """
    Concatenates a list of CSV files into a single merged CSV and optionally deletes the originals.

    The merged file is written to filename_path using the folder's basename as the filename.
    If a file with that name already exists, an integer suffix is appended to avoid overwriting.

    Args:
        filename_path (str): Path to the result folder containing the files to merge.
        file_list (list): List of CSV filenames (not full paths) within filename_path to merge.
        delete_old (bool): If True, sends the original files to the system trash after merging.
                           Defaults to False.
    """
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

def merge_ungrouped_files(delete_old: bool=False) -> None:
    """
    Interactive workflow that merges all unmerged result CSVs with per-set user confirmation.

    Iterates over all result subdirectories with unmerged files, displays the file list,
    prompts for confirmation, and executes the merge if confirmed.

    Args:
        delete_old (bool): If True, sends original files to trash after each merge. Defaults to False.
    """
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
    