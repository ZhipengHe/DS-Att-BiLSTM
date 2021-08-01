import inspect
import os
import sys

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from data.data_cleaning import orderCols, readDataset

WORKFOLDER = os.getcwd()


def main():
    """Main function for dataset cleaning"""

    # Change your raw_data_path here, 
    # if you do not put this python file and raw data in the same folder
    raw_data_path = os.path.join(WORKFOLDER, "Sepsis Cases - Event Log.xes.gz")
    log_df = readDataset(raw_data_path)

    dynamic_cols = ['case:concept:name', 'concept:name', 'time:timestamp', 'org:group', 
    'lifecycle:transition', 'CRP', 'LacticAcid', 'Leucocytes']
    log_df = orderCols(dynamic_cols, log_df)

    processed_data_path = os.path.join(WORKFOLDER, "sepsis_processed.csv")
    log_df.to_csv(processed_data_path)


if __name__ == "__main__":
    main()
