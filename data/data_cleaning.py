from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
import pandas as pd

def readDataset(file_path):
    """Read event log dataset from XES file to dataframe.

    Args:
        The name of xes file

    Returns: 
        The dataframe of raw event log
    """
    
    log = xes_importer.apply(file_path)
    dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    return dataframe

def orderCols(dynamic_cols, dataframe):
    """Reorder the columns in dataframe for dynamic and static features

    Args:
        dynamic_cols: list for all dynamic feature names
        dataframe: The dataframe of raw event log

    Returns:
        dataframe: The reordered dataframe
        cols: list for all static feature names
    """
    cols = list(dataframe.columns.values) #Make a list of all of the columns in the df
    
    for col in dynamic_cols:
        cols.pop(cols.index(col)) #Remove dynamic cols from list

    cols.sort()
    dataframe = dataframe[dynamic_cols + cols] #Create new dataframe with columns in the order you want
    dataframe[cols] = dataframe[cols].fillna(method='ffill')

    return dataframe