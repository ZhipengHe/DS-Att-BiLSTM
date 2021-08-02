from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
import pandas as pd
import numpy as np

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

def check_if_activity_exists(group, activity_col, label_col, activity, pos_label, neg_label):
    relevant_activity_idxs = np.where(group[activity_col] == activity)[0]
    if len(relevant_activity_idxs) > 0:
        idx = relevant_activity_idxs[0]
        group[label_col] = pos_label
        return group[:idx]
    else:
        group[label_col] = neg_label
        return group

def check_if_any_of_activities_exist(group, activity_col, activities):
    if np.sum(group[activity_col].isin(activities)) > 0:
        return True
    else:
        return False