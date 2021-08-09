from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

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
    """
    """
    relevant_activity_idxs = np.where(group[activity_col] == activity)[0]
    if len(relevant_activity_idxs) > 0:
        idx = relevant_activity_idxs[0]
        group[label_col] = pos_label
        return group
    else:
        group[label_col] = neg_label
        return group

def check_if_any_of_activities_exist(group, activity_col, activities):
    """
    """
    if np.sum(group[activity_col].isin(activities)) > 0:
        return True
    else:
        return False

def extract_timestamp_features(group, timestamp_col):
    
    group = group.sort_values(timestamp_col, ascending=False, kind='mergesort')

    tmp = group[timestamp_col] - group[timestamp_col].iloc[-1]
    # tmp = tmp.fillna(0)
    group["elapsed_time"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) # m is for minutes

    group = group.sort_values(timestamp_col, ascending=True, kind='mergesort')
    
    return group

def featureCorrelation(dataframe, cor_columns, name):
    """
    """
    d = dataframe[cor_columns]

    # Compute the correlation matrix
    corr = d.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 17))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    f.savefig(name+'.png')

def extract_logs_metadata(df):
    keys = ["[PAD]", "[UNK]"]
    activities = list(df["concept:name"].unique())
    keys.extend(activities)
    val = range(len(keys))

    coded_activity = dict({"x_word_dict":dict(zip(keys, val))})
    code_activity_normal = dict({"y_word_dict": dict(zip(activities, range(len(activities))))})

    coded_activity.update(code_activity_normal)
    coded_json = json.dumps(coded_activity)
    with open(f"metadata.json", "w") as metadata_file:
        metadata_file.write(coded_json)