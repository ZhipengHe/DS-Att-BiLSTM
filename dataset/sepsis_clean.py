import inspect
import os
import sys

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from data.data_cleaning import *


WORKFOLDER = os.getcwd()

# Inspired by https://github.com/irhete/predictive-monitoring-benchmark/blob/master/preprocessing/preprocess_logs_sepsis_cases.py

dataset_name = "Sepsis Cases - Event Log.xes.gz"
case_id_col = "case:concept:name"
timestamp_col = "time:timestamp"
activity_col = "concept:name"
label_col = "label"
pos_label = True
neg_label = False

# features for classifier
dynamic_cat_cols = ['concept:name', 'org:group']
dynamic_num_cols = ['CRP', 'LacticAcid', 'Leucocytes']
static_cat_cols = ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
       'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
       'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
       'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
       'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
       'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
       'SIRSCritTemperature', 'SIRSCriteria2OrMore']
static_num_cols = ['Age']

static_cols = static_cat_cols + static_num_cols
dynamic_cols = dynamic_cat_cols + dynamic_num_cols

cat_cols = dynamic_cat_cols + static_cat_cols

full_cols = [case_id_col] + [timestamp_col] + dynamic_cols + static_cols

cor_cols = static_cols + [label_col]

# Release Type
release_type = ["Release A", "Release B", "Release C", "Release D", "Release E"]

def _clean_missing(df):

    processed_df = df[[case_id_col] + [timestamp_col] + dynamic_cols]

    processed_df[static_cols] = df[static_cols].fillna(method='ffill')
    # processed_df.to_csv(os.path.join(WORKFOLDER, "sepsis_processed.csv"), index=False)

    return processed_df

def _clean_imcompleted(df):

    tmp = df.groupby([case_id_col]).apply(check_if_any_of_activities_exist, activity_col=activity_col, activities=release_type)
    incomplete_cases = tmp.index[tmp==False]
    filtered_df = df[~df[case_id_col].isin(incomplete_cases)]
    # filtered_df.to_csv(os.path.join(WORKFOLDER, "sepsis_filtered.csv"), index=False)

    return filtered_df

def _add_time_features(df):
    df = df.reset_index(drop=True)

    # add features extracted from timestamp
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.groupby(case_id_col).apply(extract_timestamp_features, timestamp_col=timestamp_col)

    return df

def _predictor_1(inputs, filtered):
    # label for predictor 1 - IC Admission
    df = _clean_missing(inputs)
    if filtered is True:
        df = _clean_imcompleted(df)
    labeled_df = df.sort_values(
        by=[case_id_col, timestamp_col], ascending=True, kind="mergesort").groupby(case_id_col).apply(
            check_if_activity_exists, activity_col=activity_col, label_col=label_col, activity="Admission IC")
    
    print('Distribution of cases by the target variable\n')
    print(labeled_df.groupby([label_col])[case_id_col].nunique())

    labeled_df = _add_time_features(labeled_df)

    if filtered is True:
        featureCorrelation(labeled_df, cor_cols, "Sepsis_p1_filter")
        labeled_df.to_csv(os.path.join(WORKFOLDER, "sepsis_p1_filter.csv"), index=False)
    else:
        featureCorrelation(labeled_df, cor_cols, "Sepsis_p1")
        labeled_df.to_csv(os.path.join(WORKFOLDER, "sepsis_p1.csv"), index=False)
    
    labeled_df = labeled_df.reset_index(drop=True)

    prefix_df = prefix_helper_func(labeled_df)
    prefix_df.to_csv(os.path.join(WORKFOLDER, "sepsis_p1_prefix.csv"), index=False)

def _predictor_2(inputs, filtered):
    """label for predictor 2 - Release type

    Args:
        filtered (Boolean): 
    """

    # 
    df = _clean_missing(inputs)
    if filtered is True:
        df = _clean_imcompleted(df)
    labeled_df = df.sort_values(
        by=[case_id_col, timestamp_col], ascending=True, kind="mergesort").groupby(case_id_col).apply(
            check_if_activity_exists, activity_col=activity_col, label_col=label_col, activity="Release A")
    
    print('Distribution of cases by the target variable\n')
    print(labeled_df.groupby([label_col])[case_id_col].nunique())

    labeled_df = _add_time_features(labeled_df)

    if filtered is True:
        featureCorrelation(labeled_df, cor_cols, "Sepsis_p2_filter")
        labeled_df.to_csv(os.path.join(WORKFOLDER, "sepsis_p2_filter.csv"), index=False)
    else:
        featureCorrelation(labeled_df, cor_cols, "Sepsis_p2")
        labeled_df.to_csv(os.path.join(WORKFOLDER, "sepsis_p2.csv"), index=False)

def _predictor_3(inputs, filtered):
    # label for predictor 3 - Four trajectories
    df = _clean_missing(inputs)
    if filtered is True:
        df = _clean_imcompleted(df)
    
    # need to change the label method.
    labeled_df = df.sort_values(
        by=[case_id_col, timestamp_col], ascending=True, kind="mergesort").groupby(case_id_col).apply(
            check_activity_order, activity_col=activity_col, label_col=label_col, pre_act="Admission NC", post_act="Admission IC")
    
    print('Distribution of cases by the target variable\n')
    print(labeled_df.groupby([label_col])[case_id_col].nunique())

    labeled_df = _add_time_features(labeled_df)

    if filtered is True:
        featureCorrelation(labeled_df, cor_cols, "Sepsis_p3_filter")
        labeled_df.to_csv(os.path.join(WORKFOLDER, "sepsis_p3_filter.csv"), index=False)
    else:
        featureCorrelation(labeled_df, cor_cols, "Sepsis_p3")
        labeled_df.to_csv(os.path.join(WORKFOLDER, "sepsis_p3.csv"), index=False)
    



def main():
    """Main function for dataset cleaning"""

    # Change your raw_data_path here, 
    # if you do not put this python file and raw data in the same folder
    raw_data_path = os.path.join(WORKFOLDER, dataset_name)
    log_df = readDataset(raw_data_path)
    extract_logs_metadata(log_df)

    # _predictor_1(log_df, False)
    _predictor_1(log_df, True)
    # _predictor_2(log_df, False)
    # _predictor_2(log_df, True)
    # _predictor_3(log_df, False)
    # _predictor_3(log_df, True)

    # act_seq = labeled_df.groupby(case_id_col)[activity_col].apply(list).reset_index(name='act_seq')
    # res_seq = labeled_df.groupby(case_id_col)['org:group'].apply(list).reset_index(name='res_seq').sort_values(by=[case_id_col])
    # t_seq = labeled_df.groupby(case_id_col)['elapsed_time'].apply(list).reset_index(name='t_seq').sort_values(by=[case_id_col])
    # label_seq = labeled_df.groupby(case_id_col).first()['label'].reset_index(name='label_seq').sort_values(by=[case_id_col])

    # seq_df = act_seq
    # seq_df['res_seq'] = res_seq['res_seq']
    # seq_df['t_seq'] = t_seq['t_seq']
    # seq_df[label_col] = label_seq['label_seq']
    # seq_df.to_csv(os.path.join(WORKFOLDER, "seq_df.csv"))


if __name__ == "__main__":
    main()



