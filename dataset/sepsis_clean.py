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

# Release Type
release_type = ["Release A", "Release B", "Release C", "Release D", "Release E"]

def main():
    """Main function for dataset cleaning"""

    # Change your raw_data_path here, 
    # if you do not put this python file and raw data in the same folder
    raw_data_path = os.path.join(WORKFOLDER, "Sepsis Cases - Event Log.xes.gz")
    log_df = readDataset(raw_data_path)

    processed_df = log_df[full_cols].fillna(method='ffill')
    # processed_df.to_csv(os.path.join(WORKFOLDER, "sepsis_processed.csv"), index=False)

    processed_df = processed_df.reset_index(drop=True)

    tmp = processed_df.groupby([case_id_col]).apply(check_if_any_of_activities_exist, activity_col=activity_col, activities=release_type)
    incomplete_cases = tmp.index[tmp==False]
    filtered_df = processed_df[~processed_df[case_id_col].isin(incomplete_cases)]
    # filtered_df.to_csv(os.path.join(WORKFOLDER, "sepsis_filtered.csv"), index=False)

    # label for predictor 1
    labeled_df = processed_df.sort_values(
        by=[case_id_col, timestamp_col], ascending=True, kind="mergesort").groupby(case_id_col).apply(
            check_if_activity_exists, activity_col=activity_col, label_col=label_col, activity="Admission IC", pos_label=pos_label, neg_label=neg_label)

    # label for predictor 2
    # labeled_df = filtered_df.sort_values(
    #     by=[case_id_col, timestamp_col], ascending=True, kind="mergesort").groupby(case_id_col).apply(
    #         check_if_activity_exists, activity_col=activity_col, label_col=label_col, activity="Release A", pos_label=pos_label, neg_label=neg_label)


    print('Distribution of cases by the target variable\n')
    print(labeled_df.groupby([label_col])[case_id_col].nunique())

    labeled_df = labeled_df.reset_index(drop=True)

    # add features extracted from timestamp
    labeled_df[timestamp_col] = pd.to_datetime(labeled_df[timestamp_col], utc=True)
    labeled_df = labeled_df.groupby(case_id_col).apply(extract_timestamp_features, timestamp_col=timestamp_col)

    labeled_df.to_csv(os.path.join(WORKFOLDER, "sepsis_labeled.csv"), index=False)

    cor_columns = static_cols + [label_col]

    featureCorrelation(labeled_df, cor_columns, "Sepsis_IC_Admission")
    # featureCorrelation(labeled_df, cor_columns, "Sepsis_Release_A")
    # featureCorrelation(labeled_df, cor_columns, "Sepsis_IC_Admission_filter")
    # featureCorrelation(labeled_df, cor_columns, "Sepsis_Release_A_filter")


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


