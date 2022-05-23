import pandas as pd
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental import preprocessing

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    """
    Use one-hot encoding method to encode categorical features. Come from tensorflow tutorials:
    https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers#demonstrate_the_use_of_preprocessing_layers

    Args:
        name: The name of feature
        dataset: The input dataset
        dtype: 'string' or others

    Returns:
        A layer which maps values from a vocabulary to integer indices and one-hot encodes the features

    """
    #Create a StringLookup layer which will turn strings into integer indices
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_tokens=max_tokens)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Create a Discretization for our integer indices.
    encoder = preprocessing.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply one-hot encoding to our indices. The lambda function captures the
    # layer so we can use them, or include them in the functional model later.
    return lambda feature: encoder(index(feature))


def get_normalization_layer(name, dataset):
    """
    Normalize numerical features. Come from tensorflow tutorials:
    https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers#categorical_columns

    Args:
        name: The name of feature
        dataset: The input dataset
    
    Returns:
        A layer which applies featurewise normalization to numerical features

    """
    # Create a Normalization layer for our feature.
    normalizer = preprocessing.Normalization(axis=None)

    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer


def prepare_data(df, 
    act_word_dict, res_word_dict, 
    max_case_length):
    
    act = df["prefix"].values
    res = df["res_list"].values
    label = df["label"].values
    t = df["elapsed_time_log"].values

    token_act = list()
    for _act in act:
        token_act.append([act_word_dict[s] for s in _act.split(":||:")])

    token_res = list()
    for _res in res:
        token_res.append([res_word_dict[s] for s in _res.split(":||:")])
    
    token_label = list()
    for _label in label:
        token_label.append(_label)

    token_t = list()
    for _t in t:
        token_t.append(_t)

    token_act = tf.keras.preprocessing.sequence.pad_sequences(
        token_act, maxlen=max_case_length, padding='post')

    token_res = tf.keras.preprocessing.sequence.pad_sequences(
        token_res, maxlen=max_case_length, padding='post')
    
    

    token_act = np.array(token_act, dtype=np.float32)
    token_res = np.array(token_res, dtype=np.float32)
    token_t = np.array(token_t, dtype=np.float32)
    token_label = np.array(token_label, dtype=np.int)
    token_label = np.transpose(token_label)
    token_t = np.array(token_t)
    token_t = np.transpose(token_t)

    return token_act, token_res, token_label, token_t



def split_train_test(df, percentage):
    cases = set(df["case:concept:name"].unique().tolist())

    # num_test_cases = int(np.round(len(cases)*percentage))
    # test_cases = cases[:num_test_cases]
    # train_cases = cases[num_test_cases:]

    num_test_cases = int(np.round(len(cases)*percentage)/2) # set the number to select here.
    test_cases = random.Random(2021).sample(cases, num_test_cases)
    rest_cases = cases.difference(test_cases)
    val_cases = random.Random(2021).sample(rest_cases, num_test_cases)
    train_cases = rest_cases.difference(val_cases)

    df_train, df_test, df_val = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for case in train_cases:
        df_train = df_train.append(df[df["case:concept:name"] == case]) 
    df_train = df_train.sort_values('time:timestamp', ascending=True).reset_index(drop=True)
 
    for case in test_cases:
        df_test = df_test.append(df[df["case:concept:name"]==case]) 
    df_test = df_test.sort_values('time:timestamp', ascending=True).reset_index(drop=True)

    for case in val_cases:
        df_val = df_val.append(df[df["case:concept:name"]==case]) 
    df_val = df_val.sort_values('time:timestamp', ascending=True).reset_index(drop=True)
    
    return df_train, df_test, df_val

