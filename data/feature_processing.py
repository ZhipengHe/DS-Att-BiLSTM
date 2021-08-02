import pandas as pd
import numpy as np

import tensorflow as tf
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



