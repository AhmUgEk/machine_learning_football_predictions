"""
Data pipeline functions to process data into the correct format for machine learning models.
"""

# Imports:
import glob
import numpy as np
import os
import pandas as pd
import tensorflow as tf


def arrays_to_dataset(features: np.ndarray, target: np.ndarray, shuffle=True, batch_size=32):
    """
    Function to convert Numpy n-dimensional arrays to TensorFlow Dataset.
    :param features: Numpy n-dimensional array of feature columns (i.e. all columns excluding target).
    :param target: Numpy n-dimensional array of target column.
    :param shuffle: determines whether or not returned Dataset is shuffled. True by default.
    :param batch_size: Number of rows of data to be returned in a single Dataset batch.
    :return: Tensorflow Dataset Batch.
    """
    dataset = tf.data.Dataset.from_tensor_slices((features, target))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))

    dataset = dataset.batch(batch_size)
    return dataset


def csv_to_dataframe(file_path: str):
    """
    Collects all .csv files in a given path and returns a Pandas Dataframe concatenated by column.
    'NAN' values included.
    :param file_path: Local folder path containing .csv files.
    :return: Pandas Dataframe object.
    """
    csv_files = glob.glob(os.path.join(file_path, "*.csv"))
    df_files = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_files, ignore_index=True, sort=False)
    return df


def features_to_densefeatures(dataframe, target_column: str):
    """
    Function to convert feature columns from a Pandas DataFrame to a Numpy n-dimensional array of DenseFeatures.
    :param dataframe: Pandas DataFrame.
    :param target_column: Target column to be excluded from DenseFeatures Object.
    :return: Numpy n-dimensional array of DenseFeatures.
    """
    feature_columns = []
    for column in filter(lambda columns: columns != target_column, dataframe.columns):
        if dataframe[column].dtypes == 'object':
            feature_columns.append(tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_hash_bucket(key=column,
                                                                      hash_bucket_size=dataframe[column].nunique()),
                dimension=3)
            )
        elif dataframe[column].dtypes == 'float64':
            feature_columns.append(tf.feature_column.numeric_column(key=column))

    dense_features = tf.keras.layers.DenseFeatures(feature_columns)
    return dense_features(dict(dataframe)).numpy()
