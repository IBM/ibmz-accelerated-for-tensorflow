#!/usr/bin/env python3

# IBM Confidential
# © Copyright IBM Corp. 2025, 2026

"""
Credit Card Fraud Training
"""

import argparse
from collections.abc import Generator
import joblib
import math
import os
from pathlib import Path
import pickle as pk
from typing import Any

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf


SEQ_LENGTH = 7


def time_encoder(x: pd.DataFrame) -> pd.DataFrame:
    """
    Encoder for time data.
    """

    x_hm = x['Time'].str.split(':', expand=True)
    x_date = pd.DataFrame({
        'year': x['Year'],
        'month': x['Month'],
        'day': x['Day'],
        'hour': x_hm[0],
        'minute': x_hm[1]})
    d = pd.to_datetime(x_date).astype(np.int64)
    return pd.DataFrame(d, columns=['Year_Month_Day_Time'])


def amt_encoder(x: pd.DataFrame) -> pd.DataFrame:
    """
    Encoder for decimal data.
    """

    return x.map(lambda amt: amt.lstrip('$')).astype(np.float32).map(
        lambda amt: max(1.0, amt)).map(math.log)


def decimal_encoder(x: pd.DataFrame, length: int = 5) -> pd.DataFrame:
    """
    Encoder for integer data.
    """

    col_name = x.columns[0]
    x = np.ravel(x)
    x_new = pd.DataFrame()
    for i in range(length):
        x_new[f'{col_name}_x{i}'] = np.mod(x, 10)
        x = np.floor_divide(x, 10)
    return x_new.astype(np.int64)


def fraud_encoder(x: pd.DataFrame) -> pd.DataFrame:
    """
    Encoder for boolean data.
    """

    return x.map(lambda v: '1' if v == 'Yes' else '0').astype(np.int64)


def create_test_sample(df: pd.DataFrame, indices: np.ndarray[np.int64, Any]):
    """
    Writes test data and indices to files.
    """

    rows = indices.shape[0]
    index_array = np.zeros((rows, SEQ_LENGTH), dtype=np.int64)
    for i in range(SEQ_LENGTH):
        index_array[:, i] = indices + 1 - SEQ_LENGTH + i
    uniques = np.unique(index_array.flatten())
    df.loc[uniques].to_csv('test_100k.csv', index_label='Index')
    np.savetxt('test_100k.indices', indices.astype(np.int64), fmt='%d')


def create_training_sets(csv_path: Path) \
        -> tuple[pd.DataFrame, np.ndarray[Any, Any], np.ndarray[Any, Any],
                 np.ndarray]:
    """
    Reads csv from path and creates training, validation, and test indices.
    """

    x_original = pd.read_csv(csv_path)

    x_original.sort_values(by=['User', 'Card'], inplace=True)
    x_original.reset_index(inplace=True, drop=True)
    x_original.info()

    # Get first of each User-Card combination
    first = x_original[['User', 'Card']].drop_duplicates()
    f = np.array(first.index)
    print(first)

    # Drop the first N transactions
    drop_list = np.concatenate([np.arange(x, x + SEQ_LENGTH - 1) for x in f])
    index_list = np.setdiff1d(x_original.index.values, drop_list)

    # Split into 0.5 train, 0.3 validate, 0.2 test
    tot_length = index_list.shape[0]
    train_length = tot_length // 2
    validate_length = (tot_length - train_length) * 3 // 5
    test_length = tot_length - train_length - validate_length
    print(tot_length, train_length, validate_length, test_length)

    # Generate list of indices for train, validate, test
    np.random.seed(1111)
    train_indices = np.random.choice(index_list, train_length, replace=False)
    tv_list = np.setdiff1d(index_list, train_indices)
    validate_indices = np.random.choice(tv_list, validate_length, replace=False)
    test_indices = np.setdiff1d(tv_list, validate_indices)
    print(train_indices, validate_indices, test_indices)

    # Write test data and indices to file
    create_test_sample(x_original, test_indices[:100000])

    return (x_original, train_indices, validate_indices, test_indices)


def map_sample(
        df: pd.DataFrame, fraud_indices: np.ndarray,
        non_fraud_indices: np.ndarray,
        mapper: ColumnTransformer) -> tuple[np.ndarray, np.ndarray]:
    """
    Maps equal distribution of fraud_indices and non_fraud_indices of df, using
    the provided mapper, and returns as inputs and labels.
    """

    indices = np.concatenate((fraud_indices, np.random.choice(
        non_fraud_indices, fraud_indices.shape[0], replace=False)))
    np.random.shuffle(indices)
    rows = indices.shape[0]
    index_array = np.zeros((rows, SEQ_LENGTH), dtype=np.int64)
    for i in range(SEQ_LENGTH):
        index_array[:, i] = indices + 1 - SEQ_LENGTH + i
    full_df = df.loc[index_array.flatten()]
    full_df.reset_index(inplace=True, drop=True)
    full_df = mapper.transform(full_df)
    data = full_df.drop(
        ['Is Fraud?'], axis=1).to_numpy().reshape(rows, SEQ_LENGTH, -1)
    targets = full_df['Is Fraud?'].to_numpy().reshape(rows, SEQ_LENGTH, 1)
    # Take the label for the final sample in sequence as the sequence label.
    targets = targets[:, -1, :]
    return (data, targets)


def gen_training_batch(
        df: pd.DataFrame, mapper: ColumnTransformer, index_list: np.ndarray,
        batch_size: int) -> Generator[tuple[np.ndarray, np.ndarray], Any, Any]:
    """
    Generator that generates class-balanced batches with shape:
        data    = [batch_size, SEQ_LENGTH, features]
        targets = [batch_size, 1]
    """

    np.random.seed(98765)
    train_df = df.loc[index_list]
    fraud_indices = train_df[train_df['Is Fraud?'] == 'Yes'].index.values
    non_fraud_indices = train_df[train_df['Is Fraud?'] == 'No'].index.values
    del train_df

    while True:
        data, targets = map_sample(df, fraud_indices, non_fraud_indices, mapper)
        count = 0
        while (count + batch_size) <= data.shape[0]:
            yield data[count:count + batch_size], targets[count:count + batch_size]
            count += batch_size


def build_mapper(rnn_type: str) -> ColumnTransformer:
    """
    Build the ColumnTransformer mapper.
    """

    return ColumnTransformer(
        [
            ('Is Fraud?', FunctionTransformer(fraud_encoder), ['Is Fraud?']),
            (
                'Merchant State',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='constant')),
                    ('ordinal', OrdinalEncoder(dtype=np.int64)),
                    ('decimal_encoder', FunctionTransformer(decimal_encoder)),
                    ('one_hot', OneHotEncoder(sparse_output=False))
                ]),
                ['Merchant State']
            ),
            (
                'Zip',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='constant')),
                    ('decimal_encoder', FunctionTransformer(decimal_encoder)),
                    ('one_hot', OneHotEncoder(sparse_output=False))
                ]),
                ['Zip']
            ),
            (
                'Merchant Name',
                Pipeline([
                    ('ordinal', OrdinalEncoder()),
                    ('decimal_encoder', FunctionTransformer(decimal_encoder)),
                    ('one_hot', OneHotEncoder(sparse_output=False))
                ]),
                ['Merchant Name']
            ),
            (
                'Merchant City',
                Pipeline([
                    ('ordinal', OrdinalEncoder()),
                    ('decimal_encoder', FunctionTransformer(decimal_encoder)),
                    ('one_hot', OneHotEncoder(sparse_output=False))
                ]),
                ['Merchant City']
            ),
            (
                'MCC',
                Pipeline([
                    ('ordinal', OrdinalEncoder(dtype=np.int64)),
                    ('decimal_encoder', FunctionTransformer(decimal_encoder)),
                    ('one_hot', OneHotEncoder(sparse_output=False))
                ]),
                ['MCC']
            ),
            (
                'Use Chip',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='constant')),
                    ('ordinal', OrdinalEncoder(dtype=np.int64)),
                    ('one_hot', OneHotEncoder(sparse_output=False))
                ]),
                ['Use Chip']
            ),
            (
                'Errors?',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='constant')),
                    ('ordinal', OrdinalEncoder(dtype=np.int64)),
                    ('one_hot', OneHotEncoder(sparse_output=False))
                ]),
                ['Errors?']
            ),
            (
                'Year_Month_Day_Time',
                Pipeline([
                    ('time_encoder', FunctionTransformer(time_encoder)),
                    ('min_max', MinMaxScaler())
                ]),
                ['Year', 'Month', 'Day', 'Time']
            ),
            (
                'Amount',
                Pipeline([
                    ('amt_encoder', FunctionTransformer(amt_encoder)),
                    ('min_max', MinMaxScaler())
                ]),
                ['Amount']
            )
        ],
        verbose_feature_names_out=False
    )


def prepare_training_data(
        rnn_type: str, batch_size: int) \
        -> tuple[Generator[tuple[np.ndarray, np.ndarray], Any, Any], int]:
    """
    Load and preprocess training data. Returns the training generator and
    the number of input features (excluding the label column).
    """

    csv_path = Path('./card_transaction.v1.csv')
    x_original, train_indices, _, _ = create_training_sets(csv_path)

    mapper = build_mapper(rnn_type)
    mapper.set_output(transform='pandas')

    mapper_path = f'./fitted_mapper_v2_{rnn_type}.pkl'
    if os.path.exists(mapper_path):
        print('Loading saved mapper . . .')
        with open(mapper_path, 'rb') as f:
            fitted_mapper = joblib.load(f)
    else:
        print('Fitting mapper . . .')
        fitted_mapper = mapper.fit(x_original)
        with open(mapper_path, 'wb') as f:
            pk.dump(fitted_mapper, f)

    # Sample the mapper to determine the number of input features
    mapped_sample = fitted_mapper.transform(x_original[:100])
    input_size = mapped_sample.shape[-1] - 1

    return gen_training_batch(
        x_original, fitted_mapper, train_indices, batch_size), input_size


def prepare_model(
        rnn_type: str, input_size: int) -> tf.keras.models.Model:
    """
    Build and compile the model.
    """

    if rnn_type == 'lstm':
        rnn_layer = tf.keras.layers.LSTM
    else:
        rnn_layer = tf.keras.layers.GRU

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(SEQ_LENGTH, input_size)),
        rnn_layer(200, return_sequences=True),
        rnn_layer(200, return_sequences=False),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.TrueNegatives(name='tn')
        ]
    )

    return model


def main(rnn_type: str = 'lstm', batch_size: int = 2048):
    """
    main
    """

    train_generator, input_size = prepare_training_data(rnn_type, batch_size)

    model = prepare_model(rnn_type, input_size)

    print(model.summary())

    # Train the model for 20 epochs, 50,000 batches per epoch.
    model.fit(train_generator, epochs=20, steps_per_epoch=50000, verbose=1)

    # Save model to file
    if not os.path.exists('./saved_model'):
        os.makedirs('./saved_model')
    model.save(f'./saved_model/{rnn_type}.keras')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rnn-type',
        type=str.lower,
        choices=['lstm', 'gru'],
        default='lstm',
        help='RNN type used within model (default: lstm)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2048,
        help='Batch size for training (default: 2048)',
    )
    args = parser.parse_args()

    main(args.rnn_type, args.batch_size)
