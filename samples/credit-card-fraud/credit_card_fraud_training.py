#!/usr/bin/env python3

# IBM Confidential
# © Copyright IBM Corp. 2025

"""
Credit Card Fraud Training
"""

import argparse
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf

import data_utils


def prepare_model(
    rnn_type: str = 'lstm',
    seq_length: int = 7
) -> tf.keras.models.Model:
    """
    Setup to get the model. Return the compiled model.
    """

    if rnn_type == 'lstm':
        rnn_layer = tf.keras.layers.LSTM
    else:
        rnn_layer = tf.keras.layers.GRU

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(seq_length, 220,)),
        rnn_layer(200, return_sequences=True),
        rnn_layer(200, return_sequences=False),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = 'adam'
    loss = 'binary_crossentropy'
    metrics = [
        'accuracy',
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.TrueNegatives(name='tn')
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def main(rnn_type: str = 'lstm', batch_size: int = 2000, seq_length: int = 7):
    """
    main
    """

    train_generator = data_utils.prepare_training_data(batch_size, seq_length)

    model = prepare_model(rnn_type, seq_length)

    print(model.summary())

    # Train the model on the training dataset for 20 epochs.
    # Generates 50,000 batches of batch_size per-epoch.
    model.fit(train_generator, epochs=20, steps_per_epoch=50000, verbose=1)

    # Save model to file
    if not os.path.exists('./saved_model'):
        os.makedirs('./saved_model')
    keras_model_path = f'./saved_model/{rnn_type}.keras'
    model.save(keras_model_path)


if __name__ == '__main__':
    # CLI interface
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
        default=32,
        help='Batch size for training data (default: 32)',
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        default=7,
        help='Sequence length for training data (default: 7)',
    )
    args = parser.parse_args()

    main(args.rnn_type, args.batch_size, args.seq_length)
