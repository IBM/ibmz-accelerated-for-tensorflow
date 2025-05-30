#!/usr/bin/env python3

# IBM Confidential
# © Copyright IBM Corp. 2025

"""
Credit Card Fraud Inference
"""

import argparse
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf

import data_utils


def prepare_model(rnn_type: str = 'lstm') -> tf.keras.models.Model:
    """
    Setup to get the model. Return the compiled model.
    """

    keras_model_path = f'./saved_model/{rnn_type}.keras'
    model = tf.keras.models.load_model(keras_model_path)

    return model


def main(rnn_type: str = 'lstm', batch_size: int = 2000, seq_length: int = 7):
    """
    main
    """

    test_generator = data_utils.prepare_inference_data(batch_size, seq_length)

    model = prepare_model(rnn_type)

    print(model.summary())

    y_pred = []
    y_true = []
    for input_batch, batch_label in test_generator:
        y_pred.extend(model.predict(input_batch, batch_size=batch_size))
        y_true.extend(batch_label)

    y_pred = tf.concat(y_pred, axis=0)
    y_true = tf.constant(y_true)
    correct_prediction = tf.equal(tf.cast(tf.round(y_pred), tf.int32), y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Test accuracy:', accuracy.numpy())


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
        default=2000,
        help='Batch size for inference data (default: 2000)',
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        default=7,
        help='Sequence length for inference data (default: 7)',
    )
    args = parser.parse_args()

    main(args.rnn_type, args.batch_size, args.seq_length)
