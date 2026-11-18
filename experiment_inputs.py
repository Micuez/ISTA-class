"""Data entry points imported by the adapted author code (one framework per process)."""

import os

import numpy as np

from dataset import RecoveryData


def open_data():
    return RecoveryData(os.environ["RECOVERY_DATA"])


def column_batches(data, split, batch_size, snr_db=None, epoch=0):
    for y, x in data.batches(split, batch_size, snr_db, epoch):
        yield y.T, x.T


def tf_training_inputs(data, batch_size):
    import tensorflow as tf

    def stream():
        epoch = 0
        while True:
            for pair in column_batches(data, "train", batch_size, epoch=epoch):
                yield pair
            epoch += 1

    m, n = data.A.shape
    ds = tf.data.Dataset.from_generator(
        stream, (tf.float32, tf.float32), (tf.TensorShape([m, None]), tf.TensorShape([n, None])))
    y, x = ds.make_one_shot_iterator().get_next()
    val_size = data.config["clean_signals"]["val"] * len(data.levels)
    y_val, x_val = next(column_batches(data, "val", val_size))
    return y, x, tf.constant(y_val), tf.constant(x_val)


def tf_test_inputs(data):
    import tensorflow as tf

    y, x = data.arrays("test", float(os.environ["RECOVERY_TEST_DB"]))
    return tf.constant(y.T), x.T


def torch_batches(data, split, batch_size, snr_db=None, epoch=0):
    import torch

    for y, x in data.batches(split, batch_size, snr_db, epoch):
        yield torch.from_numpy(y), torch.from_numpy(x)
