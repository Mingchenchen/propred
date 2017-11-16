#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def shuffle_arrays(x, y):
    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    return x[p], y[p]


def calculate_accuracy(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    total_residues = total_correct_residues = 0

    for yt, yp_prob in zip(y_true, y_pred):
        yp = np.zeros_like(yp_prob, dtype=np.int)  # Convert softmax probabilities to one-hot
        yp[np.arange(len(yp_prob)), yp_prob.argmax(axis=1)] = 1

        yt_trimmed = yt[~np.all(yt == 0, axis=1)]  # Only want to compute accuracy for non-zero rows
        yp_trimmed = yp[:len(yt_trimmed)]

        total_residues += len(yt_trimmed)
        total_correct_residues += np.sum(np.all(yt_trimmed == yp_trimmed, axis=1))

    return total_correct_residues / total_residues
