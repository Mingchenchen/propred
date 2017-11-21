#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import confusion_matrix


def shuffle_arrays(x, y):
    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    return x[p], y[p]


def calculate_accuracy(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sum(np.equal(np.argmax(y_true, axis=-1),
                           np.argmax(y_pred, axis=-1)) * np.sum(y_true, axis=-1)) / np.sum(y_true)


def generate_confusion_matrix(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    n_classes = y_true.shape[2]
    cm = np.zeros((n_classes, n_classes))
    labels = np.arange(n_classes)

    for yt, yp in zip(y_true, y_pred):
        yt_trimmed = yt[~np.all(yt == 0, axis=1)]  # Remove zero rows
        yp_trimmed = yp[:len(yt_trimmed)]

        cm += confusion_matrix(yt_trimmed.argmax(axis=-1), yp_trimmed.argmax(axis=-1), labels=labels)

    return cm


def get_true_false_positives_negatives(cm):
    TP = np.diag(cm)
    TN = np.trace(cm) - TP
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    return TP, TN, FP, FN


def calculate_recall_precision_f1_mcc(cm):
    TP, TN, FP, FN = get_true_false_positives_negatives(cm)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = 2.0 * precision * recall / (precision + recall)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return {'recall': recall, 'precision': precision, 'f1': f1, 'mcc': mcc}
