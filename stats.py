#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
from keras.models import load_model

from util import calculate_accuracy, generate_confusion_matrix, calculate_sensitivity_precision_f1_mcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, metavar='MODEL', help='Path to Keras model')
    parser.add_argument('x_path', type=str, metavar='X', help='Path to inputs')
    parser.add_argument('y_path', type=str, metavar='Y', help='Path to targets')
    parser.add_argument('-f', '--output_file', type=str, default=None, metavar='FILE', help='File to print output to')
    args = parser.parse_args()

    model = load_model(args.model_path)
    x = np.load(args.x_path)
    y = np.load(args.y_path)

    y_pred = model.predict(x)

    acc = calculate_accuracy(y, y_pred)
    cm = generate_confusion_matrix(y, y_pred)
    cm_norm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]
    stats = calculate_sensitivity_precision_f1_mcc(cm)

    print('Accuracy: {:.2f}%'.format(acc * 100.0))
    print('Confusion matrix (rows are actual, columns are predicted):')
    print(cm)
    print('Normalized confusion matrix:')
    print(cm_norm)
    print('Sensitivity: {}'.format(stats['sensitivity']))
    print('Precision: {}'.format(stats['precision']))
    print('F1 score: {}'.format(stats['f1']))
    print('Matthews correlation coefficient: {}'.format(stats['mcc']))

    if args.output_file is not None:
        with open(args.output_file, 'w') as f:
            f.write('Accuracy: {:.2f}%\n'.format(acc * 100.0))
            f.write('Confusion matrix (rows are actual, columns are predicted):\n')
            f.write(str(cm) + '\n')
            f.write('Normalized confusion matrix:\n')
            f.write(str(cm_norm) + '\n')
            f.write('Sensitivity: {}\n'.format(stats['sensitivity']))
            f.write('Precision: {}\n'.format(stats['precision']))
            f.write('F1 score: {}\n'.format(stats['f1']))
            f.write('Matthews correlation coefficient: {}\n'.format(stats['mcc']))


if __name__ == '__main__':
    main()
