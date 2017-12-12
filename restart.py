#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np

from keras import backend as K

from models import load_and_train


def main():
    data_dir, model_path, nthreads = parse_arguments()

    print('Loading tensors...')
    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    x_val = np.load(os.path.join(data_dir, 'x_val.npy'))
    #x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
    x_test = [np.load(os.path.join(data_dir, 'x_test_casp10.npy')), np.load(os.path.join(data_dir, 'x_test_casp11.npy')), np.load(os.path.join(data_dir, 'x_test_cullpdb.npy'))]
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    #y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    y_test = [np.load(os.path.join(data_dir, 'y_test_casp10.npy')), np.load(os.path.join(data_dir, 'y_test_casp11.npy')), np.load(os.path.join(data_dir, 'y_test_cullpdb.npy'))]

    max_epochs = 50
    batch_size = 32
    patience = 5

    if nthreads is not None:
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=nthreads, inter_op_parallelism_threads=nthreads)))

    print('Restarting training')
    load_and_train(model_path, x_train, x_val, x_test, y_train, y_val, y_test,
                   max_epochs=max_epochs, batch_size=batch_size, patience=patience)

    
def parse_arguments():
    """
    Read directory containing data tensors and model path from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, metavar='DATA_DIR', help='Directory containing .npy tensors')
    parser.add_argument('model_path', type=str, metavar='MODEL', help='Path to Keras model')
    parser.add_argument('-t', '--threads', type=int, metavar='NTHREADS', help='Number of parallel threads')
    args = parser.parse_args()

    return args.data_dir, args.model_path, args.threads


if __name__ == '__main__':
    main()
