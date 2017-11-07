#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Structure of input. Size=(samples, max_seq_len, # amino acids)
x = [
    [[0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],
     [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0],
     [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0],
     ...],
    ...
]

Structure of output. Size=(samples, max_seq_len, # secondary structure classes (3 or 8))
y = [
    [[1 0 0],
     [1 0 0],
     [0 0 1],
     [0 1 0],
     ...],
    ...
]
"""

import argparse
import glob
import os

import bcolz
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

from models import blstm
from util import shuffle_arrays


def main():
    np.random.seed(7)
    seq_dir, ss_dir, out_dir, nthreads, hidden_units, layers, max_seq_len, dropout = parse_arguments()

    if out_dir is None:
        out_dir = os.getcwd()

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Specify a maximum number of sequences for now, which speeds up execution.
    # `maxseq` only affects how many files to read, it doesn't actually
    # correspond to the number of training points. It's probably a good idea to
    # read all files and then specify a smaller subset for training/testing in
    # `make_data_tensors`. A value of -1 will read all files.
    print('Reading files...')
    maxseq = -1
    seqs, sss = read_seqs_and_sss(seq_dir, ss_dir, maxseq=maxseq, max_len=max_seq_len)
    np.save(os.path.join(out_dir, 'seqs_dict.npy'), seqs)  # Save dictionaries
    np.save(os.path.join(out_dir, 'sss_dict.npy'), sss)

    # Construct data for Keras. This pads sequences with rows of zeros for ones
    # that are shorter than the longest sequence in `seqs`.
    print('Making tensors...')
    ndata = 'all'  # Specify number of data points (includes train, val, and test)
    x, y = make_data_tensors(seqs, sss, ndata=ndata)
    print('Number of data points: {}'.format(len(x)))

    train_split = 0.8  # Fraction of points to use as training data. Rest is divided equally into val/test
    train_end = int(train_split * len(x))
    val_end = int((1.0 - (1.0 - train_split) / 2.0) * len(x))
    x_train, x_val, x_test = x[:train_end], x[train_end:val_end], x[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    np.save(os.path.join(out_dir, 'x_train.npy'), x_train)  # Save data
    np.save(os.path.join(out_dir, 'x_val.npy'), x_val)
    np.save(os.path.join(out_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test)
    
    # Set parameters for Keras model
    max_epochs = 1000
    batch_size = 32
    patience = 30
    if dropout:
        dropout = recurrent_dropout = 0.5
    else:
        dropout = recurrent_dropout = 0.0

    # Build model and train
    if nthreads is not None:
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=nthreads, inter_op_parallelism_threads=nthreads)))
    model = blstm(x_train, x_val, x_test, y_train, y_val, y_test, out_dir,
                  hidden_units=hidden_units, layers=layers, max_epochs=max_epochs, batch_size=batch_size,
                  patience=patience, dropout=dropout, recurrent_dropout=recurrent_dropout)


def parse_arguments():
    """
    Read directories containing encoded protein sequences and secondary
    structures from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_dir', type=str, metavar='SEQ_DIR', help='Directory containing encoded protein sequences')
    parser.add_argument('ss_dir', type=str, metavar='SS_DIR', help='Directory containing encoded secondary structures')
    parser.add_argument('-u', '--hidden_units', type=int, default=100, metavar='HU', help='Number of hidden units per LSTM layer')
    parser.add_argument('-l', '--layers', type=int, default=1, metavar='L', help='Number of BLSTM layers')
    parser.add_argument('-m', '--max_seq_len', type=int, default=None, metavar='MAX_LEN', help='Maximum sequence length')
    parser.add_argument('-d', '--dropout', action='store_true', help='Use 0.5 dropout/recurrent_dropout')
    parser.add_argument('-o', '--out_dir', type=str, metavar='OUT_DIR', help='Directory to save output in')
    parser.add_argument('-t', '--threads', type=int, metavar='NTHREADS', help='Number of parallel threads')
    args = parser.parse_args()

    return args.seq_dir, args.ss_dir, args.out_dir, args.threads, args.hidden_units, args.layers, args.max_seq_len, args.dropout


def read_seqs_and_sss(seq_dir, ss_dir, maxseq=-1, max_len=None):
    """
    Read bcolz files containing one-hot representations of protein sequences 
    (in `seq_dir`) and secondary structure annotations (in `ss_dir`). For faster
    execution during debugging, a maximum number of sequences to be read can be
    specified. The maximum sequence length can be restricted using `max_len`.

    Returns dictionaries of sequences and secondary structures (dictionary key
    is the file name without its extension).
    """
    seqs, sss = {}, {}
    
    for seq_f in glob.iglob(os.path.join(seq_dir, '*.bc')):
        seq_f_base = os.path.basename(seq_f)
        ss_f = os.path.join(ss_dir, seq_f_base)
        seq = bcolz.open(seq_f)[:]
        ss = bcolz.open(ss_f)[:]

        # For now ignore proteins that have unknown amino acids or the end character
        if not np.any(seq[:,20]) and not np.any(seq[:,23]):
            # Only add sequences up to specified length
            if max_len is None or len(seq) <= max_len:
                name = os.path.splitext(seq_f_base)[0]
                seqs[name] = seq
                sss[name] = ss

        if len(seqs) == maxseq:
            break
        
    return seqs, sss


def make_data_tensors(seqs, sss, ndata='all'):
    """
    Convert the sequence and secondary structure dictionaries to data tensors.
    The number of data points to keep after random shuffling can be specified
    by `ndata`.

    Shorter protein sequences are padded with zeros.
    """
    x = pad_sequences(seqs.values(), padding='post')
    y = pad_sequences(sss.values(), padding='post')
    assert x.shape[:2] == y.shape[:2]
    x, y = shuffle_arrays(x, y)

    if ndata == 'all':
        return x, y
    elif isinstance(ndata, int):
        return x[:ndata], y[:ndata]
    else:
        raise Exception('ndata has invalid type')
    

if __name__ == '__main__':
    main()
