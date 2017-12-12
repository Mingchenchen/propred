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

from models import blstm, bgru 
from util import shuffle_arrays


def main():
    np.random.seed(7)
    seq_dir, ss_dir, test_seq_dir, test_ss_dir, out_dir, nthreads, hidden_units, layers, max_seq_len, dropout, ndata = parse_arguments()

    if out_dir is None:
        out_dir = os.getcwd()

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    print('Reading files...')
    if not os.path.isfile(os.path.join(out_dir, 'seqs_dict.npy')) or not os.path.isfile(os.path.join(out_dir, 'sss_dict.npy')):
        seqs, sss = read_seqs_and_sss(seq_dir, ss_dir, max_len=max_seq_len)
        np.save(os.path.join(out_dir, 'seqs_dict.npy'), seqs)  # Save dictionaries
        np.save(os.path.join(out_dir, 'sss_dict.npy'), sss)
    else:
        seqs = np.load(os.path.join(out_dir, 'seqs_dict.npy'))
        seqs = seqs.flat[0] 
        sss = np.load(os.path.join(out_dir, 'sss_dict.npy'))
        sss = sss.flat[0] 
    
    # Save test sets as well?
    test_sets_seqs = []; test_sets_sss = []
    for test_set in ['casp10','casp11','cullpdb']:
        actual_seq_dir = os.path.join(test_seq_dir, test_set)
        actual_ss_dir = os.path.join(test_ss_dir, test_set)
        seqs_test, sss_test = read_seqs_and_sss(actual_seq_dir, actual_ss_dir, max_len=max_seq_len)
        test_sets_seqs.append(seqs_test); test_sets_sss.append(sss_test)
        # np.save(os.path.join(out_dir, 'seqs_dict_'+test_set+'.npy'), seqs_test)
        # np.save(os.path.join(out_dir, 'sss_dict_'+test_set+'.npy'), sss_test) 
      

    # Construct data for Keras. This pads sequences with rows of zeros for ones
    # that are shorter than the longest sequence in `seqs`.
    print('Making tensors...')
    # Number of data points includes train, val, and test
    if ndata is None or ndata > len(seqs):
        ndata = 'all'
    x, y = make_data_tensors(seqs, sss, ndata=ndata)
    print('Number of data points: {}'.format(len(x)))
    
    x_test = []; y_test = []
    for i in range(3):
        xt, yt = make_data_tensors(test_sets_seqs[i], test_sets_sss[i], max_len=max_seq_len)
        x_test.append(xt), y_test.append(yt)

    train_split = 0.85  # Fraction of points to use as training data. Rest is validation
    train_end = int(train_split * len(x))
    x_train, x_val = x[:train_end], x[train_end:]
    y_train, y_val = y[:train_end], y[train_end:]
    np.save(os.path.join(out_dir, 'x_train.npy'), x_train)  # Save data
    np.save(os.path.join(out_dir, 'x_val.npy'), x_val)
    np.save(os.path.join(out_dir, 'x_test_casp10.npy'), x_test[0])
    np.save(os.path.join(out_dir, 'x_test_casp11.npy'), x_test[1])
    np.save(os.path.join(out_dir, 'x_test_cullpdb.npy'), x_test[2])
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(out_dir, 'y_test_casp10.npy'), y_test[0])
    np.save(os.path.join(out_dir, 'y_test_casp11.npy'), y_test[1])
    np.save(os.path.join(out_dir, 'y_test_cullpdb.npy'), y_test[2])
    
    # Set parameters for Keras model
    max_epochs = 50
    batch_size = 32
    patience = 5
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
    parser.add_argument('test_seq_dir', type=str, metavar='SEQ_TEST_DIR', help='Directory containing encoded protein sequences for test sets')
    parser.add_argument('test_ss_dir', type=str, metavar='SS_TEST_DIR', help='Directory containing encoded secondary structures for test sets')
    parser.add_argument('-u', '--hidden_units', type=int, default=100, metavar='HU', help='Number of hidden units per LSTM layer')
    parser.add_argument('-l', '--layers', type=int, default=1, metavar='L', help='Number of BLSTM layers')
    parser.add_argument('-m', '--max_seq_len', type=int, default=None, metavar='MAX_LEN', help='Maximum sequence length')
    parser.add_argument('-n', '--ndata', type=int, default=None, metavar='NDATA', help='Number of data points to use')
    parser.add_argument('-d', '--dropout', action='store_true', help='Use 0.5 dropout/recurrent_dropout')
    parser.add_argument('-o', '--out_dir', type=str, metavar='OUT_DIR', help='Directory to save output in')
    parser.add_argument('-t', '--threads', type=int, metavar='NTHREADS', help='Number of parallel threads')
    args = parser.parse_args()

    return args.seq_dir, args.ss_dir, args.test_seq_dir, args.test_ss_dir, args.out_dir, args.threads, args.hidden_units, args.layers, args.max_seq_len, args.dropout, args.ndata


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
    
    counter = 0 
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
        
        counter += 1
        if counter % 100 == 0:
            print('Done loading %d files'%counter)
        if len(seqs) == maxseq:
            break
        
    return seqs, sss

def make_data_tensors(seqs, sss, ndata='all', max_len = None):
    """
    Convert the sequence and secondary structure dictionaries to data tensors.
    The number of data points to keep after random shuffling can be specified
    by `ndata`.

    Shorter protein sequences are padded with zeros.
    """
    x = pad_sequences(seqs.values(), maxlen = max_len, padding='post')
    y = pad_sequences(sss.values(), maxlen = max_len, padding='post')
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
