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

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, Bidirectional, TimeDistributed, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

from util import shuffle_arrays


def main():
    np.random.seed(7)
    seq_dir, ss_dir, out_dir = parse_arguments()

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Specify a maximum number of sequences for now, which speeds up execution.
    # `maxseq` only affects how many files to read, it doesn't actually
    # correspond to the number of training points. It's probably a good idea to
    # read all files and then specify a smaller subset for training/testing in
    # `make_data_tensors`.
    maxseq = -1
    seqs, sss = read_seqs_and_sss(seq_dir, ss_dir, maxseq=maxseq)

    # Construct data for Keras. This pads sequences with rows of zero for ones
    # that are shorter than the longest sequence in `seqs`.
    x, y = make_data_tensors(seqs, sss, ndata=5000)
    assert x.shape[:2] == y.shape[:2]

    train_split = 0.8
    train_end = int(train_split * len(x))
    val_end = int((1.0 - (1.0 - train_split) / 2.0) * len(x))
    x_train, x_val, x_test = x[:train_end], x[train_end:val_end], x[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    
    num_samples = x_train.shape[0]
    max_seq_len = x_train.shape[1]
    num_features = x_train.shape[2]
    num_classes = y_train.shape[2]

    # Build Keras model
    hidden_units = 32
    max_epochs = 500
    batch_size = 32
    
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(max_seq_len, num_features)))
    model.add(Bidirectional(LSTM(hidden_units, return_sequences=True, input_shape=(max_seq_len, num_features))))
    model.add(TimeDistributed(Dense(num_classes)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    # Train model. Use early-stopping on validation data to determine when to stop training.
    model_path = os.path.join(out_dir, 'blstm_model.h5')
    checkpointer = ModelCheckpoint(model_path, save_best_only=True)
    model.fit(x_train, y_train, epochs=max_epochs, batch_size=batch_size, verbose=2,
              validation_data=(x_val, y_val), callbacks=[EarlyStopping(patience=10), checkpointer])

    model = load_model(model_path)  # Best model is not necessarily current model instance b/c patience != 0
    score_train = model.evaluate(x_train, y_train)[1]
    score_test = model.evaluate(x_test, y_test)[1]
    print('Train accuracy: {:.2f}%'.format(score_train * 100.0))
    print('Test accuracy: {:.2f}%'.format(score_test * 100.0))


def parse_arguments():
    """
    Read directories containing encoded protein sequences and secondary
    structures from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_dir', type=str, metavar='SEQ_DIR', help='Directory containing encoded protein sequences.')
    parser.add_argument('ss_dir', type=str, metavar='SS_DIR', help='Directory containing encoded secondary structures')
    parser.add_argument('-o', '--out_dir', type=str, metavar='OUT_DIR', help='Directory to save output in (such as Keras models')
    args = parser.parse_args()

    return args.seq_dir, args.ss_dir, args.out_dir


def read_seqs_and_sss(seq_dir, ss_dir, maxseq=-1):
    """
    Read bcolz files containing one-hot representations of protein sequences 
    (in `seq_dir`) and secondary structure annotations (in `ss_dir`). For faster
    execution during debugging, a maximum number of sequences to be read can be
    specified.

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
    x, y = shuffle_arrays(x, y)

    if ndata == 'all':
        return x, y
    elif isinstance(ndata, (int, long)):
        return x[:ndata], y[:ndata]
    else:
        raise Exception('ndata has invalid type')
    

if __name__ == '__main__':
    main()
