#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os

import bcolz
import numpy as np

from keras.preprocessing.sequence import pad_sequences


def main():
    data_dir, seq_len, name, out_dir = parse_arguments()
    if out_dir is None:
        out_dir = os.getcwd()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    data_dict = read(data_dir, max_len=seq_len)
    x = make_data_tensor(data_dict, seq_len)
    np.save(os.path.join(out_dir, name + '.npy'), x)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, metavar='DDIR', help='Directory containing x or y data')
    parser.add_argument('seq_len', type=int, metavar='LEN', help='Padded sequence length')
    parser.add_argument('name', type=str, metavar='NAME', help='Name of output file')
    parser.add_argument('-o', '--out_dir', type=str, metavar='ODIR', help='Output directory')
    args = parser.parse_args()

    return args.data_dir, args.seq_len, args.name, args.out_dir


def read(data_dir, max_len=None):
    data_dict = {}

    for f in glob.iglob(os.path.join(data_dir, '*.bc')):
        data = bcolz.open(f)[:]

        if max_len is None or len(data) <= max_len:
            name = os.path.splitext(os.path.basename(f))[0]
            data_dict[name] = data

    return data_dict


def make_data_tensor(data_dict, seq_len):
    x = pad_sequences(data_dict.values(), maxlen=seq_len, padding='post')
    return x


if __name__ == '__main__':
    main()
