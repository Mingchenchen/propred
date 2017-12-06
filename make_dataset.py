#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os

import bcolz
import numpy as np

from keras.preprocessing.sequence import pad_sequences


def main():
    data_dir, name, out_dir = parse_arguments()
    if out_dir is None:
        out_dir = os.getcwd()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    data_dict = read(data_dir)
    x = make_data_tensor(data_dict)
    np.save(os.path.join(out_dir, name + '.npy'), x)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, metavar='DDIR', help='Directory containing x or y data')
    parser.add_argument('name', type=str, metavar='NAME', help='Name of output file')
    parser.add_argument('-o', '--out_dir', type=str, metavar='ODIR', help='Output directory')
    args = parser.parse_args()

    return args.data_dir, args.name, args.out_dir


def read(dir):
    data_dict = {}

    for f in glob.iglob(os.path.join(dir, '*.bc')):
        x = bcolz.open(f)[:]
        name = os.path.splitext(os.path.basename(f))[0]
        data[name] = x

    return data_dict


def make_data_tensor(data_dict):
    x = pad_sequences(data_dict.values(), padding='post')
    return x


if __name__ == '__main__':
    main()
