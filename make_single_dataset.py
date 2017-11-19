#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np

from main import read_seqs_and_sss, make_data_tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_dir', type=str, metavar='SEQ_DIR', help='Directory containing encoded protein sequences')
    parser.add_argument('ss_dir', type=str, metavar='SS_DIR', help='Directory containing encoded secondary structures')
    parser.add_argument('-n', '--name', type=str, default='', metavar='NAME', help='Base name for output files')
    parser.add_argument('-o', '--out_dir', type=str, metavar='OUT_DIR', help='Directory to save output in')
    args = parser.parse_args()

    seq_dir = args.seq_dir
    ss_dir = args.ss_dir
    name = args.name
    out_dir = args.out_dir

    if out_dir is None:
        out_dir = os.getcwd()

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    print('Reading files...')
    seqs, sss = read_seqs_and_sss(seq_dir, ss_dir)
    np.save(os.path.join(out_dir, name + '_seqs_dict.npy'), seqs)
    np.save(os.path.join(out_dir, name + '_sss_dict.npy'), sss)

    print('Making tensors...')
    x, y = make_data_tensors(seqs, sss)
    print('Number of data points: {}'.format(len(x)))
    np.save(os.path.join(out_dir, name + '_x.npy'), x)
    np.save(os.path.join(out_dir, name + '_y.npy'), y)


if __name__ == '__main__':
    main()
