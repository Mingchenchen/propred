#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def shuffle_arrays(x, y):
    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    return x[p], y[p]
