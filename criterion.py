import numpy as np

"""""""""
Algorithm Criterion File:
This file contains the stopping criteria for the optimization function.
Such stopping conditions include:
    - Difference of successive objective values being less than a given amount
    - Gradient norm being less than a given amount
    - Reaching the iteration limit
    - Checking for failure to continue iterating
"""""""""


def criterion(stor, n):
    if n > 0:
        if abs(stor.val[len(stor.val) - 1] - stor.val[len(stor.val) - 2]) < 10 ** (-12):
            print('Difference of terms achieved')
            return True
        if stor.norm[len(stor.norm) - 1] < 10 ** (-8):
            print('Gradient sufficiently small')
            return True
        if len(stor.val) < n + 1:
            print('Error: no new point')
            return True
    if n == 9999999:
        return True
    return False
