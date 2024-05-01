import numpy as np
from scipy.stats import qmc

"""""""""
This file contains sampling methods for various algorithms
"""""""""


def get_prime():
    def check_prime(num):
        for integer in range(2, int(num ** 0.5) + 1):
            if (num % integer) == 0:
                return False
            return True

    prime = 3

    while (1):
        if check_prime(prime):
            yield prime
        prime += 2


def vd_corput(num, base=2):
    vdcNum, denominator = 0, 1
    while num:
        denominator *= base
        num, remainder = divmod(num, base)
        vdcNum += remainder / float(denominator)
    return vdcNum


def halton(size, dimension, offset=0):
    sequence = []
    sample = []
    prime = get_prime()
    for dim in range(dimension):
        base = next(prime)
        sequence.append([vd_corput(integer + offset, base) for integer in range(size)])
    for integer in range(size):
        sample.append([[sequence[dim][integer]] for dim in range(dimension)])
    sample = np.array(sample)
    return sample


def computer(size, dimension):
    sample = np.random.random(size * dimension).reshape(size * dimension, 1).reshape(size, dimension, 1)
    return sample


def latin(size, dimension):
    sampler = qmc.LatinHypercube(d=dimension)
    sample = sampler.random(n=size).reshape(size * dimension, 1).reshape(size, dimension, 1)
    return sample

