import numpy as np
from Optimization.Algorithm import classy as cl, optisolve as op

"""""""""
This file contains the Augmented Lagrange algorithm which is an optimization method for constrained optimization. 
This algorithm takes the funct input from the classy.py file and creates a new funct class to pass into the optisolve.py file algorithm
and optimizes a new objective function determined by the constraint violations.
"""""""""


def Lagrangeobj(input, para, p):
    # New objective function
    # Evaluate constraints
    E, gE, be, I, gI, bi = para.constraint.E, para.constraint.gE, para.constraint.be, para.constraint.I, para.constraint.gI, para.constraint.bi
    # Grab parameter information
    lmbda, mu, n, m = para.data
    # Separate original inputs from slack variables
    x = input[:len(input) - n]
    y = input[len(input) - n:]
    # Calculate original objective functions information
    f, g = para.pr.function(x, para.pr.para, 2)
    # Evaluate violations
    CE = E(x) - be
    CI = I(x) - bi - y ** 2
    CW = np.concatenate((CE, CI))
    # Update parameters
    if p == 3:
        lmbda = lmbda - mu * CW
        if mu < 10 ** 12:
            mu *= 5
        para.data = [lmbda, mu, n, m]
    # Evaluate new function
    L = f - np.dot(lmbda, CW) + mu / 2 * (np.dot(CW, CW))
    if p == 0:
        return L
    if p > 0:
        # Determine gradient information
        Y = np.diag(y)
        fung = np.concatenate((g, np.zeros(n)))
        cstr = np.block([[np.transpose([gE(x)]), np.transpose(gI(x))], [np.zeros((n, m)), -2 * Y]])
        prod = lmbda - mu * CW
        LG = fung - np.matmul(cstr, prod).reshape(len(x) + n)
        if p > 1:
            return L, LG
        return LG


def Lagrange(pr):
    # Determine constraint sizes
    n, m = len(pr.para.constraint.I(pr.input)), len(pr.para.constraint.E(pr.input))
    # Create slack variables
    input = np.concatenate((pr.input, np.ones(n)))
    # Initialize the lambda parameter and mu size
    lmbda = np.zeros(m + n)
    mu = 100000000
    # Create new function class to optimize and store old function class data
    para = cl.para(pr.para.c1, pr.para.c2, [lmbda, mu, n, m], pr.para.pr, pr.para.constraint, pr.para.boundary, pr)
    pR = cl.funct(Lagrangeobj, pr.method, pr.linesearch, input, para, pr.print)
    # Optimize!
    x = op.optimize(pR).input
    # Return the optimal parameters for the original function
    x = x[:len(x) - n]
    print(str(x) + '___' + str(pr.function(x, pr.para, 0)))
    return x
