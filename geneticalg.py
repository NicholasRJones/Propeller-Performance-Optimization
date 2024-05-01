import numpy as np
from Optimization.Algorithm import sampling as sp
import random as rd

"""""""""
This file contains the code for the genetic algorithm. This is a derivative free method in which we may guarantee convergence
on the global solution in infinite time for sufficiently smooth function. 

This algorithm takes in the funct class from the classy.py file and returns the optimal input for the functions

In funct.para.method insert whether you wish to minimize or maximize your given function. 

You are also able to specify a feasible region. Simply create a function in funct.para.boundary to return True if the input if feasible
or return False if the input is infeasible.
"""""""""


def scatter(input, points):
    # Obtain a uniform sample from the initial input using the halton sequence method
    input = np.array(input)
    scat = []
    sampling = sp.halton(points, len(input), 12345678) / 100
    for k in range(points):
        scat.append(input + sampling[k].flatten())
    return scat


def fitness(sample, pr):
    # Determine fitness values of current sample
    fit = []
    for k in range(len(sample)):
        fit.append([sample[k], pr.function(sample[k], pr.para, 0)])
    fmax = max(fit[k][1] for k in range(len(sample)))
    for k in range(len(sample)):
        fit[k][1] = fmax - fit[k][1] + 1
    return fit


def conditions(best, iteration):
    # Stopping criteria. Check for stagnation which is determined by no improvement in 'n' iterations
    n = 10000
    if iteration > n:
        if best[iteration - 1] == best[iteration - n]:
            print('Stagnation. Terminating')
            return True
    return False


def sort(values, method):
    # Sort the sample to find best inputs
    if method == 'min':
        for j in range(len(values)):
            k = j
            while k < len(values):
                if values[k][1] > values[j][1]:
                    values[k], values[j] = values[j], values[k]
                    k = j
                k += 1
    if method == 'max':
        for j in range(len(values)):
            k = j
            while k < len(values):
                if values[k][1] < values[j][1]:
                    values[k], values[j] = values[j], values[k]
                    k = j
                k += 1
    return values


def genetic(pr):
    # Determine sample size
    size = 20
    # Retrieve initial sample
    sample = scatter(pr.input, size)
    # Check for feasibility
    for i in range(size):
        if not pr.para.boundary(sample[i]):
            return print('Initial sample not in domain')
    # Obtain fitness values of sample
    values = fitness(sample, pr)
    best = []
    for iteration in range(100000):
        # Sort the sample
        values = sort(values, pr.method)
        # Store best sample
        best.append(pr.function(values[0][0], pr.para, 0))
        # Print progress
        if iteration % pr.print == 0:
            print(str(iteration) + '___' + str(best[iteration]))
        # Check stopping criteria
        if conditions(best, iteration):
            print(str(iteration) + '___' + str(sample[0]) + '___' + str(best[iteration]))
            return sample[0]
        # Start next sample, keeping best 5 from the last generation
        sample = [values[0][0], values[1][0], values[2][0], values[3][0], values[4][0]]
        # Create new samples
        while len(sample) < size:
            # Determine parents for new sample
            rand1 = rd.random()
            rand2 = rd.random()
            sum = values[0][1] + values[1][1] + values[2][1] + values[3][1] + values[4][1]
            F1 = (values[0][1]) / sum
            F2 = F1 + (values[1][1]) / sum
            F3 = F2 + (values[2][1]) / sum
            F4 = F3 + (values[3][1]) / sum
            F = [F1, F2, F3, F4, 1]
            for j in range(len(F)):
                if rand1 < F[j]:
                    U = values[j][1]
                    Xu = values[j][0]
                    break
            for j in range(len(F)):
                if rand2 < F[j]:
                    V = values[j][1]
                    Xv = values[j][0]
            theta = U / (U + V)
            sample.append(theta * Xu + (1 - theta) * Xv)
            # Add possibility of a mutation
            for i in range(len(pr.input)):
                rand3 = rd.random()
                if rand3 < .5:
                    sample[len(sample) - 1][i] += rd.gauss(0, 0.0001)
            # If this new sample is not feasible, reject it
            if not pr.para.boundary(sample[len(sample) - 1]):
                sample.pop()
        # Determine fitness of next generation
        values = fitness(sample, pr)
    # Maximum iteration achieved
    print('Failed to converge')
    print(str(iteration) + '___' + str(sample[0]) + '___' + str(best[iteration]))
    return sample[0]
    
