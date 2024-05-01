import numpy as np
import pandas as pd
from Optimization.Algorithm import geneticalg as ga, classy as cl, Lagrange as L

"""""""""
def quadfit(input, para, p):
    n = para.parameter
    A = np.zeros((n, n))
    k = 0
    for j in range(n):
        for i in range(n):
            if i >= j:
                A[j][i] = input[k]
                k += 1
    b = np.array(input[k:k + n])
    c = np.array(input[len(input) - 1])
    dist = 0
    for j in range(len(para.data)):
        x = para.data[j][:len(para.data[j]) - 1]
        y = para.data[j][len(para.data[j]) - 1]
        dist += (np.dot(x, np.matmul(A, x)) + np.dot(b, x) + c - y) ** 2
    return dist


def true(input):
    return True


data = np.array(pd.read_csv(r'../Data/PropData.csv'))
fdata = []

for j in range(len(data)):
    fdata.append([data[j][0], data[j][1], data[j][2], data[j][4]])

fdata = np.array(fdata)

input = np.zeros(14)

para = cl.para(0.0001, 0.19, fdata, 3, true)
pr = cl.funct(quadfit, 'min', '', input, para, 1000)

ga.genetic(pr)

Tcoefficients = [-0.01473516, 0.74309256, 0.14202548, -0.05620069, 0.08078697, -0.05166961,
                 0.06937329, 0.17872796, 0.09526768, -0.02141651, 0.01061365, -0.00748409,
                 0.04928439, -0.20615375]
Qcoefficients = [4.41528783e-01, -1.66715028e-02, -1.41756872e-01, 3.63016357e-01,
                 -1.31056301e-02, 1.25249239e-01, -8.47397857e-01, -1.96718516e+00,
                 2.50971386e+00, -3.55951388e-03, -3.23630078e-03, 1.67436083e-02,
                 1.30467883e-03, 1.37469823e+00]
"""""""""


def objective(input, para, p):
    n = len(input)
    coefficients = [4.41528783e-01, -1.66715028e-02, -1.41756872e-01, 3.63016357e-01,
                    -1.31056301e-02, 1.25249239e-01, -8.47397857e-01, -1.96718516e+00,
                    2.50971386e+00, 1.37469823e+00]
    A = np.zeros((n, n))
    k = 0
    for j in range(n):
        for i in range(n):
            if i >= j:
                A[j][i] = coefficients[k]
                k += 1
    A = (A + np.transpose(A)) / 2
    b = np.array(coefficients[k:k + n])
    c = np.array(coefficients[len(input) - 1])
    Q = np.dot(input, np.matmul(A, input)) / 2 + np.dot(b, input) + c
    gQ = np.matmul(A, input) + b
    f = input[0] * input[2] * Q
    if p == 0:
        return f
    if p > 0:
        g = np.array([input[2] * Q, 0, input[0] * Q]) + input[0] * input[2] * gQ
        if p > 1:
            return f, g
        return g


def E(input):
    n = len(input)
    coefficients = [-0.01473516, 0.74309256, 0.14202548, -0.05620069, 0.08078697, -0.05166961,
                    0.06937329, 0.17872796, 0.09526768, -0.20615375]
    A = np.zeros((n, n))
    k = 0
    for j in range(n):
        for i in range(n):
            if i >= j:
                A[j][i] = coefficients[k]
                k += 1
    A = (A + np.transpose(A)) / 2
    b = np.array(coefficients[k:k + n])
    c = np.array(coefficients[len(input) - 1])
    f = np.dot(input, np.matmul(A, input)) / 2 + np.dot(b, input) + c
    return np.array([f])


def gE(input):
    n = len(input)
    coefficients = [-0.01473516, 0.74309256, 0.14202548, -0.05620069, 0.08078697, -0.05166961,
                    0.06937329, 0.17872796, 0.09526768, -0.20615375]
    A = np.zeros((n, n))
    k = 0
    for j in range(n):
        for i in range(n):
            if i >= j:
                A[j][i] = coefficients[k]
                k += 1
    A = (A + np.transpose(A)) / 2
    b = np.array(coefficients[k:k + n])
    g = np.matmul(A, input) + b
    return g


be = np.array([6])


def I(input):
    return np.matmul(np.array([[-1, 0, 0],
                               [0, -1, 0],
                               [0, 0, -1],
                               [1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]]), input)


def gI(input):
    return np.array([[-1, 0, 0],
                     [0, -1, 0],
                     [0, 0, -1],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])


bi = np.array([-4, -10, -7, 1, 0, 1])

input = [3, 3, 3]

parameters = []

quad = cl.quad(E, gE, be, I, gI, bi)
para = cl.para(0.0001, 0.19, 0, parameters, quad, 0, 0)
pr = cl.funct(objective, 'BFGS', 'strongwolfe', input, para, 1)

x = L.Lagrange(pr)
print()
print('T - T0 = ' + str(E(x) - be))
