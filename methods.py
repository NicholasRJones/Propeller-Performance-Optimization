"""""""""
Algorithm Algorithm File:
This file contains each method used in the optimization function such as,
    - Gradient descent [GD]
    - Conjugate gradient [CG]
    - Quasi-Newton [BFGS]
    - Quasi-Newton [LBFGS]
    - Quasi-Newton [SR1]
and calls line search methods,
    - Armijo [armijo]
    - Strong Wolfe [strongWolfe]
    - Trust region [TRdog]
    - Trust region [TRcong]
"""""""""

import numpy as np
from Optimization.Algorithm import linesearch as ls


def gradient(pr, stor, n):
    # Initialize with function information at initial input
    if n == 0:
        initial = pr.function(pr.input, pr.para, 3)
        stor.inp.append(pr.input)
        stor.val.append(initial[0])
        stor.grad.append(np.array(initial[1]))
        stor.norm.append(np.linalg.norm(stor.grad[n]))
    stor.dirt = -stor.grad[n]
    # Perform the line search
    ls.linesearch(pr, stor, n, 1 / pr.para.c2 ** 2)
    # Store the next descent direction


def conjugate(pr, stor, n):
    # Check line search constant conditions
    if pr.para.c2 >= 1 / 2:
        return print('Invalid parameters: pr.para.c2 must be less than a half')
    # Initialize with function information at initial input
    if n == 0:
        initial = pr.function(pr.input, pr.para, 3)
        stor.inp.append(pr.input)
        stor.val.append(initial[0])
        stor.grad.append(np.array(initial[1]))
        stor.norm.append(np.linalg.norm(stor.grad[n]))
        stor.dirt = -stor.grad[n]
    else:
        # Determine descent direction described by the conjugate gradient method mod 20
        # If iteration count is 0 mod 20 then reset descent direction with current gradient information
        match n % 20:
            case 0:
                stor.dirt = -stor.grad[n]
            case _:
                B_n = (stor.norm[n] / stor.norm[n - 1]) ** 2
                stor.dirt = -stor.grad[n] + B_n * stor.dirt
    # Perform the line search
    ls.linesearch(pr, stor, n, 1 / pr.para.c2 ** 2)


def BFGS(pr, stor, n):
    # Initialize with function information at initial input
    if n == 0:
        initial = pr.function(pr.input, pr.para, 3)
        stor.inp.append(pr.input)
        stor.val.append(initial[0])
        stor.grad.append(np.array(initial[1]))
        stor.norm.append(np.linalg.norm(stor.grad[n]))
        stor.H = abs(stor.val[n]) * np.identity(len(stor.grad[n]))
        stor.invH = abs(stor.val[n]) * np.identity(len(stor.grad[n]))
        stor.dirt = -stor.invH.dot(stor.grad[n])
    else:
        # This is the line search method, approximate the inverse hessian
        s_n = np.array(stor.inp[n] - stor.inp[n - 1])
        y_n = stor.grad[n] - stor.grad[n - 1]
        dot = np.dot(s_n, y_n)
        I = np.identity(len(stor.grad[n]))
        stor.invH = np.matmul(np.matmul((I - np.outer(s_n, y_n) / dot), stor.invH),
                      (I - np.outer(y_n, s_n) / dot)) + np.outer(s_n, s_n) / dot
        if pr.linesearch == 'TRdog' or pr.linesearch == 'TRcong':
            # This is the trust region method, approximate the hessian
            HT = np.transpose(stor.H)
            prodT = np.matmul(stor.H, np.matmul(np.outer(s_n, s_n), HT))
            prodB = np.dot(s_n, np.matmul(stor.H, s_n))
            stor.H = stor.H + np.outer(y_n, y_n) / dot - prodT / prodB
            initH = stor.H + 0
        if not pr.linesearch == 'TRcong':
            # Check for positive definiteness on matrix for direction / Newton step
            idot = np.matmul(np.matmul(np.transpose(stor.grad[n]), stor.invH), stor.grad[n])
            if idot <= 0:
                print('Hessian update: positive definite')
                m = (1 / 100 - idot) / (stor.norm[n] ** 2)
                stor.dirt = -(stor.invH + m * np.identity(len(stor.grad[n]))).dot(stor.grad[n])
                if pr.linesearch == 'TRdog':
                    stor.H = stor.H + m * np.identity(len(stor.grad[n]))
            else:
                stor.dirt = -stor.invH.dot(stor.grad[n])
    # Find next iterate
    ls.linesearch(pr, stor, n, 1)
    # Restore original hessian
    if pr.linesearch == 'TRdog' and n > 0:
        stor.H = initH


def SR1(pr, stor, n):
    # Initialize with function information at initial input
    if n == 0:
        initial = pr.function(pr.input, pr.para, 3)
        stor.inp.append(pr.input)
        stor.val.append(initial[0])
        stor.grad.append(np.array(initial[1]))
        stor.norm.append(np.linalg.norm(stor.grad[n]))
        stor.H = abs(stor.val[n]) * np.identity(len(stor.grad[n]))
        stor.invH = abs(stor.val[n]) * np.identity(len(stor.grad[n]))
        stor.dirt = -stor.invH.dot(stor.grad[n])
    else:
        # Approximate the inverse hessian
        s = stor.inp[n] - stor.inp[n - 1]
        y = stor.grad[n] - stor.grad[n - 1]
        w = s - np.dot(stor.invH, y)
        stor.invH = stor.invH + np.outer(w, w) / np.dot(w, y)
        if pr.linesearch == 'TRdog' or pr.linesearch == 'TRcong':
            # This is the trust region method, approximate the hessian
            z = y - np.dot(stor.H, s)
            stor.H = stor.H + np.outer(z, z) / np.dot(z, s)
            initH = stor.H + 0
        # Check for positive definiteness on matrix and find direction / Newton step
        if not pr.linesearch == 'TRcong':
            idot = np.dot(stor.grad[n], np.dot(stor.invH, stor.grad[n]))
            if idot <= 0:
                print('Hessian update: positive definite')
                m = (1 / 100 - idot) / (stor.norm[n] ** 2)
                stor.dirt = -(stor.invH + m * np.identity(len(stor.grad[n]))).dot(stor.grad[n])
                if pr.linesearch == 'TRdog':
                    stor.H = stor.H + m * np.identity(len(stor.grad[n]))
            else:
                stor.dirt = -stor.invH.dot(stor.grad[n])
    # Find next iterate
    ls.linesearch(pr, stor, n, 1)
    # Restore original hessian
    if pr.linesearch == 'TRdog' and n > 1:
        stor.H = initH


def LBFGS(pr, stor, n):
    if pr.linesearch == 'TRdog' or pr.linesearch == 'TRcong':
        # Check for valid method
        return print('Error: LBFGS uses no hessian')
    # Initialize with function information at initial input
    if n == 0:
        initial = pr.function(pr.input, pr.para, 3)
        stor.inp.append(pr.input)
        stor.val.append(initial[0])
        stor.grad.append(np.array(initial[1]))
        stor.norm.append(np.linalg.norm(stor.grad[n]))
        stor.dirt = - stor.grad[n]
    else:
        # Determine descent direction
        m = n - 1
        if n > 7:
            m = 6
            stor.S.pop(0)
            stor.Y.pop(0)
            stor.rho.pop(0)
        stor.S.append(stor.inp[n] - stor.inp[n - 1])
        stor.Y.append(stor.grad[n] - stor.grad[n - 1])
        stor.rho.append(1 / np.dot(stor.S[m], stor.Y[m]))
        q = - stor.grad[n]
        alpha = np.zeros(m + 1)
        for j in range(m + 1):
            alpha[m - j] = stor.rho[m - j] * np.dot(stor.S[m - j], q)
            q = q - alpha[m - j] * stor.Y[m - j]
        gamma = np.dot(stor.S[m], stor.Y[m]) / np.dot(stor.Y[m], stor.Y[m])
        z = gamma * q
        for j in range(m + 1):
            beta = stor.rho[j] * np.dot(stor.Y[j], z)
            z = z + stor.S[j] * (alpha[j] - beta)
        stor.dirt = z + 0
    # Find next iterate
    ls.linesearch(pr, stor, n, 1)



