import numpy as np

"""""""""
This file contains the line search methods for the optisolve.py file algorithm.
Called from the methods.py file.
"""""""""


def linesearch(pr, stor, n, alpha):
    # Check line search method
    if pr.linesearch == 'armijo':
        # Begin loop to find the next input
        for k in range(1000):
            # Find new point in descent direction
            newInp = stor.inp[n] + (1 / 2) ** k * stor.dirt
            newObj = pr.function(newInp, pr.para, 0)
            # Check Armijo conditions and store function information
            if newObj <= stor.val[n] + pr.para.c1 * (1 / 2) ** k * np.dot(stor.dirt, stor.grad[n]):
                pr.input = newInp
                stor.inp.append(pr.input)
                f, g = pr.function(pr.input, pr.para, 3)
                stor.val.append(f)
                stor.grad.append(np.array(g))
                stor.norm.append(np.linalg.norm(g))
                break
            # Check to iteration failure
            if k == 999:
                return print('Failed to iterate')
    if pr.linesearch == 'strongwolfe':
        # Begin loop for next input
        for k in range(1000):
            # Find new point in descent direction
            newInp = stor.inp[n] + alpha * stor.dirt
            armijo = stor.val[n] + pr.para.c1 * alpha * np.dot(stor.dirt, stor.grad[n])
            obj = pr.function(newInp, pr.para, 2)
            newObj = obj[0]
            newGrad = obj[1]
            # Check Strong Wolfe conditions and store function information
            if newObj <= armijo and - np.dot(stor.dirt, newGrad) <= - pr.para.c2 * np.dot(stor.dirt, stor.grad[n]):
                pr.input = newInp
                f, g = pr.function(pr.input, pr.para, 3)
                stor.inp.append(pr.input)
                stor.val.append(f)
                stor.grad.append(np.array(g))
                stor.norm.append(np.linalg.norm(g))
                break
            # Determine whether to backtrack for move forward
            if newObj > armijo:
                alpha *= 1 / 7
            else:
                alpha *= 3
            # Check for iteration failure
            if k == 999:
                return print('Failed to iterate')
    if pr.linesearch == 'TRdog':
        # Check for appropriate method
        if pr.method == 'GD' or pr.method == 'CG':
            return print('Error: Must use a Quasi-Newton method for a Trust Region search')
        # Find Newton step and Cauchy step
        PN = stor.dirt
        nPN = np.linalg.norm(PN)
        PC = - (stor.norm[n] ** 2) / (np.dot(stor.grad[n], np.dot(stor.H, stor.grad[n]))) * stor.grad[n]
        nPC = np.linalg.norm(PC)
        # Initial trust region size
        stor.delta = nPN
        # Begin loop for next input
        for k in range(1000):
            if stor.delta <= 10 ** (-4) or stor.delta >= 10 ** 4:
                stor.delta = 10
            # Determine new input based on delta and the dog leg path
            if stor.delta >= nPN:
                modelInp = PN
                newInp = stor.inp[n] + modelInp
            elif stor.delta <= nPC:
                modelInp = - stor.delta / stor.norm[n] * stor.grad[n]
                newInp = stor.inp[n] + modelInp
            elif stor.delta < nPN and stor.delta > nPC:
                costheta = min(np.dot(PC / nPC, PN / nPN), 1)
                theta = np.arccos(costheta)
                alpha = (np.sqrt(stor.delta ** 2 - nPC ** 2 * np.sin(theta) ** 2) + nPC ** 2 * np.cos(theta)) / np.linalg.norm(PN - PC)
                modelInp = PC + alpha * (PN - PC)
                newInp = stor.inp[n] + modelInp
            # Evaluate the model function
            mP = stor.val[n] + np.dot(stor.grad[n], modelInp) + 1 / 2 * np.dot(modelInp, np.dot(stor.H, modelInp))
            # Evaluate the function
            newObj = pr.function(newInp, pr.para, 0)
            # Compare function and model differences
            ratioT = stor.val[n] - newObj
            ratioB = stor.val[n] - mP
            ratio = ratioT / ratioB
            # Correction for false positives
            if ratioT < 0 and ratioB < 0:
                ratio = - ratio
            # Determine value of improvement
            if ratio < 1 / 4:
                stor.delta = stor.delta / 4
            if ratio > 3 / 4:
                stor.delta = stor.delta * 3
            # If improvement is achieved, use current point as next iterate
            if ratio > 1 / 25:
                pr.input = newInp
                f, g = pr.function(pr.input, pr.para, 3)
                stor.inp.append(pr.input)
                stor.val.append(f)
                stor.grad.append(np.array(g))
                stor.norm.append(np.linalg.norm(g))
                break
            # Check for iteration failure
            if k == 999:
                return print('Failed to iterate')
    if pr.linesearch == 'TRcong':
        # Check for appropriate method
        if pr.method == 'GD' or pr.method == 'CG':
            return print('Error: Must use a Quasi-Newton method for a Trust Region search')
        # Begin loop for next input
        for k in range(1000):
            # Initialize trust region algorithm
            if stor.delta <= 10 ** (-4) or stor.delta >= 10 ** 4:
                stor.delta = 10
            z = 0
            r = stor.grad[n]
            d = -r
            # Find a new point on the model function
            for j in range(1000):
                prod = np.dot(d, np.dot(stor.H, d))
                nd = np.linalg.norm(d)
                if prod <= 0:
                    nz = np.linalg.norm(z)
                    if nz == 0:
                        modelInp = stor.delta / nd * d
                        newInp = stor.inp[n] + modelInp
                    else:
                        costheta = min(np.dot(z / nz, d / nd), 1)
                        theta = np.arccos(costheta)
                        tilde = (np.sqrt(stor.delta ** 2 - nz ** 2 * np.sin(theta) ** 2) + nz ** 2 * np.cos(theta)) / nd
                        modelInp = z + tilde * d
                        newInp = stor.inp[n] + modelInp
                    break
                alpha = np.dot(r, r)
                znow = z
                z = z + alpha / prod * d
                if np.linalg.norm(z) >= stor.delta:
                    nz = np.linalg.norm(z)
                    nznow = np.linalg.norm(znow)
                    if nznow == 0:
                        modelInp = z / nz * stor.delta
                        newInp = stor.inp[n] + modelInp
                    else:
                        costheta = min(np.dot(znow / nznow, z / nz), 1)
                        theta = np.arccos(costheta)
                        tilde = (np.sqrt(stor.delta ** 2 - nznow ** 2 * np.sin(theta) ** 2) + nznow ** 2 * np.cos(theta)) / nz
                        modelInp = znow + tilde * z
                        newInp = stor.inp[n] + modelInp
                    break
                r = r + alpha / prod * np.dot(stor.H, d)
                if np.linalg.norm(r) < max(1 / 2, stor.norm[n] ** (1 / 2)) * stor.norm[n]:
                    modelInp = z
                    newInp = stor.inp[n] + modelInp
                    break
                beta = np.dot(r, r) / alpha
                d = - r + beta * d
            # Evaluate the functions
            mP = stor.val[n] + np.dot(stor.grad[n], modelInp) + 1 / 2 * np.dot(modelInp, np.dot(stor.H, modelInp))
            newObj = pr.function(newInp, pr.para, 0)
            # Compare function and model differences
            ratioT = stor.val[n] - newObj
            ratioB = stor.val[n] - mP
            ratio = ratioT / ratioB
            # Correction for false positives
            if ratioT < 0 and ratioB < 0:
                ratio = - ratio
            # Determine value of improvement
            if ratio < 1 / 4:
                stor.delta = stor.delta / 4
            if ratio > 3 / 4:
                stor.delta = stor.delta * 3
            # If improvement is achieved, use current point as next iterate
            if ratio > 1 / 25:
                pr.input = newInp
                f, g = pr.function(pr.input, pr.para, 3)
                stor.inp.append(pr.input)
                stor.val.append(f)
                stor.grad.append(np.array(g))
                stor.norm.append(np.linalg.norm(g))
                break
            # Check for iteration failure
            if k == 999:
                return print('Failed to iterate')

