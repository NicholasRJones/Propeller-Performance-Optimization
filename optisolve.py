"""""""""
Algorithm Algorithm File:
This is the main file for the optimization process. This function will call each other file and iterate until
stopping criteria is achieved.
The optimization process is as follows:
    - Check criteria
    - Choose a method
    - Repeat 
"""""""""
import Optimization.Algorithm.classy as classy
import Optimization.Algorithm.criterion as criterion
import Optimization.Algorithm.methods as methods


def optimize(pr):
    # Check line search parameter validity
    if pr.para.c1 >= pr.para.c2:
        return print('Invalid parameters: pr.para.c1 must be less than pr.para.c2')
    # Create storage class
    stor = classy.stor()
    # Begin main optimization loop
    for n in range(10000000):
        # Check criteria
        c = criterion.criterion(stor, n)
        # Proceed if we have failed criteria
        if not c:
            # Print function information at iteration mod pr.print
            if n % pr.print == 0 and n > 0:
                print(str(n) + '___' + str(stor.val[len(stor.val) - 1]))
            # Choose preferred method
            match pr.method:
                case 'GD':
                    methods.gradient(pr, stor, n)
                case 'CG':
                    methods.conjugate(pr, stor, n)
                case 'BFGS':
                    methods.BFGS(pr, stor, n)
                case 'SR1':
                    methods.SR1(pr, stor, n)
                case 'LBFGS':
                    methods.LBFGS(pr, stor, n)
        # If we pass criteria, print results and stop
        else:
            print(str(n) + '___' + str(stor.val[len(stor.val) - 1]) + '___' + str(pr.input))
            return pr

