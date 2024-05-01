"""""""""
Algorithm Class File:
This file defines several classes that are used throughout the optimization methods

funct: The singular input to the optimization function with all the information needed to optimize such as,
    - A defined function
    - Descent direction & line search method (if this is a genetic method use 'min' or 'max')
    - Initial input and parameters
    - Iteration print incidence

stor: The storage object containing all the information gathered throughout the optimization methods

para: The parameter data to pass into funct such as,
    - Line search constants
    - Function data
    - Function parameters
    - Constraint and boundary functions
    - (pr) Option to store the funct class
    
quad: Constraint information in terms of equality and lower bound inequality constraints
    - E/I are equality/inequality functions
    - gE/gI are gradients functions
    - be/bi are the constants which restrict E/I
"""""""""


class funct:
    def __init__(self, function, method, linesearch, input, para, printNum):
        self.function = function
        self.method = method
        self.linesearch = linesearch
        self.input = input
        self.para = para
        self.print = printNum


class stor:
    def __init__(self):
        self.inp = []
        self.val = []
        self.grad = []
        self.norm = []
        self.H = 0
        self.invH = 0
        self.dirt = 0
        self.delta = 1
        self.S = []
        self.Y = []
        self.rho = []


class para:
    def __init__(self, c1, c2, data, parameter, constraint, boundary, pr):
        self.c1 = c1
        self.c2 = c2
        self.data = data
        self.parameter = parameter
        self.constraint = constraint
        self.boundary = boundary
        self.pr = pr


class quad:
    def __init__(self, E, gE, be, I, gI, bi):
        self.E = E
        self.gE = gE
        self.be = be
        self.I = I
        self.gI = gI
        self.bi = bi
