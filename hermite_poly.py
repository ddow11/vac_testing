import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
from math import factorial
import numpy as np
import matplotlib.pyplot as plt

class Hermite(object):

    def __init__(self, n, d = 0):
        self.n = n
        self.d = d

    def eval(self, x):
        x = np.matrix(x)[self.d,:]
        y = 0
        for m in range(self.n // 2 + 1):
            y += (-1)**m * np.power(x, (self.n - 2*m)) / (factorial(m)*factorial(self.n - 2*m) * 2 ** m)

        return np.squeeze(np.array(factorial(self.n) * y))

    def to_fcn(self):
        def g(x):
            return self.eval(x)
        return g

class Poly(object):
    """docstring for Poly."""

    def __init__(self, n):
        self.n = n

    def eval(self, x):
        return np.sum(x ** self.n, axis = 0)

    def to_fcn(self):
        def g(x):
            return self.eval(x)
        return g
