import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
from math import factorial
import numpy as np
import matplotlib.pyplot as plt

class Hermite(object):

    def __init__(self, n):
        self.n = n

    def eval(self, x):
        y = 0
        for m in range(self.n // 2 + 1):
            y += (-1)**m * x ** (self.n - 2*m) / (factorial(m)*factorial(self.n - 2*m) * 2 ** m)

        return factorial(self.n) * y
