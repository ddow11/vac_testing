import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
import matplotlib.path as path
from hermite_poly import Hermite
from models_and_functions import simulate, VAC, well_well
import scipy.stats

class indicator(object):
    """Makes an indicator function. Meant to be used to make a grid of indicator
    functions on the square [-N, N]^d, where d is the dimension. The dimension
    is implicitly determined by the dimension of the center."""

    def __init__(self, fineness, endpoint, center):
        self.fineness = fineness
        self.endpoint = endpoint
        self.center = np.array(center)
        self.length = endpoint/(fineness - 1)


    def eval(self, x):
        x = np.matrix(x)
        left = np.tile(np.matrix(self.center - self.length).T, (1,len(x[0,:])))
        right = np.tile(np.matrix(self.center + self.length).T, (1,len(x[0,:])))
        return np.squeeze(np.array(np.all((x >= left) & (x < right), axis = 0).astype(int)))


    def to_fcn(self):
        def g(x):
            return self.eval(x)
        return g


def indicator(x, left, right):
    x = np.squeeze(x)
    return ((x >= left) & (x < right)).astype(int)

def compose(function, variables):
    def g(x):
        return function(x, *variables)
    return g

def makeIndicators(N):
    result = []
    endpoints = [scipy.stats.norm.ppf(i/N) for i in range(0,N+1)]
    results = [compose(indicator, [endpoints[i], endpoints[i+1]]) for i in range(N)]
    return results


# f = indicator(10, 10, [0,0])
#
# print(f.eval([[0,0]]), f.eval([[0,1]]), f.eval([[-1,.5]]))
#
# f = indicator(10,10,0)
#
# print(f.eval([0,1]))
