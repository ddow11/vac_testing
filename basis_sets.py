import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
import matplotlib.path as path
from hermite_poly import Hermite
from models_and_functions import simulate, VAC, well_well

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




# f = indicator(10, 10, [0,0])
#
# print(f.eval([[0,0]]), f.eval([[0,1]]), f.eval([[-1,.5]]))
#
# f = indicator(10,10,0)
#
# print(f.eval([0,1]))
