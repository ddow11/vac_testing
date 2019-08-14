import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
import matplotlib.path as path
from hermite_poly import Hermite
from simple_models import simulate, VAC, well_well

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
        left = self.center - self.length
        right = self.center  +  self.length
        return np.array([int((v >= left).all() and (v < right).all()) for v in np.array(x)])


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
