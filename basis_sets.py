import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
import matplotlib.path as path
from hermite_poly import Hermite
from simple_models import simulate, VAC, well_well

class indicator(object):
    """docstring for indicator."""

    def __init__(self, fineness, endpoint, center):
        self.fineness = fineness
        self.endpoint = endpoint
        self.center = np.array(center)
        self.length = endpoint/fineness

    def eval(self, x):
        left = self.center - self.length / 2
        right = self.center  +  self.length / 2
        return np.array([int((v >= left).all() and (v < right).all()) for v in np.array(x)])



# f = indicator(10, 10, [0,0])
# 
# print(f.eval([[0,0]]), f.eval([[0,1]]), f.eval([[-1,.5]]))
#
# f = indicator(10,10,0)
#
# print(f.eval([0,1]))
