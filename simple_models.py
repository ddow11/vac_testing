import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
from hermite_poly import Hermite
import itertools


class simulate(object):
    """docstring for simulate."""

    def __init__(self, x_0, delta_t, T):
        self.x_0 = np.array(x_0)
        self.delta_t = delta_t
        self.T = T

    def set_seed(self,n):
        np.random.seed(n)

    def normal(self):
        x_n = np.array(self.x_0)
        trajectory = np.array(x_n)
        n = round(self.T/self.delta_t)
        for i in range(1,n):
            t = self.delta_t * i
            x_n = np.random.normal(x_n * np.exp(-t), 1 - np.exp(-2*t))
            trajectory = np.vstack((trajectory, x_n))
        return trajectory


    def potential(self, V):
        x_n = np.array(self.x_0)
        trajectory = np.array(x_n)
        n = round(self.T/self.delta_t)
        for i in range(0,n):
            xsi = np.random.normal(0,1, len(self.x_0))
            x_n = x_n + V(x_n)*self.delta_t + np.sqrt(2*self.delta_t) * xsi
            trajectory = np.vstack((trajectory, x_n))
        return trajectory

def well_well(x):
    return - x ** 3  +  x

# print(x)
# plt.plot(x)
# plt.show()

class VAC(object):
    """docstring for Cor."""

    def __init__(self, basis, trajectory, lag):
        self.basis = basis
        self.trajectory = trajectory
        self.lag = lag

    def auto_cor(self):
        n = len(self.basis)
        C = np.zeros((n,n))
        begining = self.lag
        end = len(self.trajectory) - self.lag
        if end <= 0:
            return 0
        for i in range(n):
            for j in range(n):
                i_first = self.basis[i].eval(self.trajectory[0:end])
                j_lagged = self.basis[j].eval(self.trajectory[begining:len(self.trajectory)])

                i_lagged = self.basis[i].eval(self.trajectory[begining:len(self.trajectory)])
                j_first = self.basis[j].eval(self.trajectory[0:end])
                cor = np.append(i_first * j_lagged, i_lagged * j_first)
                C[i][j] = np.mean(cor)
        return C


    def self_cor(self):
        n = len(self.basis)
        C = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                phi_i = self.basis[i].eval(self.trajectory)
                phi_j = self.basis[j].eval(self.trajectory)

                cor = phi_i * phi_j
                C[i][j] = np.mean(cor)
        return C

    def find_eigen(self, m):
        C_t = self.auto_cor()
        C_0 = self.self_cor()
        print(C_t)
        print(C_0)
        eigvals, eigvecs = eigh(C_t, C_0, eigvals_only=False)

        return eigvals[::-1][0:m], eigvecs[::-1][0:m]



def make_grid(endpoint, dimension = 1, n  = 100):
    points_1D = np.array([x for x in np.linspace(-endpoint,endpoint,n)])
    points = itertools.product(points_1D, repeat = dimension)
    return np.array(list(points))

def dot(f, g, endpoint, dimension = 1, n = 100):
    points = make_grid(endpoint, dimension, n)
    return sum([f(x)*g(x) for x in points]) / n**dimension


def f(x):
    return sum(np.array(x)**2)

def g(x):
    return sum(np.cos(np.array(x)))

# print(dot(f,g,10,dimension = 2))
