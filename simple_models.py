import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
from hermite_poly import Hermite
import itertools


class simulate(object):
    """
    Class for simulating different dynamics. Normal does an exact simulation
    of Gaussian potential. Potential does an inexact simulation given an arbitrary
    potential.
    """

    def __init__(self, x_0, delta_t, T, n = 5000):
        self.x_0 = np.array(x_0)
        self.delta_t = delta_t
        self.T = T
        self.n = n

    def set_seed(self,n):
        np.random.seed(n)

    def normal(self):
        N = round(self.T/self.delta_t)
        now = np.random.normal(np.zeros(self.n),1)
        storage = np.zeros((self.n, N))
        R = np.multiply(np.random.normal(storage,1), np.sqrt(2*self.delta_t))
        for i in range(N):
            storage[:,i] = now
            now = np.random.normal(now * np.exp(-self.delta_t), (1 - np.exp(-2*self.delta_t)))
        return storage

    def potential(self, V, drift = 0):
        x_n = np.array(self.x_0)
        trajectory = np.array(x_n)
        n = round(self.T/self.delta_t)
        for i in range(0,n):
            xsi = np.random.normal(0,1, len(self.x_0))
            x_n = x_n + V(x_n - drift * self.delta_t*i)*self.delta_t + np.sqrt(2*self.delta_t) * xsi
            trajectory = np.vstack((trajectory, x_n))
        return trajectory

    def potential_lots(self, V):
        N = round(self.T/self.delta_t)
        now = np.random.normal(np.zeros(self.n),1)
        storage = np.zeros((self.n, N))
        R = np.multiply(np.random.normal(storage,1), np.sqrt(2*self.delta_t))
        for i in range(0,N):
            storage[:,i] = now
            now = np.add(np.add(now, np.multiply(V(now), self.delta_t)), R[:,i])
        return storage


'''
A few sample potentials.
'''
def well_well(x):
    return -4*x ** 3 + 4*x

def well(x, wells = [-1,1,0]):
    return -np.sum([20(x - well)*np.exp(-20*(x - well)**2) for well in wells], axis = 0)

def OU(x):
    return -x

def zero(x):
    return 0



class VAC(object):
    """Class to do vac on a trajectory. Uses algorithm described in Klus et al, 2018."""

    def __init__(self, basis, trajectory, time_lag, delta_t, dimension = 1):
        self.basis = basis
        self.N = len(trajectory[0,:])
        self.trajectory = trajectory
        self.time_lag = time_lag
        self.lag = np.rint(time_lag / delta_t).astype(int)
        self.X = np.array([b(trajectory[0:dimension,0:(self.N - self.lag)]) for  b in basis])
        self.Y = np.array([b(trajectory[:,self.lag:]) for b in basis])


    def C_0(self):
        return (np.matmul(self.X, self.X.T) + np.matmul(self.Y, self.Y.T)) / (2*self.N)

    def C_t(self):
        return (np.matmul(self.X, self.Y.T) + np.matmul(self.Y, self.X.T)) / (2 * self.N)



    # def auto_cor(self):
    #     n = self.N
    #     C = np.zeros((n,n))
    #     beginning = self.lag
    #     end = len(self.trajectory) - self.lag
    #     if end <= 0:
    #         return [0]
    #     for i in range(n):
    #         for j in range(n):
    #             i_first = self.basis[i].eval(self.trajectory[0:end])
    #             j_lagged = self.basis[j].eval(self.trajectory[beginning:len(self.trajectory)])
    #
    #             i_lagged = self.basis[i].eval(self.trajectory[beginning:len(self.trajectory)])
    #             j_first = self.basis[j].eval(self.trajectory[0:end])
    #             cor = np.append(i_first * j_lagged, i_lagged * j_first)
    #             C[i][j] = np.mean(cor)
    #     return C
    #
    #
    # def self_cor(self):
    #     n = self.N
    #     C = np.zeros((n,n))
    #     for i in range(n):
    #         for j in range(n):
    #             phi_i = self.basis[i].eval(self.trajectory)
    #             phi_j = self.basis[j].eval(self.trajectory)
    #
    #             cor = phi_i * phi_j
    #             C[i][j] = np.mean(cor)
    #     return C

    def find_eigen(self, m):
        C_t = self.C_t()
        C_0 = self.C_0()
        l = len(self.basis)
        # print(C_t)
        # print(C_0)
        eigvals, eigvecs = eigh(C_t, C_0, eigvals=(l-m,l-1), eigvals_only=False)

        return [eigvals, eigvecs]


'''
functions to deal with function norms and projection norms
'''

def makegrid(endpoint, dimension = 1, n  = 100):
    points_1D = np.linspace(-endpoint,endpoint,n)
    points = itertools.product(points_1D, repeat = dimension)
    return np.array(list(points))

def dot(f, g, distribution):
    return np.sum([np.multiply(f(x), g(x)) for x in distribution]) / len(distribution.T)


def fcn_weighting(fs,weighting):
    def g(x):
        return np.dot(weighting, [f(x) for f in fs])
    return g

def L2subspaceProj_d(w_f, w_g, distribution, Phi_f = False, Phi_g = False, basis_f = False, basis_g = False):
    if len(w_f) != len(w_g):
        return "Subspaces have different dimensions."
    if type(Phi_f) == bool:
        if type(basis_f) == bool:
            return "Must have a basis for f's supplied if no Phi_f given."
        else:
            A = np.array([f(distribution) for f in basis_f])
            # print(A)

    if type(Phi_g) == bool:
        if type(basis_f) == bool:
            return "Must have a basis for g's supplied if no Phi_g given."
        else:
            B = np.array([g(distribution) for g in basis_g])

    else:
        A = Phi_f
        B = Phi_g

    A = np.dot(w_f, A)
    # print(A)

    B = np.dot(w_g, B)
    # print(B)

    P = np.dot(A,B.T)
    # print(P)

    N = np.tensordot(np.sqrt(np.sum(np.square(A), axis = 1)), np.sqrt(np.sum(np.square(B), axis = 1)), axes = 0) ** -1
    # print(N)

    P = np.multiply(P, N)
    svd = np.linalg.svd(P)[1]
    print("the svd are:", svd)

    if len(w_f) < np.sum(np.square(svd)) < len(w_f) + 1e-15:
        return 0
    elif np.sum(np.square(svd)) > len(w_f):
        return "Singular values are too large, probably not normalized correctly."
    else:
        return np.sqrt(len(w_f) - np.sum(np.square(svd)))


def f(x):
    return x **2

def h(x):
    return x

t = np.linspace(-2,2,5)

print(L2subspaceProj_d(w_f = np.array([[1,1],[1,0]]), w_g = np.array([[1,0],[0,1]]),
                distribution = t, basis_f = np.array([f,h]), basis_g = np.array([f,h])))
