import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
from hermite_poly import Hermite
import itertools
from scipy.linalg import subspace_angles


class simulate(object):
    """
    Class for simulating different dynamics. Normal does an exact simulation
    of Gaussian potential. Potential does an inexact simulation given an arbitrary
    potential.
    """

    def __init__(self, delta_t, T, x_0 = [0], n = 10):
        '''
        delta_t: float
        T: int
        x_0: array (not needed if doing n > 1)
        n: int
        '''
        self.x_0 = np.array(x_0)
        self.delta_t = delta_t
        self.T = T
        self.n = n

    def set_seed(self,n):
        np.random.seed(n)

    def normal(self, speed = np.array([1]), update = False):
        '''
        Uses exact analytical solution for the OU process: dX = - X dt + √2 dW.
        Speeds correspond to changing r. The function is set to keep the equilibrium
        distribution to be N(0,1). Changing speeds simulates the process
        dX = - r X dt + √(2 / r) dW, where r is the relaxation speed and the √(2 / r)
        factor is such that the equilibrium distribution is still N(0,1).

        speed: int, float, to 1D array
        '''

        if type(speed) == int or type(speed) == float:
            speed = np.array([speed])
        if self.n % len(speed) != 0:
            raise ValueError("Number of samples must be divisible by number of speeds.")

        speed = np.tile(np.array(speed), self.n // len(speed))

        N = round(self.T/self.delta_t)
        now = np.random.normal(np.zeros(self.n),1)
        storage = np.zeros((self.n, N))
        m = np.exp(-speed*self.delta_t)
        sigma = np.sqrt(1 - np.square(m))
        R = np.random.normal(0, np.matrix(sigma).T, [self.n, N])
        update_time = round(.01*N)
        for i in range(N):
            storage[:,i] = now
            now = np.multiply(now, m) + R[:,i]
            if update and i % update_time == 0:
                print(str(i*self.delta_t) + " seconds done out of " + str(self.T))
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

    def potential_lots(self, V, update = False):
        '''
        Give it a potential and potential_lots will give you a trajectory. Updates every 10%
        if desired. For a potential V it simulated the process dX = V d t + √2 dW.
        '''
        N = round(self.T/self.delta_t)
        now = np.random.normal(np.zeros(self.n),1)
        storage = np.zeros((self.n, N))
        R = np.multiply(np.random.normal(storage,1), np.sqrt(2*self.delta_t))
        if update:
            update_time = round(.01 * N)
        for i in range(0,N):
            storage[:,i] = now
            now = np.add(np.add(now, np.multiply(V(now), self.delta_t)), R[:,i])
            if update and i % update_time == 0:
                print(str(i*self.delta_t) + " seconds done out of " + str(self.T))
        return storage




'''
A few sample potentials.
'''
def well_well(x):
    '''
    Potential with wells at +1,-1 (and all combinations in higher dimensions). Takes
    a numpy array as well, but should not be used on matrices.
    '''
    return -x ** 3 + x

def well(x, wells = [-1,1,0]):
    return -np.sum([20(x - well)*np.exp(-20*(x - well)**2) for well in wells], axis = 0)

def OU(x):
    return -x

def zero(x):
    return 0



class VAC(object):
    """Class to do vac on a trajectory. Uses algorithm described in Klus et al, 2018."""

    def __init__(self, basis, trajectory, time_lag, delta_t, dimension = 1, update = False):
        '''
        basis: list of fcns
        N: int
        trajectory: matrix
        time_lag: float
        dimension: int
        '''
        if len(trajectory) % dimension != 0:
            raise ValueError('Dimension of trajectory not divisible by dimension desired.')
        self.basis = basis
        self.N = len(trajectory[0,:])
        self.trajectory = trajectory
        self.time_lag = time_lag
        self.lag = np.rint(time_lag / delta_t).astype(int)
        self.dimension = dimension
        self.X = np.array([b(np.hstack([trajectory[d:d+dimension, 0:(self.N - self.lag)] for d in range(0,len(trajectory), dimension)])) for b in basis])
        self.Y = np.array([b(np.hstack([trajectory[d:d+dimension, self.lag:] for d in range(0,len(trajectory), dimension)])) for b in basis])
        # self.X = np.array([b(trajectory[0:dimension,0:(self.N - self.lag)]) for  b in basis])
        # self.Y = np.array([b(trajectory[:,self.lag:]) for b in basis])
        if update:
            print("Done generating VAC Object.")

    def C_0(self):
        '''
        This is a matrix of E[f_i(X_0)f_j(X_0)] for f_i,f_j basis functions.
        '''
        return (np.matmul(self.X, self.X.T) + np.matmul(self.Y, self.Y.T)) / (2*self.N)

    def C_t(self):
        '''
        This is a matrix of E[f_i(X_(delta_t))f_j(X_0)] for f_i,f_j basis functions.
        '''
        return (np.matmul(self.X, self.Y.T) + np.matmul(self.Y, self.X.T)) / (2 * self.N)

    def find_eigen(self, m):
        C_t = self.C_t()
        C_0 = self.C_0()
        l = len(self.basis)
        eigvals, eigvecs = eigh(C_t, C_0, eigvals=(l-m,l-1), eigvals_only=False)

        return [eigvals, eigvecs]


'''
functions to deal with function norms and projection norms
'''

def makegrid(endpoint, dimension = 1, n  = 100):
    '''
    endpoint: float > 0
    '''
    points_1D = np.linspace(-endpoint,endpoint,n)
    points = itertools.product(points_1D, repeat = dimension)
    return np.array(list(points))

def dot(f, g, distribution):
    return np.sum([np.multiply(f(x), g(x)) for x in distribution]) / len(distribution.T)

def fcn_weighting(fs,weighting):
    if len(fs) != len(weighting):
        return "Error: length of fcn list and length of weightings must be the same."
    def g(x):
        return np.dot(weighting, [f(x) for f in fs])
    return g

def L2subspaceProj_d(w_f, w_g, distribution, Phi_f = False, Phi_g = False, basis_f = False, basis_g = False):
    if len(w_f) != len(w_g):
        return "Error: Subspaces have different dimensions."
    if type(Phi_f) == bool:
        if type(basis_f) == bool:
            return "Error: Must have a basis for f's supplied if no Phi_f given."
        else:
            A = np.array([f(distribution) for f in basis_f])
            # print(A)

    if type(Phi_g) == bool:
        if type(basis_f) == bool:
            return "Error: Must have a basis for g's supplied if no Phi_g given."
        else:
            B = np.array([g(distribution) for g in basis_g])
            # print(B)

    else:
        A = Phi_f
        B = Phi_g

    A = np.dot(w_f, A)
    # print(A)

    B = np.dot(w_g, B)
    # print(B)

    P = np.dot(A,B.T)
    # print(P)

    N = 1 / np.tensordot(np.sqrt(np.sum(np.square(A), axis = 1)), np.sqrt(np.sum(np.square(B), axis = 1)), axes = 0)
    # print(N)

    P = np.multiply(P, N)
    # print(P)
    svd = np.linalg.svd(P)[1]
    print(svd)

    if len(w_f) < np.sum(np.square(svd)) < len(w_f) + 1e-13:
        return 0
    elif np.sum(np.square(svd)) > len(w_f):
        return "Singular values are too large, probably not normalized correctly."
    else:
        return np.sqrt(len(w_f) - np.sum(np.square(svd)))


def f(x):
    return np.squeeze(np.array(np.sum(np.square(x+1), axis = 0)))

def h(x):
    return np.squeeze(np.array(np.sum(x+1, axis = 0)))

def g(x):
    return np.squeeze(np.sin(x))

def k(x):
    return np.squeeze(np.cos(x))

t = np.array([np.linspace(-2,2,5)])

print(L2subspaceProj_d(w_f = np.array([[1,0],[1,0]]), w_g = np.array([[0,1],[0,1]]),
                distribution = t, basis_f = np.array([g,k]), basis_g = np.array([g,k])))

t = np.ones([10,5])
#
# V = VAC([f,h], t, 1, 1, dimension = 2)
#
# print(V.X)
# print('/n')
# print(V.Y)
