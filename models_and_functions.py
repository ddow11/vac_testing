import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
from hermite_poly import Hermite
import itertools
from scipy.linalg import subspace_angles
import scipy
import math


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

    def underdamped(self, update = False):
        N = round(self.T/self.delta_t)
        now = np.random.normal(np.zeros(self.n),1)
        storage = np.zeros((self.n, N))
        delta_t = self.delta_t
        m = 2/np.sqrt(3) * np.exp(-delta_t/2) * np.array([[np.cos(np.sqrt(3)/2 * delta_t - np.pi/6), np.sin(np.sqrt(3)/2 * delta_t)], [-np.sin(np.sqrt(3)/2 * delta_t), np.cos(np.sqrt(3)/2 * delta_t + np.pi/6)]])
        # sigma2 =  np.array([[1 - 2 * np.exp(-delta_t), 0], [0, 1 - 2 * np.exp(-delta_t)]]) + 4/3 * np.exp(-delta_t) * np.array([[np.cos(np.sqrt(3)/2 * delta_t + np.pi/6), np.sin(np.sqrt(3)/2 * delta_t) ],[np.sin(np.sqrt(3)/2 * delta_t), np.cos(np.sqrt(3)/2 * delta_t - np.pi/6)]])
        sigma2 = np.array([[1 + 2/3 * np.e**(-delta_t) * (-2 + np.sin(np.pi/6 - np.sqrt(3)*delta_t)), -2/3*np.e**(-delta_t)*(-1+np.cos(np.sqrt(3)*delta_t))], [-2/3*np.e**(-delta_t)*(-1+np.cos(np.sqrt(3)*delta_t)), 1 + 2/3 * np.e**(-delta_t) * (-2 + np.sin(np.pi/6 + np.sqrt(3)*delta_t))]])
        sigma = scipy.linalg.sqrtm(sigma2)
        R = np.random.normal(0, 1, [self.n, N])
        sigma = scipy.linalg.block_diag(*([sigma]*(self.n//2)))
        update_time = round(.01*N)
        for i in range(N):
            storage[:,i] = now
            mean = np.dot(m, np.vstack([now[slice(None, None, 2)], now[slice(1, None, 2)]]))
            mean = np.dstack((mean[0,:], mean[1,:])).flatten()
            now = mean + np.dot(sigma, R[:,i])
            if update and i % update_time == 0:
                print(str(i*self.delta_t) + " seconds done out of " + str(self.T))
        return storage

    def underdamped_approx(self, update = False):
        if self.n % 2 != 0:
            return "Number of trajectories must be divisible by 2."
        N = round(self.T/self.delta_t)
        now = np.random.normal(np.zeros(self.n),1)
        storage = np.zeros((self.n, N))
        R = np.multiply(np.random.normal(np.zeros((self.n // 2, N)),1), np.sqrt(2*self.delta_t))
        if update:
            update_time = round(.01 * N)
        for i in range(0,N):
            storage[:,i] = now
            now_q = now[slice(None, None, 2)]
            now_p = now[slice(1, None, 2)]
            now[slice(None, None, 2)] = np.add(now_q, np.multiply(now_p, self.delta_t))
            now[slice(1, None, 2)] = np.add(now_p, np.add(np.multiply(np.add(now_q, now_p), -self.delta_t), R[:,i]))
            if update and i % update_time == 0:
                print(str(i*self.delta_t) + " seconds done out of " + str(self.T))
        return storage

    def underdampedApproxGamma(self, gammas, update = False):
        if self.n % 2 != 0:
            return "Number of trajectories must be divisible by 2."
        N = round(self.T/self.delta_t)
        now = np.random.normal(np.zeros(self.n),1)
        storage = np.zeros((self.n, N))
        R = np.multiply(np.random.normal(np.zeros((self.n // 2, N)),1), np.sqrt(2*self.delta_t))
        if update:
            update_time = round(.01 * N)
        for i in range(0,N):
            gamma = gammas[i]
            storage[:,i] = now
            now_q = now[slice(None, None, 2)]
            now_p = now[slice(1, None, 2)]
            now[slice(None, None, 2)] = now_q + now_p * self.delta_t * gamma
            now[slice(1, None, 2)] = now_p - (gamma * now_p + now_q) * (self.delta_t*gamma) + gamma * R[:,i]
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

    def __init__(self, basis, trajectory, time_lag, delta_t, dimension = 1, update = False, C_0 = True):
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
        self.trajectory = trajectory
        self.time_lag = time_lag
        self.lag = np.rint(time_lag / delta_t).astype(int)
        self.N = int(len(trajectory) / dimension) * (len(trajectory[0,:]) - self.lag)
        self.dimension = dimension
        self.X = np.array([b(np.hstack([trajectory[d:d+dimension, 0:(len(trajectory[0,:]) - self.lag)] for d in range(0,len(trajectory), dimension)])) for b in basis])
        self.Y = np.array([b(np.hstack([trajectory[d:d+dimension, self.lag:] for d in range(0,len(trajectory), dimension)])) for b in basis])
        # self.X = np.array([b(trajectory[0:dimension,0:(self.N - self.lag)]) for  b in basis])
        # self.Y = np.array([b(trajectory[:,self.lag:]) for b in basis])
        self.update = update
        self.C = C_0
        if self.update:
            print("Done generating VAC Object.")

    def C_0(self):
        '''
        This is a matrix of E[f_i(X_0)f_j(X_0)] for f_i,f_j basis functions.
        '''
        if self.C:
            return (np.matmul(self.X, self.X.T) + np.matmul(self.Y, self.Y.T)) / (2*(self.N  - 1))

        else:
            return np.identity(len(self.basis))
    def C_t(self):
        '''
        This is a matrix of E[f_i(X_(delta_t))f_j(X_0)] for f_i,f_j basis functions.
        '''
        return (np.matmul(self.X, self.Y.T) + np.matmul(self.Y, self.X.T)) / (2 * (self.N - 1))

    def find_eigen(self, m):
        l = len(self.basis)
        if self.update:
            print("Finding eigenvalues.")
        eigvals, eigvecs = eigh(self.C_t(), self.C_0(), eigvals=(l-m,l-1), eigvals_only=False)

        return [eigvals, eigvecs]

    def A(self):
        '''
        Returns a matrix A such that, Y = AX.
        '''

        return np.matmul(np.matmul(self.Y,self.X.T), np.linalg.inv(np.matmul(self.X, self.X.T)))

    def EDMD(self, m):
        '''
        Finds eigenvalues and eigenvectors of the matrix A.
        '''
        if self.update:
            print("Finding eigenvalues.")
        eigvals, eigvecs = np.linalg.eig(self.A().T)
        return [eigvals[:m], eigvecs[:,:m]]




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

def L2subspaceProj_d(w_f, w_g, distribution, Phi_f = False, Phi_g = False, basis_f = False, basis_g = False,
                                            normalize_f = True, normalize_g = True, dimension = 1, orthoganalize = False,
                                            tangents = False):
    '''
    w_f, w_g: these are matrices where the rows gives the weighting of the basis functions,
                i.e. dot(w_f[:, i], [f_1,...,f_d]) is the ith (estimated) eigenfunction


    '''


    if len(w_f) != len(w_g):
        return "Error: Subspaces have different dimensions."
    if type(Phi_f) == bool:
        if type(basis_f) == bool:
            return "Error: Must have a basis for f's supplied if no Phi_f given."
        else:
            A = np.array([f(distribution) for f in basis_f])
            print(A)

    if type(Phi_g) == bool:
        if type(basis_f) == bool:
            return "Error: Must have a basis for g's supplied if no Phi_g given."
        else:
            B = np.array([g(distribution) for g in basis_g])
            print(B)

    else:
        A = Phi_f # A is the matrix A_{ij} = (f_i(x_j)) where f_i is the ith
                  # basis vector and x_j is the jth data point
        B = Phi_g

    A = np.dot(w_f, A) # This transforms the matrix A into a matrix A'_{ij} = (phi_i(x_j))
                       # where phi_i is the ith (estimated) eigenfunction
    B = np.dot(w_g, B)

    if orthoganalize: # forces colulmn of A and B matrix to be orthogonal (equivalent
                      # to making eigenfunctions orthogonal)
        A = scipy.linalg.orth(A.T).T
        B = scipy.linalg.orth(B.T).T
        P = np.dot(A,B.conj().T)
    else:
        if normalize_f and normalize_g:
            N = 1 / np.tensordot(np.sqrt(np.sum(np.square(A), axis = 1)), np.sqrt(np.sum(np.square(B), axis = 1)), axes = 0)
        elif normalize_f:
            N = 1 / np.tensordot(np.sqrt(np.sum(np.square(A), axis = 1)), np.ones(len(A)) * np.sqrt(distribution.size), axes = 0)
        elif normalize_g:
            N = 1 / np.tensordot(np.ones(len(B)) * np.sqrt(distribution.size), np.sqrt(np.sum(np.square(B), axis = 1)), axes = 0)
        else:
            N = np.identity(len(w_f)) / distribution.size
        # print(N)
        P = np.multiply(np.dot(A,B.conj().T), N)

    # P is the matrix of dot products of the eigenfunctions from each eigenspace
    # i.e. P_{ij} = dot(phi_i, varphi_j) / (||phi_i|| ||varphi_j||), where
    # all of these quantities are estimated with data.
    svd = np.linalg.svd(P)[1]
    print(svd)
    if tangents:
        svd = np.sqrt(np.maximum((1 - np.square(svd)), 0)) / svd
        return np.sqrt(np.sum(np.square(svd)))
    if len(w_f) < np.sum(np.square(svd)) < len(w_f) + 1e-12:
        return 0
    elif np.sum(np.square(svd)) > len(w_f):
        return "Singular values are too large, probably not normalized correctly."
    else:
        return np.sqrt(len(w_f) - np.sum(np.square(svd)))

def d(x):
    return np.squeeze(x/x)

def f(x):
    return np.squeeze(x)

def h(x):
    return np.squeeze(x**2)

def g(x):
    return np.squeeze(x**3)

def k(x):
    return np.squeeze(x**4)

def l(x):
    return np.squeeze(x**5)

def p(x):
    return np.squeeze(np.sin(x))

def s(x):
    return np.squeeze(np.cos(x))

t = np.array(np.linspace(0,1,2))

print(L2subspaceProj_d(w_f = np.identity(3), w_g = np.identity(3),
                distribution = t, basis_f = np.array([f,f,f]), basis_g = np.array([f,f,f])))

#
# V = VAC([f,h], t, 1, 1, dimension = 2)
#
# print(V.X)
# print('/n')
# print(V.Y)

def orthonormol(vectors):
    return normalize(ortho(vectors))

def normalize(vectors):
    return np.array([v / np.linalg.norm(v) for v in vectors])

def ortho(vectors):
    v = vectors[0]
    if len(vectors) == 1:
        return vectors

    else:
        rest = ortho(vectors[1:])
        a = np.zeros(len(v))
        for u in rest:
            a += np.dot(u,v) / (np.dot(u,u)) * u
        return np.vstack([[v - a], rest])




np.sqrt(3 - np.sum(np.square([1., 0.99999986, 0.99999952])))
