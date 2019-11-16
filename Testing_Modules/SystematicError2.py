import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.path as path
from hermite_poly import Hermite, Poly
from models_and_functions import simulate, VAC, well_well, makegrid, fcn_weighting, L2subspaceProj_d, OU, dot, normalize, ortho
from mpl_toolkits import mplot3d
import basis_sets
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import tables as tb
import datetime
import math
import mpmath as mp

basisNum = [5,15]
dimension = 1
delta_t = .01
T = 1000
n = 160
h5 = tb.open_file("Trajectory_Data/OU_1D_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r') # retrieves the trajectory data
a = h5.root.data
h5.close()
k = 3 # dimension of eigenspace to estimate
time_lag = np.hstack([np.linspace(.0001, .1, 4), np.linspace(.11, 1, 12), np.linspace(1, 18, 15)]) # time lags

basis_true = [Hermite(0).to_fcn()]
basis_true = basis_true + [Hermite(n, d).to_fcn() for n in range(1, k) for d in range(dimension)]
w_f = np.identity(k)

distribution = np.random.normal(np.zeros([dimension,int(1e6)]), 1) # distribution used to calculate projection error

Phi_f = np.array([f(distribution) for f in basis_true])

sysErrors = []
minErrors = []

Ctss = []
exactProjs = []
orthoProjs = []
eigss = []

for fineness in basisNum:
    # first, compute exact projections onto basis functions
    basis = basis_sets.makeIndicators(fineness)
    basisSize = len(basis)
    exactProj = np.zeros((basisSize,basisSize))
    for j in range(basisSize):
        for i in range(basisSize):
            lower = scipy.stats.norm.ppf(i / fineness)
            upper = scipy.stats.norm.ppf((i+1) / fineness)
            v = scipy.integrate.quad(lambda x: Hermite(j).to_fcn()(x)*scipy.stats.norm.pdf(x), lower, upper, maxp1 = 500)[0]
            print("Done with:\n Hermite Poly: {},\n basis function = {}".format(j, i))
            exactProj[j,i] = v * fineness
    exactProjs.append(normalize(exactProj))

    orthoProj = np.array(normalize(ortho(exactProj[::-1]))[::-1]) / math.sqrt(fineness)
    orthoProjs.append(orthoProj)

    # then, compute exact C(t) and C(0)
    lefts = [scipy.stats.norm.ppf(i / fineness) for i in range(fineness)]
    rights = [scipy.stats.norm.ppf((i+1)/fineness) for i in range(fineness)]

    Cts = []
    for t in time_lag:
        Ct = np.zeros((basisSize, basisSize))
        for i, (left1, right1) in enumerate(zip(lefts,rights)):
            for j, (left2, right2) in enumerate(zip(lefts[:i+1], rights[:i+1])):
                Px0inI = scipy.stats.norm.cdf(right1) - scipy.stats.norm.cdf(left1)
                f = lambda x: scipy.stats.norm.pdf(x) * (scipy.stats.norm.cdf((right2 - np.exp(-t)*x) / math.sqrt(1 - np.exp(-2*t))) -
                                                        scipy.stats.norm.cdf((left2 - np.exp(-t)*x) / math.sqrt(1 - np.exp(-2*t))))
                Ct[i,j] = Px0inI*scipy.integrate.quad(f, left1, right1)[0]*basisSize # inner product with I_j and I_i w.r.t. normal distribution
                print("Done with:\n time lag: {},\n function1 = {},\n function2 = {}".format(t, i, j))
        Cts.append(Ct +  Ct.T - np.diag(np.diag(Ct)))

    Ctss.append(Cts)
    C0 = np.diag([1 for k in range(basisSize)])

    eigs = [scipy.linalg.eigh(Ct, C0) for Ct in Cts]

    eigss.append(eigs)

    # now compute distances

    Phi_g = np.array([f(distribution) for f in basis])

    # w_f, w_g are the weigthing of the basis functions for
    # the basis set fs and gs.
    # Here, w_f corresponds to the weighting of the Hermite polys (identity bc they are the actual eigenfcsn)
    # w_g corresponds to the weighting of the indicator fcns (obtained from the eigenvectors)

    minError = L2subspaceProj_d(w_f = w_f, w_g = exactProj[:k,:],
                            distribution = distribution, Phi_f = Phi_f, Phi_g = Phi_g, normalize_f = False)

    minErrors.append(minError)

    sysError = [L2subspaceProj_d(w_f = orthoProj[:k,:], w_g = ev[1].T[basisSize - k:][::-1],
                            distribution = distribution, Phi_f = Phi_g, Phi_g = Phi_g, orthoganalize = True, tangents = True)
                            for ev in eigs]
    sysErrors.append(sysError)


"""
-------------------------------------------------------------------------------
SYS ERROR ESTIMATE
-------------------------------------------------------------------------------
"""

"""
-----------------
Estimators
-----------------
"""
def orthogonalizeMatrix(C, P):
    return np.dot(np.dot(np.linalg.inv(P.T),C),P.T)

def firstOrder(C, S):
    return np.linalg.norm(np.multiply(C[k:,:k], S[k:,:k]), 'fro')

def simplification1(C, S):
    return np.linalg.norm(C[k:,:k], 'fro') / math.sqrt(k) * np.linalg.norm(S[0,:k])

def dataEstimate(S):
    return np.linalg.norm(1 / S[0,k:]) * np.linalg.norm(S[0,:k])

def newEstimator(C, S):
    return np.linalg.norm(C[k:,k-1]) * np.linalg.norm(S[k-1,k:])

def minimalist(C, S):
    return np.linalg.norm(C[k:,k-1] * S[k:,k-1])

def simple(C, S):
    n = len(C[:,k-1])
    return np.linalg.norm(C[k:,k-1]) * np.linalg.norm(S[k:,k-1]) / math.sqrt((n-k))

def known(S, S_):
    return np.linalg.norm(S[k:,k-1]) * (S_[0,k])

def unknown(C, S):
    return np.linalg.norm(C[k:,k-1]) / ((n-k) * (S[0,k]))

"""
Helper Functions to map a function over lists
"""
def map1(f, list1):
    return [f(C) for C in list1]

def mapDoubleList(f, listOfLists):
    return [[f(C) for C in Cs] for Cs in listOfLists]

def mapMatrix(f, list1, list2):
    return [[f(C,S) for C, S in zip(Cs, Ss)] for Cs, Ss in zip(list1, list2)]

def curtailLists(lists, k):
    return map1(lambda l: l[:k], lists)

"""
---------------------------
List constructions
---------------------------
"""
# Sss2 = [[map2((lambda j, i: 1 / np.exp(-j*t)), list(range(basisSize)), list(range(basisSize)) for t in time_lag] for basisSize, eigs in zip(basisNum, eigss)]

Sss = [[np.array([[ev[0][-j] for j in range(1,basisSize+1)] for i in range(basisSize)]) for ev in eigs] for basisSize, eigs in zip(basisNum, eigss)]

Sss_diff = [[np.array([[1 / (ev[0][-j] - ev[0][-i]) for j in range(1,basisSize+1)] for i in range(1,basisSize+1)]) for ev in eigs] for basisSize, eigs in zip(basisNum, eigss)]

CtssOrth = [[orthogonalizeMatrix(Ct, P) for Ct in Cts] for P, Cts in zip(orthoProjs, Ctss)]

sysEstimates1 = mapMatrix(firstOrder, CtssOrth, Sss_diff)

sysEstimates2 = mapMatrix(simplification1, CtssOrth, Sss_diff)

sysEstimates3 = mapDoubleList(dataEstimate, Sss)

sysEstimates4 = mapMatrix(newEstimator, CtssOrth, Sss_diff)

E_minimalist = mapMatrix(minimalist, CtssOrth, Sss_diff)

E_simple = mapMatrix(simple, CtssOrth, Sss_diff)

E_known = mapMatrix(known, Sss_diff, Sss)

E_unknown = mapMatrix(unknown, CtssOrth, Sss)

eigenRatio = np.square(np.array([np.e**(-t*k)/np.e**(-t*(k-1)) for t in time_lag]))

empiricaleigenRatio = np.square([[ev[0][-(k+1)] / ev[0][-k] for ev in eigs] for eigs in eigss])

empiricaleigenRatioOptimized = np.square([[ev[0][-(k+1)] / (ev[0][-k] - ev[0][-(k+1)])for ev in eigs] for eigs in eigss])



"""
-------------------------------------------------------------------------------
SAVING ERRORS
-------------------------------------------------------------------------------
"""

# np.save("Trajectory_Data/sysErrors_10-40-100-300.npy", sysErrors)
# np.save("Trajectory_Data/minErrors_10-40-100-300.npy", minErrors)
# np.save("Trajectory_Data/Cts_300basis.npy", Cts)
#
# sysErrors = np.load("Trajectory_Data/sysErrors_10-40-100-300.npy")
# minErrors = np.load("Trajectory_Data/minErrors_10-40-100-300.npy")
# np.load("Trajectory_Data/Cts_300basis.npy")


"""
-------------------------------------------------------------------------------
PLOTS
-------------------------------------------------------------------------------
"""
def plotCustom(xaxis, listOfLists, label):
    for i,error in enumerate(listOfLists):
        ax[i].semilogy(xaxis, error, label = label(i))
        ax[i].legend(fontsize = "xx-small", loc = "upper right")
        ax[i].set_xlim(xmin = 0)
        ax[i].set_xlabel("Time Lag")


# sysErrors = np.load("Trajectory_Data/sysErrors_10-40-100-300.npy")
# sysErrorNormalized = [error - min for error,min in zip(sysError, minErrors)]

possibleAs = np.linspace(0,time_lag[-1],100)
# closeness = [np.array([np.sum(np.absolute(error - a*eigenRatio))) for a in possibleAs]) for error,minError in zip(sysErrors, minErrors)]

# estimatedErrors = [1*eigenRatio for error, a in closeness]
numPlots = len(basisNum)

fig, ax = plt.subplots(nrows = 1, ncols = numPlots, figsize = (15,5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

whatDoYouWantToPlot = [
                        (sysErrors,     lambda i: "{} basis functions".format(basisNum[i])),
                        (sysEstimates1, lambda i: "First Order Estimate (improved)"),
                        (E_minimalist,  lambda i: "E minimalist"),
                        (E_simple,      lambda i: "E simple")
                        # (empiricaleigenRatio, lambda i: "VAC Eigenratio")
                        # (E_known,         lambda i: "E known"),
                        # (E_unknown,       lambda i: "E unknown")
                        ]

for data, label in whatDoYouWantToPlot:
    plotCustom(time_lag[:15], curtailLists(data,15), label)

fig.suptitle("Theoretical Systematic Error vs. Error")
plt.annotate("Estimating three eigenfunctions.", (0,0), (0, -32), fontsize = 7, xycoords='axes fraction', textcoords='offset points', va='top')

plt.show()

# plt.savefig("Good Graphs/sysError_firstOrder_Estimators.png")


"""
-------------------------------------------------------------------------------
Gram–Schmidt Orthoganalization
-------------------------------------------------------------------------------
"""

distribution = np.random.normal(0,1, 10000)

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
