import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.path as path
from hermite_poly import Hermite, Poly
from models_and_functions import simulate, VAC, well_well, makegrid, fcn_weighting, L2subspaceProj_d, OU, dot
from mpl_toolkits import mplot3d
import basis_sets
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import tables as tb
import datetime

basisNum = [10]
dimension = 1
delta_t = .01
T = 1000
n = 160
h5 = tb.open_file("Trajectory_Data/OU_1D_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r')
a = h5.root.data
h5.close()
k= 3
time_lag = np.hstack([np.linspace(delta_t, .3,6), np.linspace(.4, 5, 12)])

basis_true = [Hermite(0).to_fcn()]
basis_true = basis_true + [Hermite(n, d).to_fcn() for n in range(1, k) for d in range(dimension)]
w_f = np.identity(k)
distribution = np.random.normal(np.zeros([dimension,int(1e6)]), 1)

Phi_f = np.array([f(distribution) for f in basis_true])

sysErrors = []
minErrors = []

for fineness in basisNum:
    basis = basis_sets.makeIndicators(fineness)
    basisSize = len(basis)
    exactProj = np.zeros((k,basisSize))
    for i in range(len(basis)):
        lower = scipy.stats.norm.ppf(i / fineness)
        upper = scipy.stats.norm.ppf((i+1) / fineness)
        for j in range(k):
            exactProj[j,i] = scipy.integrate.quad(lambda x: Hermite(j).to_fcn()(x)*scipy.stats.norm.pdf(x), lower, upper)[0]


    lefts = [scipy.stats.norm.ppf(i / fineness) for i in range(fineness)]
    rights = [scipy.stats.norm.ppf((i+1)/fineness) for i in range(fineness)]

    Cts = []
    for t in time_lag:
        Ct = np.zeros((basisSize, basisSize))
        for i, (left1, right1) in enumerate(zip(lefts,rights)):
            for j, (left2, right2) in enumerate(zip(lefts[:i+1], rights[:i+1])):
                f = lambda x,y: scipy.stats.multivariate_normal.pdf((x,y), mean=[0,0], cov=[[1, np.exp(-t)], [np.exp(-t), 1]])
                Ct[i,j] = scipy.integrate.dblquad(f, left1, right1, lambda x: left2, lambda x: right2)[0]
                print("Done with:\n time lag: {},\n function1 = {},\n function2 = {}".format(t, i, j))
        Cts.append(Ct)

    C0 = np.diag([1/fineness for k in range(basisSize)])

    eigs = [scipy.linalg.eigh(Ct, C0) for Ct in Cts]

    Phi_g = np.array([f(distribution) for f in basis])

    minError = L2subspaceProj_d(w_f = w_f, w_g = exactProj,
                            distribution = distribution, Phi_f = Phi_f, Phi_g = Phi_g, normalize_f = False)

    minErrors.append(minError)

    sysError = [L2subspaceProj_d(w_f = w_f, w_g = ev[1].T[basisSize - k:][::-1],
                            distribution = distribution, Phi_f = Phi_f, Phi_g = Phi_g, normalize_f = False)
                            for ev in eigs]
    sysErrors.append(sysError)


"""
-------------------------------------------------------------------------------
SAVING ERRORS
-------------------------------------------------------------------------------
"""

np.save("Trajectory_Data/sysErrors_10-40-100-300.npy", sysErrors)
np.save("Trajectory_Data/minErrors_10-40-100-300.npy", minErrors)
np.save("Trajectory_Data/Cts_300basis.npy", Cts)

sysErrors = np.load("Trajectory_Data/sysErrors_10-40-100-300.npy")
minErrors = np.load("Trajectory_Data/minErrors_10-40-100-300.npy")
np.load("Trajectory_Data/Cts_300basis.npy")



"""
-------------------------------------------------------------------------------
PLOTS
-------------------------------------------------------------------------------
"""
sysErrors = np.load("Trajectory_Data/sysErrors_10-40-100-300.npy")
time_lag = np.hstack([np.linspace(delta_t, .3,6), np.linspace(.4, 5, 12)])

sysErrorsCut = [error for error in sysErrors]
time_lagCut = time_lag
eigenRatio = np.array([np.e**(-t*k)/np.e**(-t*(k-1)) for t in time_lagCut])

sysErrorNormalized = [error - min for error,min in zip(sysErrorsCut, minErrors)]

possibleAs = np.linspace(0,1,100)
closeness = [np.array([np.sum(np.absolute(error - (minError + a*eigenRatio))) for a in possibleAs]) for error,minError in zip(sysErrorsCut, minErrors)]

estimatedErrors = [error + possibleAs[np.argmin(a)]*eigenRatio for error, a in zip(minErrors, closeness)]
numPlots = len(basisNum)

fig, ax = plt.subplots(nrows = 1, ncols = numPlots, figsize = (15,5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

[ax[i].semilogy(time_lagCut, error, label = "{} basis functions".format(basisNum[i])) for i,error in enumerate(sysErrorNormalized)]
# [ax[i].semilogy(time_lag, error, label = "Estimated Error -- a = {}, b = {}".format(round(minErrors[i],2), round(possibleAs[np.argmin(closeness[i])], 4))) for i, error in enumerate(estimatedErrors)]

# [ax[j].legend(fontsize = "xx-small", loc = "middle left") for j in range(numPlots)]
# [ax[j].set_xlim(xmin = 3) for j in range(numPlots)]
# [ax[j].set_ylim(ymin = 0) for j in range(numPlots)]
[ax[j].set_xlabel("Time Lag") for j in range(numPlots)]

fig.suptitle("Theoretical Systematic Error vs. Error")
plt.annotate("Estimating three eigenfunctions.", (0,0), (0, -32), fontsize = 7, xycoords='axes fraction', textcoords='offset points', va='top')

plt.show()

# plt.savefig("Graphs/sysError_estimateLargeBasis.png")
