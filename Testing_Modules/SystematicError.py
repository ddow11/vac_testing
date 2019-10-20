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

fineness  = 20
dimension = 1
basis = basis_sets.makeIndicators(fineness)
basisSize = len(basis)
delta_t = .01
T = 1000
n = 160
h5 = tb.open_file("Trajectory_Data/OU_1D_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r')
a = h5.root.data
h5.close()
k= 3
l = basisSize
time_lag = np.hstack([np.linspace(delta_t, .3,7), np.linspace(.4, 3, 8)])
C_true = [np.diag([np.e**(-i*t) for i in range(basisSize)]) for t in time_lag]
starts = [0,0,0]
sizes = [1,5,20]


l = basisSize
evs = []
evss = []
eigen_dist = []
sysError = []
exactProj = np.zeros((k,basisSize))
projError = []
gap_error = []
tan_error = []

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
            f = lambda x,y: scipy.stats.multivariate_normal.pdf((x,y), mean=[0,0], cov=[[1, np.e**(-t)], [np.e**(-t), 1]])
            Ct[i,j] = scipy.integrate.dblquad(f, left1, right1, lambda x: left2, lambda x: right2)[0]
            print("Done with:\n time lag: {},\n function1 = {},\n function2 = {}".format(t, i, j))
    Cts.append(Ct)

C0 = np.diag([1/fineness for k in range(basisSize)])

eigs = [scipy.linalg.eigh(Ct, C0) for Ct in Cts]

basis_true = [Hermite(0).to_fcn()]
basis_true = basis_true + [Hermite(n, d).to_fcn() for n in range(1, k) for d in range(dimension)]
w_f = np.identity(k)
distribution = np.random.normal(np.zeros([dimension,int(1e6)]), 1)

Phi_g = np.array([f(distribution) for f in basis])
Phi_f = np.array([f(distribution) for f in basis_true])

sysError = [L2subspaceProj_d(w_f = w_f, w_g = ev[1].T[basisSize - k:][::-1],
                        distribution = distribution, Phi_f = Phi_f, Phi_g = Phi_g, normalize_f = False)
                        for ev in eigs]

sysError2 = L2subspaceProj_d(w_f = w_f, w_g = exactProj,
                        distribution = distribution, Phi_f = Phi_f, Phi_g = Phi_g, normalize_f = False)


for i in sizes:
    print("Now opening trajectory data.")
    h5 = tb.open_file("Trajectory_Data/OU_1D_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r')
    a = h5.root.data
    t = np.array(a[0:i,:])
    h5.close()
    evs = [VAC(basis, t, l, delta_t, dimension = dimension, update = True).find_eigen(basisSize) for l in time_lag]
    evss.append(evs)
    eigen_dist.append([ev[0][-k] - ev[0][-k-1] for ev in evs])
    error = [L2subspaceProj_d(w_f = w_f, w_g = ev[1].T[basisSize - k:][::-1],
                            distribution = distribution, Phi_f = Phi_f, Phi_g = Phi_g, normalize_f = False)
                            for ev in evs]
    projError.append(error)
    # Ps = [np.dot(exactProj, ev[1].T[basisSize - m:][::-1].T) for ev in evs]
    # svds = [np.linalg.svd(P)[1] for P in Ps]
    # errors = [np.sqrt(len(w_f) - np.sum(np.square(svd))) for svd in svds]
    # sysError.append(errors)
    print("Done with {}th start".format(i))
    print(datetime.datetime.now())





"""
-------------------------------------------------------------------------------
PLOTS
-------------------------------------------------------------------------------
"""

eigenDist = [np.e**(-t*k) - np.e**(-t*(k+1)) for t in time_lag]

actualError = projError

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12,5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

[ax[i].plot(eigenDist, projError[i], label = "{} Thousand Seconds of Data".format(sizes[i])) for i in range(len(projError))]
[ax[i].plot(eigenDist, sysError, label = "Systematic Error") for i in range(len(projError))]

for j in range(len(projError)):
    error = projError[j]
    for i in range(0, len(time_lag), 3):
        ax[j].scatter(eigenDist[i], error[i], s = 18, color = "black")
        ax[j].annotate("{}".format(round(time_lag[i],3)), xy = (eigenDist[i] - .001, error[i] + .015), weight='bold', fontsize = 6.5)
[ax[j].plot(eigenDist, [sysError2]*len(time_lag), label = "Error Lower Bound") for j in range(len(projError))]
[ax[i].legend(fontsize = "x-small") for i in range(len(projError))]
[ax[j].set_xlim(xmin = 0) for j in range(len(projError))]
[ax[j].set_ylim(ymin = 0) for j in range(len(projError))]
[ax[j].set_xlabel("Eigen Gap") for j in range(len(projError))]

fig.suptitle("Theoretical Systematic Error vs. Error")
plt.annotate("Basis of 20 indicator functions. Estimating three eigenfunctions.", (0,0), (0, -32), fontsize = 7, xycoords='axes fraction', textcoords='offset points', va='top')

plt.show()

plt.savefig("Graphs/sysError_20.png")
