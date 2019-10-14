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
from basis_sets import indicator
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import tables as tb

dimension = 1
fineness  = 10
endpoint = 2.5
basis = [Hermite(0).to_fcn()]
basis = basis + [Hermite(n, d).to_fcn() for n in range(1, fineness) for d in range(dimension)]
basisSize = len(basis)
delta_t = .01
T = 1000
n = 160
h5 = tb.open_file("Trajectory_Data/OU_1D_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r')
a = h5.root.data
h5.close()
m = 3
l = basisSize
time_lag = np.hstack([np.linspace(delta_t, .3,5), np.linspace(.4, 4, 10)])
C_true = [np.diag([np.e**(-i*t) for i in range(basisSize)]) for t in time_lag]
starts = [0]

basis_true = [Hermite(0).to_fcn()]
basis_true = basis_true + [Hermite(n, d).to_fcn() for n in range(1, m) for d in range(dimension)]
w_f = np.identity(m)

l = basisSize
Cts = []
C0s = []
evs = []
evss = []
eigen_dist = []
proj_error = []
gap_error = []
tan_error = []
for i in starts:
    print("Now opening trajectory data.")
    h5 = tb.open_file("Trajectory_Data/OU_1D_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r')
    a = h5.root.data
    t = np.array(a[i:i+100,:])
    h5.close()
    vac = [VAC(basis, t, l, delta_t, dimension = dimension, update = True) for l in time_lag]
    print("Now getting C_ts")
    C_ts = [v.C_t() for v in vac]
    Cts.append(C_ts)
    print("Now gettings C_0s")
    C_0s = [v.C_0() for v in vac]
    C0s.append(C_0s)
    print("Done with {}th start".format(i))
    evs = [eigh(C_t,C_0, eigvals_only=False) for C_t,C_0 in zip(C_ts,C_0s)]
    evss.append(evs)
    eigen_dist.append([ev[0][-m] - ev[0][-m-1] for ev in evs])
    V = [ev[1].T[-m:][::-1].T for ev in evs]
    Q = [np.linalg.qr(v)[0] for v in V]
    svd = [np.linalg.svd(q[0:3,0:3])[1] for q in Q]
    error_proj = [np.sqrt(m - np.sum(np.square(v))) for v in svd]
    error_gap = [np.sqrt(1 - np.square(v[-1])) for v in svd]
    error_tan = [np.sqrt(np.sum((1 - np.square(v)) / np.square(v))) for v in svd]
    proj_error.append(error_proj)
    gap_error.append(error_gap)
    tan_error.append(error_tan)

Es = [[np.dot(np.linalg.inv(C_0), C_hat_t) - C_t for C_hat_t,C_t,C_0 in zip(C_ts,C_true,C_0s)] for C_ts,C_0s in zip(Cts,C0s)]



'''
----------------------------------------
PLOTTING ESTIMATED ERROR VS ACTUAL ERROR
----------------------------------------
'''
upperBound = np.sqrt(m)
actualError = np.minimum(proj_error, upperBound)

spectralGaps =  [np.array([[1/(np.e**(- t * j) - np.e**(-t * i)) for j in range(basisSize)] for i in range(basisSize)]) for t in time_lag]
error_sampling_fro = [[np.linalg.norm(E[t][m:, 1:m], 'fro') for t in range(len(time_lag))] for E in Es]
EsAdjusted = [[np.multiply(E[t], spectralGaps[t]) for t in range(len(time_lag))] for E in Es]
estimatedError_firstOrder = np.minimum([[np.linalg.norm(E[m:,1:m], 'fro') for E in Es] for Es in EsAdjusted], upperBound)
eigen_dist_exact = np.array([np.e**(-(m-1)*t) - np.e**(-m*t) for t in time_lag])


fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (12,5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# [ax.plot(time_lag, error, color = "green", alpha = .4) for error in error_precise_fro]
# ax.plot(time_lag, 1/np.array(eigen_dist_exact), color = "red", linewidth = 2, label = "Log Eigen Distance")

"""
First plot:
"""

estimated_error = [np.array(error) / eigen_dist_exact for error in error_sampling_fro]
estimated_error = np.array([np.minimum(error, upperBound) for error in estimated_error])
estimatedErrors = np.array([np.minimum(error, upperBound) for error in estimated_error])
[ax[0].plot(time_lag, error, color = "green", alpha = .2) for error in estimatedErrors]
[ax[0].plot(time_lag, error, color = "blue", alpha = .2) for error in actualError]

avgEstimated = np.sum(estimatedErrors, axis = 0) / len(estimatedErrors)
# avgEstimated = estimatedErrors
avgError = np.sum(actualError, axis  = 0) / len(actualError)

ax[0].plot(time_lag, avgEstimated, color = "green", linewidth = 3, label = "1st conjecture")
ax[0].plot(time_lag, avgError, color = "blue", linewidth = 3, label = "Actual")
ax[0].legend(loc = "upper left", fontsize = "x-small")
ax[0].set_xlim(xmin = 0)
ax[0].set_xlabel("Time Lag")

"""
Second plot:
"""
[ax[1].plot(time_lag, error, color = "green", alpha = .2) for error in estimatedError_firstOrder]
[ax[1].plot(time_lag, error, color = "blue", alpha = .2) for error in actualError]

avgEstimated = np.sum(estimatedError_firstOrder, axis = 0) / len(estimatedErrors)
# avgEstimated = estimatedErrors
avgError = np.sum(actualError, axis  = 0) / len(actualError)

ax[1].plot(time_lag, avgEstimated, color = "green", linewidth = 3, label = "First Order Approx")
ax[1].plot(time_lag, avgError, color = "blue", linewidth = 3, label = "Actual")
ax[1].legend(loc = "upper left", fontsize = "x-small")
ax[1].set_xlim(xmin = 0)
ax[1].set_xlabel("Time Lag")

"""
Third plot:
"""
estimatedErrors = np.array([np.linalg.norm(Gaps[m:,0:m], 'fro') for Gaps in spectralGaps])

constantMultiple = np.average([np.absolute([E[i][m:,0:m] / time_lag[i] for i in range(len(time_lag))]) for E in Es[0:1]])
simpleError = constantMultiple*np.array(estimatedErrors)
simpleError = np.minimum(simpleError, np.sqrt(m))
[ax[2].plot(time_lag, error, color = "blue", alpha = .2) for error in actualError]

avgError = np.sum(actualError, axis  = 0) / len(actualError)

ax[2].plot(time_lag, simpleError, color = "green", linewidth = 3, label = "Simplified 1st order, 1")
ax[2].plot(time_lag, avgError, color = "blue", linewidth = 3, label = "Actual")
ax[2].legend(loc = "upper left", fontsize = "x-small")
ax[2].set_xlim(xmin = 0)
ax[2].set_xlabel("Time Lag")

fig.suptitle("Estimated Error vs. Error")

"""
Fourth plot:
"""

estimatedErrors = [[np.linalg.norm(E[t][m:,1:m], 'fro')*np.array(estimatedErrors[t])*((m-1)*(basisSize - m))**(-1/2) for t in range(len(time_lag))] for E in Es]
estimatedErrors = np.minimum(estimatedErrors, upperBound)

[ax[3].plot(time_lag, error, color = "green", alpha = .2) for error in estimatedErrors]
[ax[3].plot(time_lag, error, color = "blue", alpha = .2) for error in actualError]

avgEstimated = np.sum(estimatedErrors, axis = 0) / len(estimatedErrors)
avgError = np.sum(actualError, axis  = 0) / len(actualError)

ax[3].plot(time_lag, avgEstimated, color = "green", linewidth = 3, label = "Simplified 1st order, 2")
ax[3].plot(time_lag, avgError, color = "blue", linewidth = 3, label = "Actual")
ax[3].legend(loc = "upper left", fontsize = "x-small")
ax[3].set_xlim(xmin = 0)
ax[3].set_xlabel("Time Lag")

fig.suptitle("Estimated Error vs. Tangent Error")


plt.savefig("Graphs/EstimateErrorvsTanError.png")

"""
--------------------
Plotting error matrices
--------------------
"""
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,5))

estimatedErrors1 = [[np.linalg.norm(E[t][m:,0:m], 'fro')*(m*(basisSize - m))**(-1/2) for t in range(len(time_lag))] for E in Es]
estimatedErrors2 = [[np.linalg.norm(E[t], 'fro') / basisSize for t in range(len(time_lag))] for E in Es]

[ax[0].plot(time_lag, error, color = "blue", alpha = .2) for error in estimatedErrors1]
[ax[0].plot(time_lag, error, color = "green", alpha = .2) for error in estimatedErrors2]

avgEstimated1 = np.sum(estimatedErrors1, axis = 0) / len(estimatedErrors1)
avgEstimated2 = np.sum(estimatedErrors2, axis = 0) / len(estimatedErrors2)

ax[0].plot(time_lag, avgEstimated1, color = "blue", linewidth = 3, label = "Bottom left submatrix")
ax[0].plot(time_lag, avgEstimated2, color = "green", linewidth = 3, label = "Entire matrix")
ax[0].legend(loc = "upper right", fontsize = "x-small")
ax[0].set_xlim(xmin = 0)
ax[0].set_ylim(ymin = 0)
ax[0].set_xlabel("Time Lag")
fig.suptitle("Norm of Error Matrix vs. Time Lag")
plt.annotate("Matrix norms are Frobenius, normalized by the square root of number of entries.", (0,0), (0, -32), fontsize = 7, xycoords='axes fraction', textcoords='offset points', va='top')


[ax[1].plot(time_lag, error, color = "blue", alpha = .2) for error in estimatedErrors1]
ax[1].plot(time_lag, avgEstimated1, color = "blue", linewidth = 3, label = "Bottom left submatrix")
ax[1].legend(loc = "upper right", fontsize = "x-small")
ax[1].set_xlim(xmin = 0)
ax[1].set_ylim(ymin = 0)
ax[1].set_xlabel("Time Lag")
fig.suptitle("Norm of Error Matrix vs. Time Lag")
plt.annotate("Matrix norms are Frobenius, normalized by the square root of number of entries.", (0,0), (0, -32), fontsize = 7, xycoords='axes fraction', textcoords='offset points', va='top')

plt.savefig("Graphs/ErrorMatrix.png")

"""
-------------------------------------------------------------------------------
Estimating Error Using Data
-------------------------------------------------------------------------------
"""

L  = [[np.sqrt(np.sum(np.square(evs[t][0][:m]))) for t in range(len(time_lag))] for evs in evss]

S = [[np.array([[1 / (evs[t][0][i] - evs[t][0][j]) for j in range(basisSize)] for i in range(basisSize)]) for t in range(len(time_lag))] for evs in evss]

errorEstimated = [[np.linalg.norm(A[m:, 1:m], 'fro') * l / (basisSize - m) for A,l in zip(As,ls)] for As, ls in zip(S, L)]

fig, ax = plt.subplots(nrows = 1, ncols = 1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# [ax.plot(time_lag, error, color = "green", alpha = .4) for error in error_precise_fro]
# ax.plot(time_lag, 1/np.array(eigen_dist_exact), color = "red", linewidth = 2, label = "Log Eigen Distance")

"""
Plot:
"""

estimated_error = np.array([np.minimum(error, upperBound) for error in errorEstimated])
[ax.plot(time_lag, error, color = "green", alpha = .2) for error in estimated_error]
[ax.plot(time_lag, error, color = "blue", alpha = .2) for error in actualError]

avgEstimated = np.sum(estimated_error, axis = 0) / len(estimated_error)
# avgEstimated = estimatedErrors
avgError = np.sum(actualError, axis  = 0) / len(actualError)

ax.plot(time_lag, avgEstimated, color = "green", linewidth = 3, label = "Approx w/ Data")
ax.plot(time_lag, avgError, color = "blue", linewidth = 3, label = "Actual")
ax.legend(loc = "upper left", fontsize = "x-small")
ax.set_xlim(xmin = 0)
ax.set_xlabel("Time Lag")

'''
-------------------------------------------------------------------------------
Spectral Gap vs. Complicated Spectral Gap
-------------------------------------------------------------------------------
'''

complicatedGap = [np.linalg.norm(Gaps[m:,:m], 'fro')*(m*(basisSize - m))**(-1/2) for Gaps in spectralGaps]
simpleGap = 1 / eigen_dist_exact

fig, ax = plt.subplots(nrows = 1, ncols = 1)

ax.plot(time_lag, complicatedGap, color = "forestgreen", label = "Complicated Gap, inverse")
ax.plot(time_lag, simpleGap, color = "blue",label = "Simple Gap, inverse")
ax.legend()
ax.set_xlim(xmin = 0)
ax.set_ylim(ymin = 0)
fig.suptitle("Simple Spectral Gap vs. Complicated Spectral Gap")

plt.savefig("Graphs/SimplevsComplicatedGap")
