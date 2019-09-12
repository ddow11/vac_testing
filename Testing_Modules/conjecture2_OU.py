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
basis = [indicator(fineness, endpoint, center = i).to_fcn() for i in  makegrid(endpoint, dimension = dimension, n = fineness)]
basis = [Hermite(0).to_fcn()]
basis = basis + [Hermite(n, d).to_fcn() for n in range(1, fineness) for d in range(dimension)]
basisSize = len(basis)
delta_t = .001
T = 1000
n = 100
length = round(T / delta_t)
print("Now opening trajectory data.")
h5 = tb.open_file("Trajectory_Data/OU_1D_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r')
a = h5.root.data
t = np.array(a[40:42,:])
h5.close()

time_lag = np.hstack([np.linspace(delta_t, 3*delta_t, 2), np.linspace(4*delta_t, .5, 8), np.linspace(.6, 3, 9)])
print("Now getting eigenvalues.")
vac = [VAC(basis, t, l, delta_t, dimension = dimension, update = True) for l in time_lag]
evs = [v.find_eigen(basisSize) for v in vac]

# np.save("Trajectory_Data/egalue_hermite_shortlag_shortlag.npy", np.array([ev[0] for ev in evs]))
# np.save("Trajectory_Data/egvector_hermite_shortlag_shortlag.npy", np.array([ev[1] for ev in evs]))

print("Now calculating error.")

distribution = np.random.normal(np.zeros([dimension,int(1e7)]), 1)
"Number of eigenfunctions to compare. Must be less than basisSize."
m = 3

basis_true = [Hermite(0).to_fcn()]
basis_true = basis_true + [Hermite(n, d).to_fcn() for n in range(1, m) for d in range(dimension)]
w_f = np.identity(m)

Phi_g = np.array([f(distribution) for f in basis])
Phi_f = np.array([f(distribution) for f in basis_true])


eigen_dist = [ev[0][basisSize - m] - ev[0][basisSize - m - 1] for ev in evs]

# evs = np.load("Trajectory_Data/egvector_hermite_shortlag_shortlag.npy", allow_pickle = True)

error = [L2subspaceProj_d(w_f = w_f, w_g = ev[1].T[basisSize - m:][::-1],
                        distribution = distribution, Phi_f = Phi_f, Phi_g = Phi_g, normalize_f = False)
                        for ev in evs]


'''
-------------------------------------------------------------------------------
What does error vs. time lag look like?
-------------------------------------------------------------------------------
'''
f,ax = plt.subplots(1)
ax.plot(eigen_dist, error)
plt.xlabel("Eigenvalue distance")
plt.ylabel("Projection Error in estimated subspaces")
plt.title("Error in estimation with varying time lags (OU process)")
plt.annotate("Using a basis of 10 Hermite polynomials and 4000 seconds of data", (0,0), (0, -32), fontsize = 8, xycoords='axes fraction', textcoords='offset points', va='top')
ax.set_xlim(xmin = 0)
plt.savefig("Graphs/eigenfcnError_OU_HermiteBasis.png")

'''
-------------------------------------------------------------------------------
Eigenvalue decay
-------------------------------------------------------------------------------
'''

ev = np.array([ev[0] for ev in evs]).T[::-1]
[plt.plot(time_lag, ev[i]) for i in range(m+1)]

plt.legend()
plt.xlabel("Time Lag")
plt.ylabel("Eigenvalues")
plt.title("Eigenvalues vs. Time Lag (OU, 1-D)")

'''
-------------------------------------------------------------------------------
Are estimated eigenfunctions orthogonal?
-------------------------------------------------------------------------------
'''

ev = evs[len(evs) // 3]
a = np.hstack(t)
A = np.array([f(a) for f in basis])
fDotg = np.dot(np.dot(ev[1].T[basisSize - m:][::-1], A), np.dot(ev[1].T[basisSize - m:][::-1], A).T) / a.size

'''
-------------------------------------------------------------------------------
Computing exact error in eigenfunctions using QR decomp.
-------------------------------------------------------------------------------
'''
'''
If V = (v_1 v_2 v_3) is the matrix of the first three eigenvectors from VAC, then
'''

V = [ev[1].T[basisSize - m:][::-1].T for ev in evs]
Q = [np.linalg.qr(v)[0] for v in V]
error_exact = [np.sqrt(m - np.sum(np.square(np.linalg.svd(q[0:3,0:3])[1]))) for q in Q]

error_normalized_g = error
error = [L2subspaceProj_d(w_f = w_f, w_g = ev[1].T[basisSize - m:][::-1],
                        distribution = distribution, Phi_f = Phi_f, Phi_g = Phi_g, normalize_f = False, normalize_g = False)
                        for ev in evs]

f,ax = plt.subplots(1)
ax.plot(error_exact, error, color = "red", label = "Before normalizing estimated eigenfcns.")
ax.plot(error_exact, error_normalized_g, color = "green", label = "After normalizing estimated eigenfcns.")
ax.plot(error_exact, error_exact, color = "black", label = "line y = x")
plt.legend()
plt.xlabel("Exact Error using QR factorization")
plt.ylabel("Estimated error")
ax.annotate("Error estimated with $10^7$ Normally distributed points.", (0,0), (0, -32), fontsize = 8, xycoords='axes fraction', textcoords='offset points', va='top')
plt.title("Verifying estimated error is close to exact error at various time lags.")
plt.savefig("Graphs/EstimatedvsExactError.png")


'''
-------------------------------------------------------------------------------
Computing first order error in C(t).
-------------------------------------------------------------------------------
'''
evs_GEP = evs
error_GEP = error

for v in vac:
    v.C = False

evs_I = [v.find_eigen(basisSize) for v in vac]

error_I = [L2subspaceProj_d(w_f = w_f, w_g = ev[1].T[basisSize - m:][::-1],
                        distribution = distribution, Phi_f = Phi_f, Phi_g = Phi_g, normalize_f = False)
                        for ev in evs_I]

plt.rc('text', usetex=False)
plt.rc('font', family='serif')

f, ax = plt.subplots(1)
ax.plot(eigen_dist, error_GEP, color = "grey", label = "Generalized Eigen Problem")
ax.plot(eigen_dist, error_I, color = "royalblue", label = "Symmetric Eigen Problem")
ax.set_xlim(xmin = 0)
ax.set_ylim(ymin = 0)
for i in range(0, len(time_lag), 4):
    ax.scatter(eigen_dist[i], error_GEP[i], s = 15, color = "grey")
    ax.annotate("{}".format(round(time_lag[i],3)), xy = (eigen_dist[i] - .001, error_GEP[i] + .02), weight='bold', fontsize = 6.5)
for i in range(0, len(time_lag), 4):
    ax.scatter(eigen_dist[i], error_I[i], s = 15, color = "royalblue")
    ax.annotate("{}".format(round(time_lag[i],3)), xy = (eigen_dist[i] - .001, error_I[i] + .015), weight='bold', fontsize = 6.5)

plt.legend()
plt.xlabel("Eigen distance")
plt.ylabel("Projection error in estimated eigenfunctions")
plt.annotate("Basis of 10 Hermite polynomials, 10,000 seconds of data. Lag times annotated.", (0,0), (0, -32), fontsize = 7, xycoords='axes fraction', textcoords='offset points', va='top')
plt.title("Comparing genealized and symmetric eigen problems")
plt.savefig("Graphs/GEPvsSEP.png")


C_ts = [v.C_t() for v in vac]
for v in vac:
    v.C = True
C_0s = [v.C_0() for v in vac]
error_firstOrder = [np.sqrt(np.sum(np.square(C[3:,:]))) for C in C_ts]


f, ax = plt.subplots(1)
ax.plot(eigen_dist, error_firstOrder, color = "royalblue")
ax.set_xlim(xmin = 0)
ax.set_ylim(ymin = 0)
for i in range(0, len(time_lag), 4):
    ax.scatter(eigen_dist[i], error_firstOrder[i], s = 15, color = "royalblue")
    ax.annotate("{}".format(round(time_lag[i],3)), xy = (eigen_dist[i] - .001, error_firstOrder[i] + .02), weight='bold', fontsize = 6.5)
plt.xlabel("Eigen distance")
plt.annotate("Basis of 10 Hermite polynomials, 10,000 seconds of data. Lag times annotated.", (0,0), (0, -32), fontsize = 7, xycoords='axes fraction', textcoords='offset points', va='top')
plt.ylabel("Frobenius error of C(t) with top three rows removed")
plt.title("First order error in C(t) estimate")
plt.savefig("Graphs/firstOrderError.png")



'''
-------------------------------------------------------------------------------
Estimating Sampling Error
-------------------------------------------------------------------------------
'''

C_true = [np.diag([np.e**(-i*t) for i in range(m)]) for t in time_lag]
error_sampling_fro = [np.linalg.norm(C_hat_t[0:3,0:3] - np.dot(C_t, C_0[0:3,0:3]), 'fro') for C_hat_t,C_t,C_0 in zip(C_ts,C_true,C_0s)]
error_sampling_2 = [np.linalg.norm(C_hat_t[0:3,0:3] - np.dot(C_t, C_0[0:3,0:3]), 2) for C_hat_t,C_t,C_0 in zip(C_ts,C_true,C_0s)]

f, ax = plt.subplots(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax.plot(time_lag, error_sampling_fro, color = "green", label = "10,000 Seconds of Data")
ax.plot(time_lag, error_sampling_frox, color = "red", label = "2,000 Seconds of Data")
plt.legend()
ax.set_xlim(xmin = 0)
ax.set_ylim(ymin = 0)
plt.xlabel("Time Lag")
plt.annotate("Basis of 10 Hermite polynomials.", (0,0), (0, -32), fontsize = 7, xycoords='axes fraction', textcoords='offset points', va='top')
plt.ylabel(r"$\big| \big|\hat C(t) - C(t)\cdot \hat C(0)\big|\big|_F$")
plt.title("Estimate of Sampling Error")
plt.savefig("Graphs/samplingError_dataSizes.png")


'''
-------------------------------------------------------------------------------
Sampling error vs. Eigenvalue gap
-------------------------------------------------------------------------------
'''
dimension = 1
fineness  = 10
endpoint = 2.5
basis = [Hermite(0).to_fcn()]
basis = basis + [Hermite(n, d).to_fcn() for n in range(1, fineness) for d in range(dimension)]
basisSize = len(basis)
delta_t = .001
T = 1000
n = 100
h5 = tb.open_file("Trajectory_Data/OU_1D_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r')
a = h5.root.data
m = 3
l = basisSize
time_lag = np.hstack([np.linspace(delta_t, 5*delta_t, 4), np.linspace(6*delta_t, .25, 8)])
C_true = [np.diag([np.e**(-i*t) for i in range(basisSize)]) for t in time_lag]
starts = [5,15,20]

Cts = []
C0s = []
for i in starts:
    print("Now opening trajectory data.")
    h5 = tb.open_file("Trajectory_Data/OU_1D_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r')
    a = h5.root.data
    t = np.array(a[i:i+3,:])
    h5.close()
    vac = [VAC(basis, t, l, delta_t, dimension = dimension, update = True) for l in time_lag]
    print("Now getting C_ts")
    C_ts = [v.C_t() for v in vac]
    Cts.append(C_ts)
    print("Now gettings C_0s")
    C_0s = [v.C_0() for v in vac]
    C0s.append(C_0s)
    print("Done with {}th start".format(i))

error_sampling_fro = []
error_sampling_2 = []
eigen_dist = []
error_exact = []
error_precise_fro = []
error_precise_2 = []

for C_ts,C_0s in zip(Cts, C0s):
    Es = [C_hat_t - 1/2*np.dot(C_t, C_0) - 1/2*np.dot(C_t, C_hat_t) for C_hat_t,C_t,C_0 in zip(C_ts,C_true,C_0s)]
    error_sampling_fro.append([np.linalg.norm(E, 'fro') for E in Es])
    error_sampling_2.append([np.linalg.norm(E, 2) for E in Es])
    error_precise_fro.append([np.linalg.norm(E[3:,:], 'fro') for E in Es])
    error_precise_2.append([np.linalg.norm(E[3:,:], 2) for E in Es])


eigen_dist_exact = np.array([np.e**(-3*t) - np.e**(-4*t) for t in time_lag])
fig, ax = plt.subplots(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
[ax.loglog(time_lag, error, color = "green", alpha = .5) for error in error_sampling_fro]
ax.loglog(time_lag, np.array(eigen_dist_exact), color = "red", linewidth = 2, label = "Log Eigen Distance")

avg_error = np.zeros(len(time_lag))
for error in error_sampling_fro:
    avg_error += np.array(error) / len(error_sampling_fro)
ax.loglog(time_lag, avg_error, color = "green", linewidth = 2, label = "Average Log Sampling Error")
plt.legend(loc = "lower right")
ax.set_xlim(xmin = 0)
plt.xlabel("Time Lag")
plt.annotate("Basis of 10 Hermite polynomials, 3 trajectories of 3,000 seconds of data", (0,0), (0, -32), fontsize = 7, xycoords='axes fraction', textcoords='offset points', va='top')
plt.title("Sampling Error Frobenius Norm vs. Eigen Distance")
plt.savefig("Graphs/SamplingErrorvsEigenDistance_fro.png")

fig, ax = plt.subplots(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
[ax.loglog(time_lag, error, color = "green", alpha = .5) for error in error_sampling_2]
ax.loglog(time_lag, eigen_dist_exact, color = "red", linewidth = 2, label = "Log Eigen Distance")

avg_error = np.zeros(len(time_lag))
for error in error_sampling_2:
    avg_error += np.array(error) / len(error_sampling_2)
ax.loglog(time_lag, avg_error, color = "green", linewidth = 2, label = "Average Log Sampling Error")
plt.legend(loc = "lower right")
ax.set_xlim(xmin = 0)
plt.xlabel("Time Lag")
plt.annotate("Basis of 10 Hermite polynomials, 3 trajectories of 3,000 seconds of data", (0,0), (0, -32), fontsize = 7, xycoords='axes fraction', textcoords='offset points', va='top')
plt.title("Sampling Error 2-Norm vs. Eigen Distance")
plt.savefig("Graphs/SamplingErrorvsEigenDistance_2.png")


fig, ax = plt.subplots(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
[ax.loglog(time_lag, error, color = "green", alpha = .5) for error in error_precise_fro]
ax.loglog(time_lag, eigen_dist_exact, color = "red", linewidth = 2, label = "Log Eigen Distance")

avg_error = np.zeros(len(time_lag))
for error in error_precise_fro:
    avg_error += np.array(error) / len(error_precise_fro)
ax.loglog(time_lag, avg_error, color = "green", linewidth = 2, label = "Average Log Sampling Error")
plt.legend(loc = "lower right")
ax.set_xlim(xmin = 0)
plt.xlabel("Time Lag")
plt.annotate("Basis of 10 Hermite polynomials, 3 trajectories of 3,000 seconds of data", (0,0), (0, -32), fontsize = 7, xycoords='axes fraction', textcoords='offset points', va='top')
plt.title("Streamlined Sampling Fro Error from vs. Eigen Distance")
plt.savefig("Graphs/SamplingErrorvsEigenDistanceStreamlined_fro.png")

fig, ax = plt.subplots(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
[ax.loglog(time_lag, error, color = "green", alpha = .5) for error in error_precise_2]
ax.loglog(time_lag, eigen_dist_exact, color = "red", linewidth = 2, label = "Log Eigen Distance")

avg_error = np.zeros(len(time_lag))
for error in error_precise_2:
    avg_error += np.array(error) / len(error_precise_2)
ax.plot(time_lag, avg_error, color = "green", linewidth = 2, label = "Average Log Sampling Error")
plt.legend(loc = "lower right")
ax.set_xlim(xmin = 0)
plt.xlabel("Time Lag")
plt.annotate("Basis of 10 Hermite polynomials, 3 trajectories of 3,000 seconds of data", (0,0), (0, -32), fontsize = 7, xycoords='axes fraction', textcoords='offset points', va='top')
plt.title("Streamlined Sampling 2-Error from vs. Eigen Distance")
plt.savefig("Graphs/SamplingErrorvsEigenDistanceStreamlined_2.png")

'''
-------------------------------------------------------------------------------
Plotting Estinated Eigenfunctions
-------------------------------------------------------------------------------
'''

ev = evs[0]
estimated = [fcn_weighting(basis, v) for v in ev.T][::-1]
true = [fcn_weighting(basis_true, v) for v in w_f]

z = np.array([np.linspace(-1.2,1.2,300)])
w = [h(z) for h in estimated]
y = [h(z) for h in true]

plt.plot(z[0],w[0], "-r", label = "First")
plt.plot(z[0],w[1], "-b", label = "Second")
plt.plot(z[0],w[2], "-g", label = "Second")

plt.plot(z[0],y[0], "-r", label = "First")
plt.plot(z[0],y[1], "-b", label = "Second")
plt.plot(z[0],y[2], "-g", label = "Second")

plt.legend()
plt.show()


if dimension == 2:
    d1 , d2 = [np.linspace(-2, 2, 30), np.linspace(-2, 2, 30)]
    y, x = np.meshgrid(d1, d2)

    w = [np.array([[h(np.vstack([a,b])) for a in d1] for b in d2]) for h in estimated]
    # v = [np.array([[h(np.vstack([a,b])) for a in d1] for b in d2]) for h in true]

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = w[1]
    z = z[:-1, :-1]
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('First Eigenfunction of Transfer Operator \n (OU process with slow and fast relaxation times)')
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.plot(t[0,:10000], t[1,:10000], "darkgreen")
    plt.show()
