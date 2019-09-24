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

dimension = 2
fineness  = 5
endpoint = 2.5
basis = [indicator(fineness, endpoint, center = i).to_fcn() for i in  makegrid(endpoint, dimension = dimension, n = fineness)]
basis = [Hermite(0).to_fcn()]
def f(x):
    return np.prod(x, axis = 0)
basis.append(f)
basis = basis + [Hermite(n, d).to_fcn() for n in range(1, fineness) for d in range(dimension)]
basisSize = len(basis)
delta_t = .01
T = 1000
n = 160
length = round(T / delta_t)
print("Now opening trajectory data.")
h5 = tb.open_file("Trajectory_Data/UD_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r')
a = h5.root.data
start = 0
t = np.array(a[start:start + 20,:])
distribution = np.array(a[130:160,:])
h5.close()

time_lag = np.linspace(delta_t, 4, 20)

'''
-------------------------------------------------------------------------------
Sample Trajectory
-------------------------------------------------------------------------------
'''

fig,ax = plt.subplots(1)
ax.plot(t[0,:10000], label = "Position")
ax.plot(t[1,:10000], label = "Momentum")
plt.xlabel("Time (units of 10^(-3) seconds)")
plt.title("Underdamped Pendulum dynamics")
plt.legend()
plt.annotate("The position is the integral of the momentum.", (0,0), (0, -32), fontsize = 8, xycoords='axes fraction', textcoords='offset points', va='top')
ax.set_xlim(xmin = 0)
plt.savefig("Graphs/underdamped_sampleTrajectory.png")


'''
-------------------------------------------------------------------------------
What does error vs. time lag look like?
-------------------------------------------------------------------------------
'''


time_lag = np.linspace(delta_t, 4, 20)
mean = np.zeros(len(time_lag))
starts = [0,10,20]
m = 6
basis_true = basis[:m]
w_f = np.identity(m)
distribution = np.hstack([distribution[d:d+dimension, :] for d in range(0,len(distribution), dimension)])
Phi_g = np.array([f(distribution) for f in basis])
Phi_f = np.array([f(distribution) for f in basis_true])

for i in starts:
    print("Now opening trajectory data.")
    h5 = tb.open_file("Trajectory_Data/UD_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r')
    a = h5.root.data
    start = 0
    t = np.array(a[i:i + 10,:])
    h5.close()
    print("Now getting eigenvalues.")
    vac = [VAC(basis, t, l, delta_t, dimension = dimension, update = True) for l in time_lag]
    evs = [v.EDMD(basisSize) for v in vac]
    print("Now calculating error.")
    error = [L2subspaceProj_d(w_f = w_f, w_g = ev[1].T[:m][::-1],
                            distribution = distribution, Phi_f = Phi_f, Phi_g = Phi_g, normalize_f = False, orthoganalize = True)
                            for ev in evs]
    np.save("Trajectory_Data/underdamped_evectors_{}".format(i), [ev[1] for ev in evs])
    np.save("Trajectory_Data/underdamped_evalues_{}".format(i), [ev[0] for ev in evs])
    np.save("Trajectory_Data/underdampled_error_{}".format(i), error)
    print("Done with  trajectory starting at {}".format(i))

fig, ax = plt.subplots(1)
mean = np.zeros(len(time_lag))
for i in starts:
    error  = np.load("Trajectory_Data/underdampled_error_{}.npy".format(i))
    mean += error / len(starts)
    ax.plot(time_lag, error, color = "green", alpha = .5)
ax.plot(time_lag, mean, color = "green", linewidth = 3, label = "Average")
plt.legend(loc = "lower left")
plt.xlabel("Time Lag")
plt.ylabel("Projection Error in estimated subspaces")
plt.title("Error in estimation with varying time lags (underdamped)")
plt.annotate("Using a basis of 5 1-D Hermite polynomials + pq. Four trajectories, each 7,000 seconds of data", (0,0), (0, -32), fontsize = 8, xycoords='axes fraction', textcoords='offset points', va='top')
ax.set_xlim(xmin = 0)
plt.savefig("Graphs/eigenfcnError_underdamped_HermiteBasis.png")

"""
Plotting error vs. spectral gap.
"""
fig, ax = plt.subplots(1)
avgError = np.zeros(len(time_lag))
avgEigendist = np.zeros(len(time_lag))
for i in starts:
    evs = np.load("Trajectory_Data/underdamped_evalues_{}.npy".format(i))
    evs = np.array([ev[np.argsort(np.absolute(ev))] for ev in evs]).T[::-1]
    eigen_gap = np.array([np.absolute(evs[m - 1][j] - evs[m][j]) for j in range(len(time_lag))])
    error = np.load("Trajectory_Data/underdampled_error_{}.npy".format(i))
    ax.plot(eigen_gap, error, color = "green", alpha = .3)
    avgError += error / len(starts)
    avgEigendist += eigen_gap / len(starts)

ax.plot(avgEigendist, avgError, color = "green", linewidth = 3, label = "Average")

for i in range(0, len(time_lag), 2):
    ax.scatter(avgEigendist[i], avgError[i], s = 18, color = "green")
    ax.annotate("{}".format(round(time_lag[i],3)), xy = (avgEigendist[i] - .001, avgError[i] + .015), weight='bold', fontsize = 6.5)
plt.legend(loc = "lower left")
plt.xlabel("Spectral Gap")
plt.ylabel("Projection Error in estimated subspaces")
plt.title("Error in estimation with varying time lags (underdamped)")
plt.annotate("Using a basis of 5 1-D Hermite polynomials + pq. Six trajectories, each 7,000 seconds of data", (0,0), (0, -32), fontsize = 8, xycoords='axes fraction', textcoords='offset points', va='top')
ax.set_xlim(xmin = 0)
plt.savefig("Graphs/eigenfcnErrorvsSepctralGap_UD_HermiteBasis.png")

'''
-------------------------------------------------------------------------------
Does exact error equal estimated error? Yes.
-------------------------------------------------------------------------------
'''

time_lag = np.linspace(delta_t, 4, 20)
mean = np.zeros(len(time_lag))
starts = [0,20,40,60]
m = 6
error_exact = []
for i in starts:
    evs = np.load("Trajectory_Data/underdamped_evectors_{}.npy".format(i))
    Q = [np.linalg.qr(v)[0] for v in evs]
    error_exact.append([np.sqrt(m - np.sum(np.square(np.linalg.svd(q[0:m,0:m])[1]))) for q in Q])

fig, ax = plt.subplots(1)
meanError = np.zeros(len(time_lag))
meanErrorExact = np.zeros(len(time_lag))
j = 0
for error_e in error_exact:
    i = starts[j]
    error  = np.load("Trajectory_Data/underdampled_error_{}.npy".format(i))
    meanError += error / len(error_exact)
    meanErrorExact += np.array(error_e) / len(error_exact)
    ax.plot(error_e, error, color = "green", alpha = .5)
    j += 1
ax.plot(meanErrorExact, meanError, color = "green", linewidth = 3, label = "Average")
plt.legend(loc = "lower right")
plt.savefig("Graphs/EstimatedvsExactError.png")


'''
-------------------------------------------------------------------------------
Imbedding Theorem Testing
-------------------------------------------------------------------------------
'''

time_lag = np.linspace(delta_t, 4, 20)
mean = np.zeros(len(time_lag))
starts = [50,60,70,80,90,100]
m = 6
basis_true = basis[:m]
w_f = np.identity(m)
dimension = 2
distribution = np.hstack([distribution[d:d+dimension, :] for d in range(0,len(distribution), dimension)])
Phi_g = np.array([f(distribution) for f in basis])
Phi_f = np.array([f(distribution) for f in basis_true])
back_lag = 100

for i in starts:
    print("Now opening trajectory data.")
    h5 = tb.open_file("Trajectory_Data/UD_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r')
    a = h5.root.data
    t = np.array(a[i:i + 10,:])
    h5.close()
    t = t[slice(None,None,2)]
    t = np.hstack([np.array([t[i,:t.shape[1] - back_lag], t[i,back_lag:]]) for i in range(t.shape[0])])
    print("Now getting eigenvalues.")
    vac = [VAC(basis, t, l, delta_t, dimension = dimension, update = True) for l in time_lag]
    evs = [v.EDMD(basisSize) for v in vac]
    print("Now calculating error.")
    error = [L2subspaceProj_d(w_f = w_f, w_g = ev[1].T[:m][::-1],
                            distribution = distribution, Phi_f = Phi_f, Phi_g = Phi_g, normalize_f = False, orthoganalize = True)
                            for ev in evs]
    np.save("Trajectory_Data/UDTakens_evectors_{}".format(i), [ev[1] for ev in evs])
    np.save("Trajectory_Data/UDTakens_evalues_{}".format(i), [ev[0] for ev in evs])
    np.save("Trajectory_Data/UDTakens_error_{}".format(i), error)
    print("Done with trajectory starting at {}".format(i))

fig, ax = plt.subplots(1)
mean = np.zeros(len(time_lag))
for i in starts:
    error  = np.load("Trajectory_Data/UDTakens_error_{}.npy".format(i))
    mean += error / len(starts)
    ax.plot(time_lag, error, color = "green", alpha = .3)
ax.plot(time_lag, mean, color = "green", linewidth = 3, label = "Average")
plt.legend(loc = "lower right")
plt.xlabel("Time Lag")
plt.ylabel("Projection Error in estimated subspaces")
plt.title("Error in estimating using time-lagged imbedding (underdamped)")
plt.annotate("Using a basis of 5 1-D Hermite polynomials + pq. Four trajectories, each 7,000 seconds of data, ", (0,0), (0, -32), fontsize = 8, xycoords='axes fraction', textcoords='offset points', va='top')
ax.set_xlim(xmin = 0)
plt.savefig("Graphs/eigenfcnError_UDImbedding_HermiteBasis.png")


"""
Plotting error vs. spectral gap.
"""
fig, ax = plt.subplots(1)
avgError = np.zeros(len(time_lag))
avgEigendist = np.zeros(len(time_lag))
for i in starts:
    evs = np.load("Trajectory_Data/UDTakens_evalues_{}.npy".format(i))
    evs = np.array([ev[np.argsort(np.absolute(ev))] for ev in evs]).T[::-1]
    eigen_gap = np.array([np.absolute(evs[m - 1][j] - evs[m][j]) for j in range(len(time_lag))])
    error = np.load("Trajectory_Data/UDTakens_error_{}.npy".format(i))
    ax.plot(eigen_gap, error, color = "green", alpha = .3)
    avgError += error / len(starts)
    avgEigendist += eigen_gap / len(starts)

ax.plot(avgEigendist, avgError, color = "green", linewidth = 3, label = "Average")

for i in range(0, len(time_lag), 2):
    ax.scatter(avgEigendist[i], avgError[i], s = 18, color = "green")
    ax.annotate("{}".format(round(time_lag[i],3)), xy = (avgEigendist[i] - .001, avgError[i] + .015), weight='bold', fontsize = 6.5)
plt.legend(loc = "lower left")
plt.xlabel("Spectral gap")
plt.ylabel("Projection Error in estimated subspaces")
plt.title("Error in estimating using time-lagged imbedding (underdamped)")
plt.annotate("Using a basis of 5 1-D Hermite polynomials + pq. Seven trajectories, each 7,000 seconds of data", (0,0), (0, -32), fontsize = 8, xycoords='axes fraction', textcoords='offset points', va='top')
ax.set_xlim(xmin = 0)
plt.savefig("Graphs/eigenfcnErrorvsSpectralGap_UDImbedding.png")

'''
-------------------------------------------------------------------------------
Eigenvalue decay
-------------------------------------------------------------------------------
'''

time_lag = np.linspace(delta_t, 4, 20)
evs = np.load("Trajectory_Data/underdamped_evalues_40.npy")
evs = np.array([ev[np.argsort(np.absolute(ev))] for ev in evs]).T[::-1]
fig, ax = plt.subplots()
[ax.plot(time_lag, np.absolute(evs[i]), label = "{}th eigenvalue".format(i)) for i in range(0,6)]
plt.legend()
plt.xlabel("Time Lag")
plt.ylabel("Norm of Eigenvalue")
plt.title("Eigenvalue Decay")
plt.savefig("Graphs/Underdamped_EigenvalueDecay.png")


fig, ax = plt.subplots()
[plt.scatter(evs[i].real, evs[i].imag, label = "{}th eigenvalue".format(i)) for i in range(1,6)]
plt.legend()
for i in range(0, len(time_lag), 3):
    [plt.text(evs[j][i].real, evs[j][i].imag, "{}".format(round(time_lag[i],3)), fontsize = 7, weight = "bold") for j in range(1,6,3)]

plt.title("Eigenvalues vs. Time Lag (Underdamped)")


'''
-------------------------------------------------------------------------------
Are estimated eigenfunctions orthogonal?
-------------------------------------------------------------------------------
'''

ev = evs[len(evs) // 3]
a = np.hstack([t[d:d+dimension, :] for d in range(0,len(distribution), dimension)])
A = np.array([f(a) for f in basis])
fDotg = np.dot(np.dot(ev[1].T[basisSize - m:][::-1], A), np.dot(ev[1].T[basisSize - m:][::-1], A).T) / a.size


'''
-------------------------------------------------------------------------------
Plotting Estinated Eigenfunctions
-------------------------------------------------------------------------------
'''

ev = evs[0][1]
estimated = [fcn_weighting(basis, v) for v in ev.T][::-1]
true = [fcn_weighting(basis_true, v) for v in w_f]

if dimension == 2:
    d1 , d2 = [np.linspace(-3.3, 3.3, 30), np.linspace(-3.3, 3.3, 30)]
    y, x = np.meshgrid(d1, d2)

    w = [np.array([[h(np.vstack([a,b])) for a in d1] for b in d2]) for h in estimated]
    v = [np.array([[h(np.vstack([a,b])) for a in d1] for b in d2]) for h in true]

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = w[2]
    z = z[:-1, :-1]
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('First Eigenfunction of Transfer Operator \n (OU process with slow and fast relaxation times)')
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    # plt.plot(t[0,:10000], t[1,:10000], "darkgreen")
    plt.show()
