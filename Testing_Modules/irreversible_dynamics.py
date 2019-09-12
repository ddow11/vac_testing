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
delta_t = .001
T = 1000
n = 100
length = round(T / delta_t)
print("Now opening trajectory data.")
h5 = tb.open_file("Trajectory_Data/UD_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r')
a = h5.root.data
start = 40
t = np.array(a[start:start + 6,:])
distribution = np.array(a[80:100,:])
h5.close()

time_lag = np.linspace(delta_t, 4, 20)
print("Now getting eigenvalues.")
vac = [VAC(basis, t, l, delta_t, dimension = dimension, update = True) for l in time_lag]
evs = [v.EDMD(basisSize) for v in vac]

# np.save("Trajectory_Data/egalue_hermite_shortlag_shortlag.npy", np.array([ev[0] for ev in evs]))
# np.save("Trajectory_Data/egvector_hermite_shortlag_shortlag.npy", np.array([ev[1] for ev in evs]))

print("Now calculating error.")

"Number of eigenfunctions to compare. Must be less than basisSize."
m = 6

basis_true = basis[:m]
w_f = np.identity(m)
distribution = np.hstack([distribution[d:d+dimension, :] for d in range(0,len(distribution), dimension)])
Phi_g = np.array([f(distribution) for f in basis])
Phi_f = np.array([f(distribution) for f in basis_true])


eigen_dist = np.absolute([ev[0][m] - ev[0][m+1] for ev in evs])

# evs = np.load("Trajectory_Data/egvector_hermite_shortlag_shortlag.npy", allow_pickle = True)

error = [L2subspaceProj_d(w_f = w_f, w_g = ev[1].T[:m][::-1],
                        distribution = distribution, Phi_f = Phi_f, Phi_g = Phi_g, normalize_f = False, orthoganalize = True)
                        for ev in evs]

np.save("underdamped_evectors_{}".format(start), [ev[1] for ev in evs])
np.save("underdamped_evalues_{}".format(start), [ev[0] for ev in evs])
np.save("underdampled_error_{}".format(start), error)

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


f,ax = plt.subplots(1)
mean = np.zeros(len(time_lag))
starts = [0,10,20,30,40]
for i in starts:
    error  = np.load("underdampled_error_{}.npy".format(i))
    mean += error / len(starts)
    ax.plot(time_lag, error)
ax.plot(time_lag, mean, color = "black", linewidth = 3, label = "Average")
plt.legend()
plt.xlabel("Time Lag")
plt.ylabel("Projection Error in estimated subspaces")
plt.title("Error in estimation with varying time lags (underdamped)")
plt.annotate("Using a basis of 5 1-D Hermite polynomials + pq. Multiple trajectories--5,000 seconds of data", (0,0), (0, -32), fontsize = 8, xycoords='axes fraction', textcoords='offset points', va='top')
ax.set_xlim(xmin = 0)
plt.savefig("Graphs/eigenfcnError_underdamped_HermiteBasis.png")


'''
-------------------------------------------------------------------------------
Eigenvalue decay
-------------------------------------------------------------------------------
'''

evs = np.load("underdamped_evalues_40.npy")
ev = np.array([ev for ev in evs]).T
magnitudes = [np.absolute(e) for e in ev]
magnitudes = np.argsort(magnitudes)
fig, ax = plt.subplots()
[ax.plot(time_lag, np.absolute(ev[i]), label = "{}th eigenvalue".format(i)) for i in range(0,3)]
plt.plot(ev[i].real, ev[i].imag)
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
