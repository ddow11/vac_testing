import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
import matplotlib.path as path
from hermite_poly import Hermite, Poly
from simple_models import simulate, VAC, well_well, makegrid, fcn_weighting, L2subspaceProj_d, OU, dot
from mpl_toolkits import mplot3d
from basis_sets import indicator
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import tables as tb

dimension = 2
fineness  = 4
endpoint = 2.5
basis = [indicator(fineness, endpoint, center = i).to_fcn() for i in  makegrid(endpoint, dimension = dimension, n = fineness)]
basis = [Hermite(0).to_fcn()]
basis = basis + [Hermite(n, d).to_fcn() for n in range(1, fineness) for d in range(dimension)]
basisSize = len(basis)
delta_t = .001
T = 1000
n = 200
length = round(T / delta_t)
print("Now opening trajectory data.")
h5 = tb.open_file("Trajectory_Data/OU_1D_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r')
a = h5.root.data
t = np.array(a[20:26,round(length *  0):round(length * 1)])
h5.close()

time_lag = np.hstack((np.linspace(delta_t, 1, 10), np.linspace(1, 3, 10)))
print("Now getting eigenvalues.")
evs = [VAC(basis, t, l, delta_t, dimension = dimension, update = True).find_eigen(basisSize) for l in time_lag]
print("Now calculating error.")

distribution = np.random.normal(np.zeros([dimension,int(1e6)]), 1)
"Number of eigenfunctions to compare. Must be less than basisSize."
m = 4

basis_true = basis[0:m]
w_f = np.identity(m)


Phi_g = np.array([f(distribution) for f in basis])
Phi_f = np.array([f(distribution) for f in basis_true])



eigen_dist = [ev[0][basisSize - m] - ev[0][basisSize - m - 1] for ev in evs]

error = [L2subspaceProj_d(w_f = w_f, w_g = ev[1].T[basisSize - m:][::-1],
                        distribution = distribution, Phi_f = Phi_f, Phi_g = Phi_g)
                        for ev in evs]

plt.plot(eigen_dist, error)
plt.xlabel("Distance to nearest eigenvalue")
plt.ylabel("Error in estimated subspaces")
plt.title("Error in estimation with varying time lags (OU, 1D)")

ev = [[ev[0][i] for ev in evs] for i in range(m-1)]
[plt.plot(time_lag, ev[i]) for i in range(m-1)]

plt.legend()
plt.xlabel("Time Lag")
plt.ylabel("Eigenvalues")
plt.title("Eigenvalues vs. Time Lag (OU, 1-D)")


print([i for i in range(len(error)) if error[i] == min(error)])


"""
CODE FOR PLOTTING BELOW
"""


ev = evs[10]
estimated = [fcn_weighting(basis, v) for v in ev[1].T][::-1]
true = [fcn_weighting(basis_true, v) for v in w_f]
#
# z = np.array([np.linspace(-1.2,1.2,300)])
# w = [h(z) for h in estimated]
# y = [h(z) for h in true]
#
# plt.plot(z[0],w[0], "-r", label = "First")
# plt.plot(z[0],w[1], "-b", label = "Second")
# plt.plot(z[0],w[2], "-g", label = "Second")
#
# plt.plot(z[0],y[0], "-r", label = "First")
# plt.plot(z[0],y[1], "-b", label = "Second")
# plt.plot(z[0],y[2], "-g", label = "Second")
#
# plt.legend()
# plt.show()


if dimension == 2:
    d1 , d2 = [np.linspace(-2, 2, 30), np.linspace(-2, 2, 30)]
    y, x = np.meshgrid(d1, d2)

    w = [np.array([[h(np.vstack([a,b])) for a in d1] for b in d2]) for h in estimated]
    v = [np.array([[h(np.vstack([a,b])) for a in d1] for b in d2]) for h in true]

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = w[3]
    z = z[:-1, :-1]
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('pcolormesh')
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)

    plt.show()
