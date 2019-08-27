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


'''
finding the true eigenvalues
'''
print("Finding true weightings.")

fineness  = 4
endpoint = 1
dimension = 1
basis_true = [Hermite(0).to_fcn()]
basis_true = basis_true + [Hermite(n, d).to_fcn() for n in range(1,fineness) for d in range(dimension)]
truebasisSize = len(basis_true)
delta_t = .001
T = 1000
n = 1000
length = round(T / delta_t)
optimal_timeLag = .3

h5 = tb.open_file("Trajectory_Data/DW_1D_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r')
a = h5.root.data
t = np.array(a[0:80,round(length *  .05):])
w_f = VAC(basis_true, t, optimal_timeLag, delta_t, dimension = dimension, update = True).find_eigen(truebasisSize)[1].T

distribution = np.hstack([a[d:d+dimension, round(length *.05):] for d in range(500,530, dimension)])

h5.close()

'''
-----------------------------------------
'''

print("Done with finding true weightings.")

fineness  = 6
endpoint = 1.8
basis = [Hermite(n).to_fcn() for n in range(fineness)]
basis = [indicator(fineness, endpoint, center = i).to_fcn() for i in  makegrid(endpoint, dimension = dimension, n = fineness)]
basisSize = len(basis)
delta_t = .001
T = 1000
n = 1000
length = round(T / delta_t)
h5 = tb.open_file("Trajectory_Data/DW_1D_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r')
a = h5.root.data
t = np.array(a[100:104,round(length *  .05):round(length * 1)])
h5.close()

time_lag = np.hstack((np.linspace(delta_t, 1, 10)))
print("Now getting eigenvalues.")
evs = [VAC(basis, t, l, delta_t, dimension = dimension, update = True).find_eigen(basisSize) for l in time_lag]

print("Calculating Phi's")

"Number of eigenfunctions to compare. Must be less than basisSize."
m = 3

Phi_g = np.array([f(distribution) for f in basis])
Phi_f = np.array([f(distribution) for f in basis_true])

print("Now calculating error.")
eigen_dist = [ev[0][basisSize - m] - ev[0][basisSize - m - 1] for ev in evs]

error = [L2subspaceProj_d(w_f = w_f[truebasisSize - m:], w_g = ev[1].T[basisSize - m:][::-1],
                        distribution = distribution, Phi_f = Phi_f, Phi_g = Phi_g)
                        for ev in evs]


print("Now plotting some graphs.")

plt.plot(eigen_dist, error)
plt.xlabel("Distance to nearest eigenvalue")
plt.ylabel("Error in estimated subspaces")
plt.title("Error in estimation with varying time lags (DW, 1D)")

print([time_lag[i] for i in range(len(eigen_dist)) if eigen_dist[i] == max(eigen_dist)])

ev = [[ev[0][i] for ev in evs] for i in range(m-1)]
[plt.plot(time_lag, ev[i]) for i in range(m-1)]

plt.xlabel("Time Lag")
plt.ylabel("Eigenvalues")
plt.title("Eigenvalues vs. Time Lag (OU, 1-D)")


"""
The third eigenvalue is well approximated until around .41 seconds of time lag,
then the approximation gets dramatically worth.
"""

"""
CODE FOR PLOTTING BELOW
"""


ev = evs[0]
estimated = [fcn_weighting(basis, v) for v in ev[1].T][::-1]
true = [fcn_weighting(basis_true, v) for v in w_f][::-1]

if dimension == 1:
    z = np.linspace(-1.5,1.5,20)
    w = [h(z) for h in estimated]
    y = [h(z) for h in true]
    #
    plt.plot(z,w[0], "-r", label = "First")
    plt.plot(z,w[1], "-b", label = "Second")
    plt.plot(z,w[2], "-g", label = "Third")
    # plt.plot(z[0],w[3], "-g", label = "Third")
    # #
    plt.plot(z,y[0], "-r", label = "First")
    plt.plot(z,y[1], "-b", label = "Second")
    plt.plot(z,y[2], "-g", label = "Third")
# # plt.plot(z[0],y[3], "-g", label = "Fourth")
#
# plt.legend()
# plt.show()

if dimension == 2:
    d1 , d2 = [np.linspace(-1.8, 1.8, 10), np.linspace(-1.8, 1.8, 10)]
    y, x = np.meshgrid(d1, d2)

    w = [np.array([[h(np.vstack([a,b])) for a in d1] for b in d2]) for h in estimated]
    # v = [np.array([[h(np.vstack([a,b])) for a in d1] for b in d2]) for h in true]

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = w[7]
    z = z[:-1, :-1]
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('pcolormesh')
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)

    plt.show()
