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
fineness  = 5
endpoint = 2
basis_true = [Hermite(n).to_fcn() for n in range(fineness)]
delta_t = .001
T = 10000
n = 1
length = round(T / delta_t)

h5 = tb.open_file("DW_1D_delta_t=.001,T=10000,n=1.h5", 'r')
a = h5.root.data
t = np.array([a[0,round(length *  .02):round(length * .5)]])
w_f = VAC(basis_true, t, .3, delta_t).find_eigen(4)[1].T

distribution = np.array([a[0,round(length *  .5):round(length * 1)]])




h5.close()

'''
-----------------------------------------
'''

print("Done with finding true weightings.")

fineness  = 20
endpoint = 1.8
basis = [Poly(n).to_fcn() for n in range(fineness)]
basis = [Hermite(n).to_fcn() for n in range(fineness)]
basis = [indicator(fineness, endpoint, center = i).to_fcn() for i in  makegrid(endpoint, dimension = 1, n = fineness)]
delta_t = .001
T = 1000
n = 10
length = round(T / delta_t)

h5 = tb.open_file("DW_1D_delta_t=.001,T=1000,n=10.h5", 'r')
a = h5.root.data
t = np.array([a[3,round(length *  .05):round(length * .3)]])

time_lag = np.hstack((np.linspace(delta_t, .4, 10), np.linspace(.41, 3, 15)))
print("Now getting eigenvalues.")
evs = [VAC(basis, t, l, delta_t).find_eigen(4) for l in time_lag]
print("Now calculating error.")


Phi_g = np.array([f(distribution) for f in basis])
Phi_f = np.array([f(distribution) for f in basis_true])

eigen_dist = [ev[0][2] - ev[0][1] for ev in evs]
error = [L2subspaceProj_d(w_f = w_f[2:], w_g = ev[1].T[2:][::-1],
                        distribution = distribution, Phi_f = Phi_f, Phi_g = Phi_g)
                        for ev in evs]

plt.plot(eigen_dist, error)
plt.xlabel("Distance to nearest eigenvalue")
plt.ylabel("Error in estimated subspaces")
plt.title("Error in estimation with varying time lags (Double Well, 1D)")

print([i for i in range(len(error)) if error[i] == min(error)])

"""
CODE FOR PLOTTING BELOW
"""


ev = evs[11]
estimated = [fcn_weighting(basis, v) for v in ev[1].T][::-1]
true = [fcn_weighting(basis_true, v) for v in w_f][::-1]

z = np.array([np.linspace(-1.2,1.2,300)])
w = [h(z) for h in estimated]
y = [h(z) for h in true]

# plt.plot(z[0],w[0], "-r", label = "First")
# plt.plot(z[0],w[1], "-b", label = "Second")
# plt.plot(z[0],w[2], "-g", label = "Third")
# plt.plot(z[0],w[3], "-g", label = "Third")
#
# plt.plot(z[0],y[0], "-r", label = "First")
# plt.plot(z[0],y[1], "-b", label = "Second")
# plt.plot(z[0],y[2], "-g", label = "Third")
# plt.plot(z[0],y[3], "-g", label = "Third")

plt.legend()
plt.show()
