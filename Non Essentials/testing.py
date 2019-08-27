import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
import matplotlib.path as path
from hermite_poly import Hermite, Poly
from simple_models import simulate, VAC, well_well, makegrid, fcn_weighting, L2subspaceProj_d, OU
from mpl_toolkits import mplot3d
from basis_sets import indicator
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import tables as tb


fineness  = 10
endpoint = 1.2
basis = [Poly(n).to_fcn() for n in range(fineness)]
basis = [indicator(fineness, endpoint, center = i).to_fcn() for i in  makegrid(endpoint, dimension = 1, n = fineness)]
basis = [Hermite(n).to_fcn() for n in range(fineness)]
delta_t = .001
T = 10000
n = 1000

h5 = tb.open_file("DW_1D_delta_t=.001,T=10000,n=10.h5", 'r')
a = h5.root.data
trajectory = np.array([a[1,:]])

V = VAC(basis, trajectory, .385, delta_t)
ev = V.find_eigen(4)

time_lag = np.linspace(delta_t, .5, 40)
evs = np.array([VAC(basis, trajectory, l, delta_t).find_eigen(4)[0] for l in time_lag])

plt.plot(evs[:,2])
plt.plot(evs[:,1])
plt.plot(evs[:,0])

s = evs[:,2] - evs[:,1]
print([i for i in range(len(s)) if s[i] == max(s)])
'''time_lag[i] = .385'''


H_fcns = [fcn_weighting(basis, v) for v in ev[1].T][::-1]



# z = np.array([np.linspace(-1.2,1.2,300)])
# w = [h(z) for h in H_fcns]
# plt.plot(z[0],w[0], "-r", label = "First")
# plt.plot(z[0],w[1], "-b", label = "Second")
# # plt.plot(z[0],w[2], "-g", label = "Third")
# # plt.plot(z[0],w[3], label = 'Fourth')
# # plt.plot(z[0],w[4], label = 'fifth')
# plt.legend()
# plt.title("Double well, estimated eigenfcns with indicator basis (n = 30,100)")
# plt.show()
#




h5.close()
