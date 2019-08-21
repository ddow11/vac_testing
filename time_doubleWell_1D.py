import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
import matplotlib.path as path
from hermite_poly import Hermite
from simple_models import simulate, VAC, well_well, makegrid, fcn_weighting, L2subspaceProj_d, OU
from mpl_toolkits import mplot3d
from basis_sets import indicator
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import tables as tb

fineness  = 10
endpoint = 1.5
basis = [indicator(fineness, endpoint, center = i).to_fcn() for i in  makegrid(endpoint, dimension = 1, n = fineness)]
basis = [Hermite(n).to_fcn() for n in range(fineness)]
delta_t = .01
T = 5000
n = 1000

h5 = tb.open_file('trajectories.h5', 'r')
a = h5.root.data
trajectory = [a[1,:]]

V = VAC(basis, trajectory, .03, delta_t)
ev = V.find_eigen(3)

time_lag = np.linspace(0.01, 5, 40)
evs = np.array([VAC(basis, trajectory, l, delta_t).find_eigen(4)[0] for l in time_lag])

plt.plot(evs[:,2])
plt.plot(evs[:,1])
plt.plot(evs[:,0])

s = evs[:,2] - evs[:,1]
''' max at l = 3 => t = 3*delta_t = .03'''

h5.close()
