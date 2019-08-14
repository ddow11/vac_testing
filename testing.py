import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
import matplotlib.path as path
from hermite_poly import Hermite
from simple_models import simulate, VAC, well_well, makegrid, fcn_weighting, L2subspaceProj_d
from mpl_toolkits import mplot3d
from basis_sets import indicator
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show



basis_I = [indicator(fineness = 5, endpoint = 1, center = i) for i in makegrid(endpoint = 1, dimension = 2, n = 5)]

x = simulate([0,0], .1, 100)
# x.set_seed(5)

z = x.potential(well_well)
Z = VAC(basis_I, z, 1)
ev_z_I = Z.find_eigen(3)

basis_I = [i.to_fcn() for i in basis_I]

I_fcns_z = [fcn_weighting(basis_I, v) for v in ev_z_I[1].T][::-1]


# X = makegrid(1, dimension = 2, n = 5)
# Z = [h(X) for h in I_fcns_z]
#
# im = imshow(Z[0],cmap=cm.RdBu)
# cset = contour(Z[0],arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
# clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
# colorbar(im)

# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# ax.plot(z[:, 0], z[:, 1], z[:, 2])
# plt.show()
# basis = [Hermite(n) for n in range(10)]
# x = simulate(0, 1, 10000)
# ev = find_eigen(3, basis, x, 1)


#
# basis = [indicator(fineness = 4,endpoint = 4, center =  i) for i in range(-4,4)]
# x = simulate([0], .1, 100)
#
# y = x.normal()
# Y = VAC(basis, y, 1)
# ev_y = Y.find_eigen(4)
#
# z = x.potential(well_well)
# Z = VAC(basis, z, 1)
# ev_z = Z.find_eigen(4)
