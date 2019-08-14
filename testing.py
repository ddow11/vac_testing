import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
import matplotlib.path as path
from hermite_poly import Hermite
from simple_models import simulate, VAC, well_well
from mpl_toolkits import mplot3d
from basis_sets import indicator


# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# ax.plot(z[:, 0], z[:, 1], z[:, 2])
# plt.show()
# basis = [Hermite(n) for n in range(10)]
# x = simulate(0, 1, 10000)
# ev = find_eigen(3, basis, x, 1)


basis = [indicator(fineness = 4,endpoint = 4, center =  i) for i in range(-4,4)]
x = simulate([0], .1, 100)

y = x.normal()
Y = VAC(basis, y, 1)
ev_y = Y.find_eigen(4)

z = x.potential(well_well)
Z = VAC(basis, z, 1)
ev_z = Z.find_eigen(4)




basis = [indicator(fineness = 4,endpoint = 4, center =  i) for i in range(-4,4)]
x = simulate([0,0], .1, 100)

y = x.normal()
Y = VAC(basis, y, 1)
ev_y = Y.find_eigen(4)

z = x.potential(well_well)
Z = VAC(basis, z, 1)
ev_z = Z.find_eigen(4)
