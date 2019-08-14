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


# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# ax.plot(z[:, 0], z[:, 1], z[:, 2])
# plt.show()
# basis = [Hermite(n) for n in range(10)]
# x = simulate(0, 1, 10000)
# ev = find_eigen(3, basis, x, 1)

m = 10
basis_H = [Hermite(n) for n in range(3)]
basis_I = [indicator(fineness = 6, endpoint = 2.5, center = i) for i in makegrid(endpoint = 2.5, n = 6)]

x = simulate([0], 1, 100000)
x.set_seed(5)

y = x.normal()
Y = VAC(basis_H, y, 1)
ev_y_H = Y.find_eigen(3)

# z = x.potential(well_well)
# Z = VAC(basis_H, z, 1)
# ev_z_H = Z.find_eigen(3)
#
# Y = VAC(basis_I, y, 1)
# ev_y_I = Y.find_eigen(3)
#
# Z = VAC(basis_I, z, 1)
# ev_z_I = Z.find_eigen(3)

basis_H = [h.to_fcn() for h in basis_H]
basis_I = [i.to_fcn() for i in basis_I]

H_fcns_y = [fcn_weighting(basis_H, v) for v in ev_y_H[1].T][::-1]
# H_fcns_z = [fcn_weighting(basis_H, v) for v in ev_z_H[1].T][::-1]
# I_fcns_y = [fcn_weighting(basis_I, v) for v in ev_y_I[1].T][::-1]
# I_fcns_z = [fcn_weighting(basis_I, v) for v in ev_z_I[1].T][::-1]


print(L2subspaceProj_d(basis_H, H_fcns_y, endpoint = 2.5, dimension = 1, n = 100))

# print(
# L2subspaceProj_d(H_fcns_y, I_fcns_y, endpoint = 2.5, dimension = 1, n = 100),
# L2subspaceProj_d(H_fcns_z, I_fcns_z, endpoint = 2.5, dimension = 1, n = 100)
# )

z = np.linspace(-2.5,2.5,100)
w = [h(z) for h in basis_H]
l = [h(z) for h in H_fcns_y]

plt.plot(z,l[0], "-r", label = "First")
plt.plot(z,l[1], "-b", label = "Second")
plt.plot(z,l[2], "-g", label = "Third")
plt.legend()
plt.title("Gaussian potential, first three estimated eigenfunctions")
plt.show()


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
