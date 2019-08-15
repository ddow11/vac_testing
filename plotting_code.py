from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
import matplotlib.path as path
from hermite_poly import Hermite
from simple_models import simulate, VAC, well_well, makegrid, fcn_weighting, L2subspaceProj_d, OU
from mpl_toolkits import mplot3d
from basis_sets import indicator



m = 4
basis_H = [Hermite(n) for n in range(m)]
basis_I = [indicator(fineness = 2.5, endpoint = 2, center = i) for i in makegrid(endpoint = 2.5, n = 10)]

x = simulate([1], .1, 200)
# x.set_seed(10)

y = x.potential(OU)
Y = VAC(basis_H, y, 1)
ev_y_H = Y.find_eigen(3)

z = x.potential(well_well)
Z = VAC(basis_H, z, 1)
ev_z_H = Z.find_eigen(3)

Y = VAC(basis_I, y, 1)
ev_y_I = Y.find_eigen(3)

Z = VAC(basis_I, z, 1)
ev_z_I = Z.find_eigen(3)

basis_H = [h.to_fcn() for h in basis_H]
basis_I = [i.to_fcn() for i in basis_I]

H_fcns_y = [fcn_weighting(basis_H, v) for v in ev_y_H[1].T][::-1]
H_fcns_z = [fcn_weighting(basis_H, v) for v in ev_z_H[1].T][::-1]
I_fcns_y = [fcn_weighting(basis_I, v) for v in ev_y_I[1].T][::-1]
I_fcns_z = [fcn_weighting(basis_I, v) for v in ev_z_I[1].T][::-1]


print(L2subspaceProj_d(basis_H, H_fcns_y, endpoint = 2.5, dimension = 1, n = 100))

# print(
# L2subspaceProj_d(H_fcns_y, I_fcns_y, endpoint = 2.5, dimension = 1, n = 100),
# L2subspaceProj_d(H_fcns_z, I_fcns_z, endpoint = 2.5, dimension = 1, n = 100)
# )

t = np.linspace(-2,2,100)
w = [h(t) for h in basis_H]
l = [h(t) for h in H_fcns_y]
e = [h(t) for h in I_fcns_y]
k = [h(t) for h in H_fcns_z]
o = [h(t) for h in I_fcns_z]


plt.plot(t,o[0], "-r", label = "First, I")
plt.plot(t,o[1], "-b", label = "Second, I")
plt.plot(t,o[2], "-g", label = "Third, I")
plt.plot(t,k[0], "-m", label = "First, H")
plt.plot(t,k[1], "-c", label = "Second, H")
plt.plot(t,k[2], "-k", label = "Third, H")

plt.legend(loc=8, prop={'size': 7})
plt.title("Estimated eigenfcns with indicator and Hermite bases")
plt.show()
