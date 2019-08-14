from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
import matplotlib.path as path
from hermite_poly import Hermite
from simple_models import simulate, VAC, well_well, makegrid, fcn_weighting, L2subspaceProj_d
from mpl_toolkits import mplot3d
from basis_sets import indicator



m = 3
basis_H = [Hermite(n) for n in range(m)]
basis_I = [indicator(fineness = 5, endpoint = 2.5, center = i) for i in makegrid(endpoint = 2.5, n = 5)]

x = simulate([0], .1, 100)
x.set_seed(5)

y = x.normal()
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

z = np.linspace(-2.5,2.5,100)
w = [h(z) for h in basis_H]
l = [h(z) for h in H_fcns_y]
e = [h(z) for h in I_fcns_y]
k = [h(z) for h in H_fcns_z]
o = [h(z) for h in I_fcns_z]


plt.plot(z,o[0], "-r", label = "First")
plt.plot(z,o[1], "-b", label = "Second")
plt.plot(z,o[2], "-g", label = "Third")
plt.legend()
plt.title("Double well, estimated eigenfcns with indicator basis (n = 5)")
plt.show()
