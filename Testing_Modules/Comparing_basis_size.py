from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
import matplotlib.path as path
from hermite_poly import Hermite
from simple_models import simulate, VAC, well_well, makegrid, fcn_weighting, L2subspaceProj_d, OU
from mpl_toolkits import mplot3d
from basis_sets import indicator

x = simulate([0], .1, 100)
x.set_seed(10)
z = x.potential(well_well)



f = simulate([0], .1, 1000)
f.set_seed(10)
y = f.potential(OU)

m = 3
basis_H = [Hermite(n) for n in range(m)]
Z = VAC(basis_H, y, 1)
ev_y_H = Z.find_eigen(3)

basis_H = [h.to_fcn() for h in basis_H]

H_fcns_y = [fcn_weighting(basis_H, v) for v in ev_y_H[1].T][::-1]


t = np.linspace(-1.7,1.7,100)
k = [h(z) for h in H_fcns_y]

plt.plot(z,k[0], "-m", label = "First, H")
plt.plot(z,k[1], "-c", label = "Second, H")
plt.plot(z,k[2], "-k", label = "Third, H")
plt.legend(loc=8, prop={'size': 7})
plt.title("Estimated eigenfcns with indicator and Hermite bases")
plt.show()

m = 10
basis_I_20 = [indicator(fineness = m, endpoint = 2, center = i) for i in makegrid(endpoint = 2, n = m)]
basis_I_40 = [indicator(fineness = 2*m, endpoint = 2, center = i) for i in makegrid(endpoint = 2, n = 2*m)]
basis_I_80 = [indicator(fineness = 4*m, endpoint = 2, center = i) for i in makegrid(endpoint = 2, n = 4*m)]



Z_20 = VAC(basis_I_20, z, 1)
Z_40 = VAC(basis_I_40, z, 1)
Z_80 = VAC(basis_I_80, z, 1)

ev_z_I_20 = Z_20.find_eigen(3)
print(20)

ev_z_I_40 = Z_40.find_eigen(3)
print(40)

ev_z_I_80 = Z_80.find_eigen(3)
print(80)

basis_I_20 = [i.to_fcn() for i in basis_I_20]
basis_I_40 = [i.to_fcn() for i in basis_I_40]
basis_I_80 = [i.to_fcn() for i in basis_I_80]

I_fcns_z_20 = [fcn_weighting(basis_I_20, v) for v in ev_z_I_20[1].T][::-1]
I_fcns_z_40 = [fcn_weighting(basis_I_40, v) for v in ev_z_I_40[1].T][::-1]
I_fcns_z_80 = [fcn_weighting(basis_I_80, v) for v in ev_z_I_80[1].T][::-1]


# print(
# L2subspaceProj_d(H_fcns_y, I_fcns_y, endpoint = 2.5, dimension = 1, n = 100),
# L2subspaceProj_d(H_fcns_z, I_fcns_z, endpoint = 2.5, dimension = 1, n = 100)
# )
t = np.linspace(-1.7,1.7,100)
o_20 = [h(t) for h in I_fcns_z_20]
o_40 = [h(t) for h in I_fcns_z_40]
o_80 = [h(t) for h in I_fcns_z_80]


plt.plot(t,o_20[0], "maroon", label = "First, n = 20")
plt.plot(t,o_20[1], "midnightblue", label = "Second, n = 20")
plt.plot(t,o_20[2], "darkgreen", label = "Third, n = 20")

plt.plot(t,o_40[0], "red", label = "First, n = 40")
plt.plot(t,o_40[1], "blue", label = "Second, n = 40")
plt.plot(t,o_40[2], "limegreen", label = "Third, n = 40")

plt.plot(t,o_80[0], "tomato", label = "First, n = 80")
plt.plot(t,o_80[1], "cornflowerblue", label = "Second, n = 80")
plt.plot(t,o_80[2], "darkgreen", label = "Third, n = 80")

plt.legend(loc=8, prop={'size': 5})
plt.title("Estimated eigenfcns with indicator bases")
plt.show()
