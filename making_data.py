import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
import matplotlib.path as path
from hermite_poly import Hermite
from models_and_functions import simulate, VAC, well_well, makegrid, fcn_weighting, L2subspaceProj_d, OU
from mpl_toolkits import mplot3d
from basis_sets import indicator
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import tables as tb

'''
creating some data
'''
# delta_t = .001
# T = 1000
# n = 10
# x = simulate(delta_t, T, n)
#
# f = tb.open_file("DW_1D_delta_t=.001,T=1000,n=10.h5", "w")
# filters = tb.Filters(complevel=5, complib='blosc')
# out = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(n, np.rint(T/delta_t)), filters=filters)
# out[:,:] = x.potential_lots(well_well)
# f.close()
#
# print("Done with double well samples.")

# delta_t = .001
# T = 1000
# n = 1000
# x = simulate(delta_t, T, n)
#
# f = tb.open_file("Trajectory_Data/DW_1D_delta_t=.001,T=1000,n=1000.h5", "w")
# filters = tb.Filters(complevel=5, complib='blosc')
# out = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(n, np.rint(T/delta_t)), filters=filters)
# out[:,:] = x.potential_lots(well_well, update = True)
# f.close()
#
# print("Done with double well long sample.")



# delta_t = .001
# T = 1000
# n = 200
# x = simulate(delta_t, T, n)
#
# f = tb.open_file("Trajectory_Data/OU_1D_delta_t={},T={},n={}.h5".format(delta_t, T, n), "w")
# filters = tb.Filters(complevel=5, complib='blosc')
# out = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(n, np.rint(T/delta_t)), filters=filters)
# out[:,:] = x.normal(update = True)
# f.close()
#
# print("Done with OU.")

delta_t = .001
T = 1000
n = 150
speeds = [.05,1,5]
x = simulate(delta_t, T, n = n)

f = tb.open_file("Trajectory_Data/OU_1D_delta_t={},T={},n={},speeds={}.h5".format(delta_t, T, n, speeds), "w")
filters = tb.Filters(complevel=5, complib='blosc')
out = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(n, np.rint(T/delta_t)), filters=filters)
out[:,:] = x.normal(speed = speeds, update = True)
f.close()

print("Done with OU varying speeds.")

# delta_t = .01
# T = 1000
# n = 160
# x = simulate(delta_t, T, n = n)
#
# f = tb.open_file("Trajectory_Data/OU_1D_delta_t={},T={},n={}.h5".format(delta_t, T, n), "w")
# filters = tb.Filters(complevel=5, complib='blosc')
# out = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(n, np.rint(T/delta_t)), filters=filters)
# out[:,:] = x.normal(update = True)
# f.close()
#
# print("Done with OU varying speeds.")


delta_t = .0005
T = 100
n = 20
total = round(T / delta_t)
base = np.ones(round(total / 5))
gammas = np.hstack([base * .25, base * .5, base, base *4, base * 10])

np.linspace(.05, 20, round(T / delta_t))
x = simulate(delta_t, T, n = n)

f = tb.open_file("Trajectory_Data/UDG_delta_t={},T={},n={}.h5".format(delta_t, T, n), "w")
filters = tb.Filters(complevel=5, complib='blosc')
out = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(n, np.rint(T/delta_t)), filters=filters)
out[:,:] = x.underdampedApproxGamma(gammas, update = True)
f.close()
