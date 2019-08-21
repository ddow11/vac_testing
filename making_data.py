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

'''
creating some data
'''
# delta_t = .001
# T = 1000
# n = 10
# x = simulate(np.random.normal(0,1), delta_t, T, n)
#
# f = tb.open_file("DW_1D_delta_t=.001,T=1000,n=10.h5", "w")
# filters = tb.Filters(complevel=5, complib='blosc')
# out = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(n, np.rint(T/delta_t)), filters=filters)
# out[:,:] = x.potential_lots(well_well)
# f.close()
#
# print("Done with double well samples.")

# delta_t = .001
# T = 50000
# n = 1
# x = simulate(np.random.normal(0,1), delta_t, T, n)
#
# f = tb.open_file("DW_1D_delta_t=.001,T=10000,n=1.h5", "w")
# filters = tb.Filters(complevel=5, complib='blosc')
# out = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(n, np.rint(T/delta_t)), filters=filters)
# out[:,:] = x.potential_lots(well_well)
# f.close()
#
# print("Done with double well long sample.")
#
#
#
delta_t = .001
T = 5000
n = 10
x = simulate(np.random.normal(0,1), delta_t, T, n)

f = tb.open_file("OU_1D_delta_t=.001,T=5000,n=10.h5", "w")
filters = tb.Filters(complevel=5, complib='blosc')
out = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(n, np.rint(T/delta_t)), filters=filters)
out[:,:] = x.potential_lots(OU)
f.close()

print("Done with OU.")
