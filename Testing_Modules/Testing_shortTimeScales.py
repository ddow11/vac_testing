import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.path as path
from hermite_poly import Hermite, Poly
from models_and_functions import simulate, VAC, well_well, makegrid, fcn_weighting, L2subspaceProj_d, OU, dot
from mpl_toolkits import mplot3d
from basis_sets import indicator
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import tables as tb
import math


dimension = 1
fineness  = 4
endpoint = 2.5
basis = [indicator(fineness, endpoint, center = i).to_fcn() for i in  makegrid(endpoint, dimension = dimension, n = fineness)]
basis = [Hermite(0).to_fcn()]
basis = basis + [Hermite(n, d).to_fcn() for n in range(1, fineness) for d in range(dimension)]
basisSize = len(basis)
delta_t = .0001
T = 1000
n = 80
length = round(T / delta_t)
print("Now opening trajectory data.")
h5 = tb.open_file("Trajectory_Data/OU_1D_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r')
a = h5.root.data
t = np.array(a[30:40,:])
h5.close()

time_lag = np.hstack([np.linspace(delta_t, .005, 2), np.linspace(.006, .3, 8), np.linspace(.31, 1, 8)])
C_t = [np.array([[1,0,0],[0,np.exp(-t),0],[0,0,np.exp(-2*t)]]) for t in time_lag]

data_sizes = [.5,1,4,10]

for i in data_sizes:
    s = t[0:math.ceil(i), : int((length * i / math.ceil(i)))]
    print(s.shape)
    print("Now caclulating C_t's.")
    C_tHat = [VAC(basis, s, l, delta_t, dimension = dimension, update = True).C_t()[0:3,0:3] for l in time_lag]
    np.save("Trajectory_Data/shortLag_{}Seconds.npy".format(i), np.array([time_lag, C_tHat]))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
for i in data_sizes:
    time_lag, C_hat = np.load("Trajectory_Data/shortLag_{}Seconds.npy".format(i), allow_pickle = True)
    error = [np.linalg.norm(C - C_h, 2) for C, C_h in zip(C_t, C_hat)]
    plt.plot(time_lag, error, label = "{} Thousand Seconds of Data".format(i))

plt.legend()
plt.xlabel("Time Lag")
plt.ylabel(r"$||C(t) - \hat{C}(t)||_{2}$")
# plt.annotate("$C(t)$ and $\hat{C}(t)$ normalized to have 2-norm 1", (0,0), (0, -28), fontsize = 8, xycoords='axes fraction', textcoords='offset points', va='top')
plt.title("C(t) error for various time lags and data sizes (OU process)", fontsize = 16)
plt.savefig("Graphs/C(t)_error_2.png")
