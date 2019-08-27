import scipy.stats
from scipy.special import hermite
from scipy.linalg import eigh
import numpy as  np
import matplotlib.pyplot as plt
from hermite_poly import Hermite

def well_well(x):
    return - x ** 3 / 2 + 2 * x

def next_state(x_1, delta_t, V):
    xsi = np.random.normal(0,1,1)
    x_2 = x_1 + V(x_1)*delta_t + np.sqrt(2*delta_t) * xsi
    return x_2

def simulate(x_0, delta_t, T, V):
    trajectory = np.array([x_0])
    x_n = x_0
    n = round(T/delta_t)
    for i in range(0,n):
        x_n = next_state(x_n, delta_t, V)
        trajectory = np.append(trajectory, x_n)
    return trajectory

x = simulate(0,.1,500, well_well)

plt.plot(x)
plt.show()
