from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import tables as tb
from pylab import *
import matplotlib.ticker as ticker

dimension = 1
delta_t = .0005
T = 100
n = 20
total = round(T / delta_t)
base = np.ones(round(total / 5))
gammas = np.hstack([base * .25, base * .5, base, base *4, base * 10])
h5 = tb.open_file("Trajectory_Data/UDG_delta_t={},T={},n={}.h5".format(delta_t, T, n), 'r') # retrieves the trajectory data
a = h5.root.data
x1 = a[:]
h5.close()

y = x1[14]
x = np.linspace(0, T, total)

speed = 150

def init():
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlim(-3, 3)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,

fig, ax = plt.subplots()
fig.suptitle("Langevin dynamics with increasing friction.")
line, = ax.plot([], [], lw=2)
xdata, ydata = ([], [])
text = ax.text(.1, 2, "Gamma = 0")
ax.set_xlabel("Time")
ax.set_ylabel("Position")
point = ax.plot([0], [0], marker='o', markersize=5.5, color = "black")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

def run(num):
    # update the data
    current = num*speed
    past = round(max(current - 15/delta_t, 0))
    xdata = x[past : current]
    ydata = y[past : current]
    ax.set_xlim(x[past], x[current] + 4)
    text.set_position((x[past] + 1, 2))
    text.set_text("Gamma = {}".format(round(gammas[current], 3)))
    point[0].set_xdata(x[current])
    point[0].set_ydata(y[current])
    line.set_data(xdata, ydata)
    return line,

ani = animation.FuncAnimation(fig, run, range(1,round(len(x) / speed) - 1),
                              repeat=False, init_func=init, save_count=700, interval = 20)

dpi = 100

ani.save('test.mp4')

plt.show()
