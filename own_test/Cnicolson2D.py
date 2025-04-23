

import numpy as np
from numpy import pi
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

a = .5  #  alpha constant

t_final = 2

L = 10
n = 2
m = 3

N = 20

x = np.linspace(start=0, stop=L, num=N)
y = np.linspace(start=0, stop=L, num=N)

h = x[2] - x[1]

X, Y = np.meshgrid(x, y)

T_init = np.sin((n*pi*X)/L)*np.sin((m*pi*Y)/L)

T_10 = T_init*np.exp(-a * ((m*pi/L)**2 + (n*pi/L)**2) * t_final)


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Plot initial temperature as a wireframe
wire1 = ax.plot_wireframe(X, Y, T_init, color='C0')
scatter1 = ax.scatter(X, Y, T_init, color='C0', marker='o', label='Initial Points')

# Plot temperature at t=10 as a wireframe
wire2 = ax.plot_wireframe(X, Y, T_10, color='C1')
scatter2 = ax.scatter(X, Y, T_10, color='C1', marker='o', label=f'Temperature Points (t={t_final})')

# Add labels
ax.set_title("Temperature Surfaces in One Plot")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Temperature")

# Create a legend (note: plot_wireframe doesn't automatically create legend handles)
legend_lines = [
    Line2D([0], [0], color='C0', lw=2, label='Initial T(x,y,0)'),
    Line2D([0], [0], color='C1', lw=2, label=f'Temperature T(x,y,{t_final})')
]
ax.legend(handles=legend_lines, loc='upper right')

plt.tight_layout()
plt.show()

