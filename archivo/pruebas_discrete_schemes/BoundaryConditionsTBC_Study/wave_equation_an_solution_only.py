import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import from the helpers package
from archivo.helpers import u_gaussian

# Parameters
L = 10.0          # Domain length
c = 4.0          # Wave speed
A = 1.0          # Initial displacement amplitude
B = 0.5          # Initial velocity amplitude
omega = np.pi * c / L

Nx = 1200
Nt = 200

# Spatial and temporal grids
x = np.linspace(0, L, Nx)
tf = L /(1.8*c)  # half travel
ti = 0
t = np.linspace(ti, tf, Nt)

# Set up the figure
fig, ax = plt.subplots()
line, = ax.plot(x, u_gaussian(x, 0, σ=0.3, c=c))
ax.set_xlim(0, L)
ax.set_ylim(-abs(A) - abs(B/omega) - 0.1, abs(A) + abs(B/omega) + 0.1)
ax.set_xlabel('x')
ax.set_ylabel('u(x, t)')
ax.set_title('1D Wave Equation Single-Mode Solution')

# Animation update function
def animate(i):
    y = u_gaussian(x, t[i], σ=0.3, c=c)
    line.set_ydata(y)
    ax.set_title(f'time t = {t[i]:.2f}')
    return line,

# Create animation
ani = FuncAnimation(fig, animate, frames=len(t), interval=50)

# Display
plt.show()
