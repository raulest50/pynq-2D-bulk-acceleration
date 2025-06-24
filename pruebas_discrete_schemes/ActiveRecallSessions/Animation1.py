import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

L=10
c=1 
A=1
k = np.pi / L  # wave number
omega = c * k  # angular frequency
Nx = 1000  # number of spatial points

x = np.linspace(0, L, Nx)

ti = 0
tf = 50
Nt = 200
t = np.linspace(ti, tf, Nt, dtype=float)

def standing_wave(x, t):
    return A * np.sin(5*k * x) * np.cos(omega * t)

fig, ax = plt.subplots()
ax.set_xlabel("posicion x")
ax.set_ylabel("Amplitud Onda")
ax.set_title("Onda Estacionaria")
ax.set_ylim(-A, A)

line, = ax.plot(x, standing_wave(x, t[0]), lw=2, color='blue')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)


def animate(i):
    line.set_ydata(standing_wave(x, t[i]))
    time_text.set_text(f't = {t[i]:.2f}')
    return line


ani = FuncAnimation(fig, animate, frames=len(t), interval=50)
plt.show()