import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from MyHelperFunctions import create_tridiagonal_matrix

# -------------------------
# Parámetros del dominio (normalizados)
dx = 0.05
dz = 0.01
x = np.arange(0, 12, dx)
z = np.arange(0, 50, dz)
Nx = len(x)
Nz = len(z)

# -------------------------
# Constantes físicas normalizadas
λ = 1.0                  # longitud de onda en unidades arbitrarias
k = 2 * np.pi / λ        # número de onda = 2π
kx =  2*k             # inclinación de fase (genera v_g = -0.3)

# -------------------------
# Parámetros del pulso inicial
x0 = 8.0                 # centro
w0 = 1                 # ancho

# Campo inicial en z = 0
E = np.exp(-((x - x0)**2) / w0**2) * np.exp(-1j * kx * x)

# -------------------------
# Construcción de matrices Crank–Nicolson
g  = 1j * dz / (4 * k * dx**2)
I  = create_tridiagonal_matrix(n=Nx, md=1, od=0)
D2 = create_tridiagonal_matrix(n=Nx, md=-2, od=1)

A = - I + g * D2
B = I + g * D2

# -------------------------
# Configuración de la figura
fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(E), lw=2)
ax.set_xlabel('x (unidades arb.)')
ax.set_ylabel('|E(x,z)|')
ax.set_title('Propagación Gaussiana (BPM, CN en z)')
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0, np.max(np.abs(E)) * 1.2)
ax.grid(True)

# -------------------------
# Paso de Crank–Nicolson (primer orden en z)
def paso_CN(E_prev):
    ratio = E_prev[-1] / E_prev[-2]
    ghost = E_prev[-1] * ratio
    b = B.dot(E_prev)
    b[-1] += ghost*g
    A_mod = A.copy()
    A_mod[-1, -1] += g * ratio
    E_next = np.linalg.solve(A_mod, b)
    return E_next

# -------------------------
# Animación: cada frame avanza un paso en z
def animate(i):
    global E
    E = paso_CN(E)
    line.set_ydata(np.abs(E))
    return line,

ani = FuncAnimation(fig, animate, frames=Nz, interval=5, blit=True)
plt.show()
