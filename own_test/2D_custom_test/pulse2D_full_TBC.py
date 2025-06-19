import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from HelperFunctions import create_tridiagonal_matrix

eps = 1e-12

# ---------------------------------------------------
# Parámetros físicos (en unidades SI: metros, radianes)
# ---------------------------------------------------
lambda0 = 800e-9       # Longitud de onda: 800 nm
k = 2 * np.pi / lambda0  # Número de onda [rad/m]
w0 = 1e-3              # Cintura del haz (1 mm)
x_max = 2e-3           # 2 mm en x
y_max = 2e-3           # 2 mm en y
Nx = Ny = 47           # Puntos en x e y

# ---------------------------------------------------
# Construcción de la rejilla espacial (en metros)
# ---------------------------------------------------
x = np.linspace(-x_max, x_max, Nx)
y = np.linspace(-y_max, y_max, Ny)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

# ---------------------------------------------------
# Perfil gaussiano 2D en z = 0
# ---------------------------------------------------
r2 = X**2 + Y**2
f =  2 # 5mm
E0 = np.exp(-r2 / w0**2)   # E(x,y; z=0) # sin lente
#E0 = np.exp(-r2/w0**2) * np.exp(-1j * k * r2/(2*f)) # con lente

# ---------------------------------------------------
# Definición de z y paso en z
# ---------------------------------------------------
zR = np.pi * w0**2 / lambda0
z_min = 0.0
z_max = 5 * zR * 2    # Propagar hasta 5 Rayleigh
n_frames = 1000
z_vals = np.linspace(z_min, z_max, n_frames)
dz = z_vals[1] - z_vals[0]
print(f"z_max = {z_max}")
# ---------------------------------------------------
# Construcción de matrices tridiagonales
# ---------------------------------------------------
ung = 1j * dz / (4 * k * dx**2)
D = create_tridiagonal_matrix(Nx, md=-2 * ung, od=ung)
I = np.eye(Nx, dtype=complex)
A = I - D
B = I + D

# ---------------------------------------------------
# Función que avanza un paso en z usando split-step ADI
# ---------------------------------------------------
def paso_split_step(E_prev):
    E_inter = np.zeros_like(E_prev, dtype=complex)
    E_next = np.zeros_like(E_prev, dtype=complex)

    # Paso en x (recorre columnas)
    for j in range(Ny):

        if abs(E_prev[1, j]) < eps:
            ratio_x0 = 1.0
        else:
            ratio_x0 = E_prev[0, j] / E_prev[1, j]

        if abs(E_prev[-2, j]) < eps:
            ratio_xn = 1.0
        else:
            ratio_xn = E_prev[-1, j] / E_prev[-2, j]

        B[0, 0] = -2 * ung + 1 + ung * ratio_x0
        B[-1, -1] = -2 * ung + 1 + ung * ratio_xn
        b = B @ E_prev[:, j]
        A[0, 0] = 2 * ung + 1 - ung * ratio_x0
        A[-1, -1] = 2 * ung + 1 - ung * ratio_xn
        E_inter[:, j] = np.linalg.solve(A, b)

    # Paso en y (recorre filas)
    for i in range(Nx):

        if abs(E_inter[i, 1]) < eps:
            ratio_y0 = 1.0
        else:
            ratio_y0 = E_inter[i, 0] / E_inter[i, 1]

        if abs(E_inter[i, -2]) < eps:
            ratio_yn = 1.0
        else:
            ratio_yn = E_inter[i, -1] / E_inter[i, -2]

        B[0, 0] = -2 * ung + 1 + ung * ratio_y0
        B[-1, -1] = -2 * ung + 1 + ung * ratio_yn
        b = B @ E_inter[i, :]
        A[0, 0] = 2 * ung + 1 - ung * ratio_y0
        A[-1, -1] = 2 * ung + 1 - ung * ratio_yn
        E_next[i, :] = np.linalg.solve(A, b)

    return E_next

# ---------------------------------------------------
# Preparar figura y eje 3D para animación
# ---------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-x_max, x_max)
ax.set_ylim(-y_max, y_max)
ax.set_zlim(0, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('|E(x,y)|')
ax.set_title('Propagación del haz gaussiano')

# Inicializar superficie con el perfil z=0
surface = ax.plot_surface(
    X, Y, np.abs(E0),
    cmap='inferno', linewidth=0, antialiased=True,
    rcount=Ny, ccount=Nx
)

# Campo inicial global para actualizar en animate
E_current = E0.copy()

# ---------------------------------------------------
# Método animate para actualizar cada cuadro
# ---------------------------------------------------
def animate(frame_idx):
    global E_current, surface

    # Avanzar un paso en z
    E_current = paso_split_step(E_current)

    # Borrar superficie anterior
    ax.cla()
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)
    ax.set_zlim(0, 1)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('|E(x,y)|')

    # Dibujar nueva superficie |E_current|
    surface = ax.plot_surface(
        X, Y, np.abs(E_current),
        cmap='inferno', linewidth=0, antialiased=True,
        rcount=Ny, ccount=Nx
    )
    ax.set_title(f'Propagación, z = {z_vals[frame_idx]:.3e} m')
    return surface,

# ---------------------------------------------------
# Crear animación con FuncAnimation
# ---------------------------------------------------
anim = FuncAnimation(
    fig,
    animate,
    frames=n_frames,
    interval=100,   # milisegundos entre cuadros
    blit=False
)

plt.show()

# (Opcional) Guardar la animación:
# anim.save('propagacion_haz.mp4', writer='ffmpeg', dpi=200)


print(f" len(z) = {len(z_vals)}")