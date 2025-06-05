import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Parámetros del haz gaussiano y del sistema
w0 = 1.0  # Cintura del haz en z=0
lambda0 = 0.5  # Longitud de onda
k0 = 2 * np.pi / lambda0
zR = np.pi * w0 ** 2 / lambda0  # Largo de Rayleigh

# Parámetros de la grilla espacial en x e y
x_max = 1.0  # Para mejor visualización
y_max = 1.0
Nx = 200
Ny = 200
x = np.linspace(-x_max, x_max, Nx)
y = np.linspace(-y_max, y_max, Ny)
X, Y = np.meshgrid(x, y)
r2 = X ** 2 + Y ** 2

# Definir la gama de distancias z para la animación
z_min = 0.0
z_max = 5 * zR  # Hasta 5 Rayleigh, por ejemplo
n_frames = 100
z_vals = np.linspace(z_min, z_max, n_frames)

# Crear la figura y el eje 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Configurar límites y etiquetas (se hace sólo una vez)
ax.set_xlim(-x_max, x_max)
ax.set_ylim(-y_max, y_max)
ax.set_zlim(0, 1)  # Intensidad normalizada entre 0 y 1
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Intensidad |E(x,y,z)|²')
ax.set_title('Propagación de un haz gaussiano en 3D')

# Ángulo de vista fijo
ax.view_init(elev=30, azim=45)

# Calcular superficie inicial (frame 0)
w_z0 = w0 * np.sqrt(1 + (z_vals[0] / zR) ** 2)
I0 = (w0 / w_z0) ** 2 * np.exp(-2 * r2 / w_z0 ** 2)

# Dibujar la superficie inicial y guardar el objeto en una lista mutable
surf_container = [ax.plot_surface(
    X, Y, I0,
    cmap='inferno',
    linewidth=0,
    antialiased=True,
    vmin=0,
    vmax=1
)]

# Crear la colorbar UNA sola vez, sobre la superficie inicial
cbar = fig.colorbar(surf_container[0], ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Intensidad normalizada |E(x,y,z)|²')


def update(frame_idx):
    # Primero, eliminar la superficie anterior
    surf_container[0].remove()

    # Calcular z y perfil de intensidad en esta posición
    z = z_vals[frame_idx]
    w_z = w0 * np.sqrt(1 + (z / zR) ** 2)
    I = (w0 / w_z) ** 2 * np.exp(-2 * r2 / w_z ** 2)

    # Dibujar la nueva superficie y guardarla en surf_container[0]
    surf_container[0] = ax.plot_surface(
        X, Y, I,
        cmap='inferno',
        linewidth=0,
        antialiased=True,
        vmin=0,
        vmax=1
    )

    # Actualizar título para indicar la posición z
    ax.set_title(f'Propagación de haz gaussiano en 3D – z = {z:.2f}')

    # Devolver el artista dibujado (para que FuncAnimation lo actualice)
    return (surf_container[0],)


# Crear la animación
ani = animation.FuncAnimation(
    fig,
    update,
    frames=n_frames,
    blit=False,  # Con 3D no se puede usar blit=True si recreamos la superficie
    interval=100  # milisegundos entre frames
)

# Si quieres guardar a MP4 (requiere ffmpeg instalado en tu sistema),
# descomenta la siguiente línea:
# ani.save('gaussian_beam_propagation_3d.mp4', writer='ffmpeg', fps=15, dpi=200)

plt.show()
