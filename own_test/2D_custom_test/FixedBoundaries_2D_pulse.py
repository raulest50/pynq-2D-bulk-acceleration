import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Suponiendo que create_tridiagonal_matrix no se usa en este fragmento,
# puedes comentar o eliminar la importación si no la necesitas aquí:
# from MyHelperFunctions import create_tridiagonal_matrix

# Parámetros del haz gaussiano
w0 = 1.0          # Cintura del haz (en unidades espaciales)
x_max = 2.0       # Extensión máxima en x (en unidades espaciales)
y_max = 2.0       # Extensión máxima en y (en unidades espaciales)
Nx = 47           # Número de puntos en el eje x
Ny = 47           # Número de puntos en el eje y

# Crear grillas en x e y
x = np.linspace(-x_max, x_max, Nx)
y = np.linspace(-y_max, y_max, Ny)
X, Y = np.meshgrid(x, y)

# Perfil gaussiano 2D centrado en (0,0)
r2 = X**2 + Y**2
E0 = np.exp(-r2 / w0**2)

# ---------------------------------------------------
# Gráfico 1: Mapa de calor con imshow
# ---------------------------------------------------
plt.figure(figsize=(6, 5))
plt.imshow(
    np.abs(E0),
    extent=[-x_max, x_max, -y_max, y_max],
    origin='lower',
    cmap='inferno'
)
plt.colorbar(label='|E₀(x,y)|')
plt.title('Perfil gaussiano (2D) – imshow')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# ---------------------------------------------------
# Gráfico 2: Superficie 3D del mismo perfil
# ---------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Dibujar la superficie
surf = ax.plot_surface(
    X, Y, np.abs(E0),
    cmap='inferno',
    linewidth=0,
    antialiased=True,
    rcount=Ny,
    ccount=Nx
)

# Etiquetas y límites
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('|E₀(x,y)|')
ax.set_title('Perfil gaussiano (2D) – Surface 3D')
ax.set_xlim(-x_max, x_max)
ax.set_ylim(-y_max, y_max)
ax.set_zlim(0, 1)  # Como E0 está entre 0 y 1

# Ajustar ángulo de vista para mejor apreciación
ax.view_init(elev=30, azim=45)

# Añadir barra de colores
cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
cbar.set_label('|E₀(x,y)|')

plt.show()


# ---------------------------------------------------
# Solucion nmumerica del sistema 2D
# ---------------------------------------------------

dx = x[1] - x[0]
dy = y[1] - y[0]

# Definir la gama de distancias z para la animación
z_min = 0.0
z_max = 5 * zR  # Hasta 5 Rayleigh, por ejemplo
n_frames = 100
z_vals = np.linspace(z_min, z_max, n_frames)


