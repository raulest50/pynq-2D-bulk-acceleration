
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros de la simulación
Nx = 128  # Aumentamos la resolución para mejor visualización
Ny = 128
Lx = 2e-3  # 2 mm
Ly = 2e-3  # 2 mm
wavelength = 780e-9  # 780 nm
n_air = 1.003
k = 2 * np.pi / wavelength * n_air  # número de onda
f = 0.1  # distancia focal (10 cm)
strength = 1.0  # factor de fuerza de la lente

# Crear la malla de coordenadas
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y)

# Crear la máscara de fase de la lente
phi = np.exp(-1j * k * strength / (2 * f) * (X ** 2 + Y ** 2))

# Crear figura para visualizar las tres componentes
fig = plt.figure(figsize=(18, 10))
fig.suptitle('Visualización de la Máscara de Fase de una Lente Delgada', fontsize=16, y=0.98)

# Añadir texto explicativo
description = """
La máscara de fase de una lente delgada se modela como: φ(x,y) = exp(-i·k·(x²+y²)/(2f))
donde k es el número de onda (2π/λ), f es la distancia focal, y (x,y) son las coordenadas espaciales.
"""
fig.text(0.5, 0.91, description, ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# 1. Parte real de la máscara
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X*1e3, Y*1e3, np.real(phi), cmap='viridis', edgecolor='none', alpha=0.8)
ax1.set_title('Parte Real de la Máscara')
ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_zlabel('Amplitud')
ax1.view_init(elev=30, azim=45)  # Ajustar ángulo de vista
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# 2. Parte imaginaria de la máscara
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X*1e3, Y*1e3, np.imag(phi), cmap='plasma', edgecolor='none', alpha=0.8)
ax2.set_title('Parte Imaginaria de la Máscara')
ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Y (mm)')
ax2.set_zlabel('Amplitud')
ax2.view_init(elev=30, azim=45)  # Ajustar ángulo de vista
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

# 3. Valor absoluto de la máscara
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X*1e3, Y*1e3, np.abs(phi), cmap='inferno', edgecolor='none', alpha=0.8)
ax3.set_title('Valor Absoluto de la Máscara')
ax3.set_xlabel('X (mm)')
ax3.set_ylabel('Y (mm)')
ax3.set_zlabel('Amplitud')
ax3.view_init(elev=30, azim=45)  # Ajustar ángulo de vista
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

# Añadir información de parámetros
param_text = f"""
Parámetros:
- Longitud de onda (λ): {wavelength*1e9:.1f} nm
- Distancia focal (f): {f*100:.1f} cm
- Número de onda (k): {k:.2e} rad/m
- Factor de fuerza: {strength}
"""
fig.text(0.02, 0.02, param_text, fontsize=10, va='bottom', ha='left', 
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout(rect=[0, 0.05, 1, 0.9])  # Ajustar layout para acomodar el título y texto
plt.show()

# Guardar la figura
plt.savefig('lens_mask_visualization.png', dpi=300, bbox_inches='tight')
