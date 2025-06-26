import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from zscan_custom.operadores_solvers import single_bpm_step_only_linear_medium


class Lente:

    def __init__(self, w_min: float, lamda_laser: float, foco: float):
        self.w_min = w_min
        self.lamda_laser = lamda_laser
        self.z_0 = np.pi*self.w_min**2/lamda_laser
        self.foco = foco

    def get_mascara_fase(self, X:np.ndarray, Y:np.ndarray) -> np.ndarray:
        """
        Devuelve la mascara de fase de la lente,
        dados los puntos X e Y en el plano de la lente.
        """
        R2 = X**2 + Y**2
        k = 2 * np.pi / self.lamda_laser
        phase = np.exp(-1j * (k * R2 )/ (2 * self.foco))
        gain = np.exp(-(k * R2) / (2 * self.z_0))
        mascara = phase * gain
        return mascara


def visualizar_mascara_fase(mascara: np.ndarray, X: np.ndarray, Y: np.ndarray):
    """
    Función para visualizar las componentes de una máscara de fase.

    Args:
        mascara: Matriz compleja que representa la máscara de fase
        X: Matriz de coordenadas X
        Y: Matriz de coordenadas Y

    Returns:
        None (muestra la figura)
    """
    # Crear figura para visualizar las tres componentes
    fig = plt.figure(figsize=(18, 10))
    titulo = "Visualización de Máscara de Fase"
    fig.suptitle(titulo, fontsize=16, y=0.98)

    # Convertir coordenadas a milímetros para mejor visualización
    X_mm = X * 1e3
    Y_mm = Y * 1e3

    # 1. Parte real de la máscara
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X_mm, Y_mm, np.real(mascara), cmap='viridis', edgecolor='none', alpha=0.8)
    ax1.set_title('Parte Real')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Amplitud')
    ax1.view_init(elev=30, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # 2. Parte imaginaria de la máscara
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X_mm, Y_mm, np.imag(mascara), cmap='plasma', edgecolor='none', alpha=0.8)
    ax2.set_title('Parte Imaginaria')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Amplitud')
    ax2.view_init(elev=30, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    # 3. Magnitud (valor absoluto) de la máscara
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X_mm, Y_mm, np.abs(mascara), cmap='inferno', edgecolor='none', alpha=0.8)
    ax3.set_title('Magnitud')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_zlabel('Amplitud')
    ax3.view_init(elev=30, azim=45)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

    plt.tight_layout(rect=[0, 0.05, 1, 0.9])
    plt.show()

    return fig


def visualizar_propagacion_lateral(mascara: np.ndarray, X: np.ndarray, Y: np.ndarray, 
                                  k: float, distancia_max: float, num_pasos: int = 100,
                                  corte_y: int = None, titulo: str = "Propagación Lateral del Láser"):
    """
    Función para visualizar el perfil de propagación lateral de un láser después de pasar por una máscara de fase.

    Args:
        mascara: Matriz compleja que representa la máscara de fase
        X: Matriz de coordenadas X
        Y: Matriz de coordenadas Y
        k: Número de onda (2π/λ)
        distancia_max: Distancia máxima de propagación (metros)
        num_pasos: Número de pasos de propagación
        corte_y: Índice de la fila para el corte en Y (si es None, se usa el centro)
        titulo: Título del gráfico

    Returns:
        Figura de matplotlib
    """
    # Obtener dimensiones y parámetros
    Ny, Nx = mascara.shape
    dx = X[0, 1] - X[0, 0]  # Espaciado en X
    dy = Y[1, 0] - Y[0, 0]  # Espaciado en Y
    dz = distancia_max / num_pasos  # Paso de propagación

    # Si no se especifica el corte, usar el centro
    if corte_y is None:
        corte_y = Ny // 2

    # Inicializar el campo con la máscara de fase
    campo = np.copy(mascara)

    # Crear matriz para almacenar la intensidad en cada paso
    intensidad_xz = np.zeros((num_pasos + 1, Nx))

    # Guardar la intensidad inicial (corte en y)
    intensidad_xz[0, :] = np.abs(campo[corte_y, :])**2

    # Propagar el campo y guardar la intensidad en cada paso
    for i in range(num_pasos):
        # Propagar un paso usando BPM
        campo = single_bpm_step_only_linear_medium(campo, k, dz, dx, dy)

        # Guardar la intensidad en el corte y
        intensidad_xz[i + 1, :] = np.abs(campo[corte_y, :])**2

    # Normalizar la intensidad para mejor visualización
    intensidad_norm = intensidad_xz / np.max(intensidad_xz)

    # Crear mallas para la visualización
    z = np.linspace(0, distancia_max, num_pasos + 1)
    x = X[0, :]
    Z, X_mesh = np.meshgrid(z, x, indexing='ij')

    # Calcular el ancho del haz en cada posición z
    beam_widths = np.zeros(num_pasos + 1)
    for i in range(num_pasos + 1):
        # Encontrar el índice del máximo de intensidad en esta posición z
        max_idx = np.argmax(intensidad_xz[i, :])
        max_val = intensidad_xz[i, max_idx]

        # Encontrar los puntos donde la intensidad cae a 1/e² (≈13.5%) del máximo
        threshold = max_val * (1/np.e**2)

        # Encontrar los índices donde la intensidad cruza el umbral
        above_threshold = intensidad_xz[i, :] > threshold
        if np.sum(above_threshold) > 0:
            # Calcular el ancho como la distancia entre los puntos extremos
            indices = np.where(above_threshold)[0]
            width_idx = indices[-1] - indices[0]
            beam_widths[i] = width_idx * dx * 1e3  # Convertir a mm
        else:
            beam_widths[i] = 0

    # Encontrar la posición del beam waist mínimo (foco)
    # Ignorar los primeros pasos para evitar artefactos cerca del origen
    start_idx = int(num_pasos * 0.05)  # Ignorar el primer 5% de los pasos

    if np.any(beam_widths[start_idx:] > 0):
        # Buscar el mínimo solo en la región después de start_idx
        valid_widths = beam_widths[start_idx:]
        local_min_idx = np.argmin(valid_widths[valid_widths > 0])

        # Convertir el índice local al índice global
        valid_indices = np.where(valid_widths > 0)[0]
        if len(valid_indices) > 0:
            min_width_idx = start_idx + valid_indices[local_min_idx]
            min_width_z = z[min_width_idx] * 1e3  # Convertir a mm
            min_width_val = beam_widths[min_width_idx]
        else:
            min_width_idx = 0
            min_width_z = 0
            min_width_val = 0
    else:
        min_width_idx = 0
        min_width_z = 0
        min_width_val = 0

    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 8))

    # Convertir a mm para mejor visualización
    X_mm = X_mesh * 1e3
    Z_mm = Z * 1e3

    # Crear mapa de calor
    im = ax.pcolormesh(X_mm, Z_mm, intensidad_norm, cmap='inferno', shading='auto')

    # Añadir barra de color
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Intensidad Normalizada')

    # Configurar etiquetas y título
    ax.set_xlabel('Posición X (mm)')
    ax.set_ylabel('Distancia de Propagación Z (mm)')
    ax.set_title(titulo)

    # Añadir línea indicando la posición del corte
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)

    # Marcar la posición del foco (beam waist mínimo)
    if min_width_val > 0:
        # Línea horizontal en la posición del foco
        ax.axhline(y=min_width_z, color='cyan', linestyle='-', linewidth=1.5, label='Posición del foco')

        # Encontrar el centro del haz en la posición del foco
        foco_perfil = intensidad_xz[min_width_idx, :]
        centro_idx = np.argmax(foco_perfil)
        centro_x = x[centro_idx] * 1e3  # Convertir a mm

        # Punto en el centro del foco
        ax.plot(centro_x, min_width_z, 'o', color='yellow', markersize=10, zorder=5)

        # Línea vertical que pasa por el foco
        ax.axvline(x=centro_x, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)

        # Rectángulo que marca el área del foco
        width_half = min_width_val / 2
        rect = plt.Rectangle((centro_x - width_half, min_width_z - 2), 
                            width_half * 2, 4, 
                            linewidth=1.5, edgecolor='yellow', facecolor='none', 
                            linestyle='-', alpha=0.7, zorder=4)
        ax.add_patch(rect)

        # Anotación con información del foco
        ax.annotate(f'FOCO\nDistancia focal: {min_width_z:.2f} mm\nAncho mínimo: {min_width_val:.2f} mm',
                   xy=(centro_x, min_width_z), xytext=(centro_x + width_half * 3, min_width_z - 10),
                   arrowprops=dict(facecolor='yellow', shrink=0.05, width=2, alpha=0.7),
                   color='white', fontsize=12, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8),
                   ha='center')

    # Añadir información sobre parámetros
    info_text = f"Parámetros:\n" \
                f"- Número de onda (k): {k:.2e} rad/m\n" \
                f"- Distancia máxima: {distancia_max*1e3:.1f} mm\n" \
                f"- Resolución: {Nx}×{Ny} puntos\n" \
                f"- Corte en Y: índice {corte_y}\n" \
                f"- Distancia focal: {min_width_z:.2f} mm\n" \
                f"- Ancho mínimo del haz: {min_width_val:.2f} mm"

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            color='white')

    plt.tight_layout()
    plt.show()

    return fig





if __name__ == "__main__":
    # Ejemplo de uso
    Nx, Ny = 128, 128
    Lx, Ly = 2e-3, 2e-3  # 2 mm

    # Crear malla de coordenadas
    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    X, Y = np.meshgrid(x, y)

    # Crear una lente
    lamda_laser = 780e-9  # 780 nm
    lente = Lente(w_min=0.5e-3, lamda_laser=lamda_laser, foco=0.1)

    # Obtener la máscara de fase
    mascara = lente.get_mascara_fase(X, Y)

    # Visualizar la máscara
    visualizar_mascara_fase(mascara, X, Y)

    # Calcular el número de onda (k)
    k = 2 * np.pi / lamda_laser

    # Visualizar la propagación lateral del láser
    print("\nVisualizando la propagación lateral del láser...")
    distancia_max = 0.2  # 20 cm
    visualizar_propagacion_lateral(
        mascara=mascara,
        X=X,
        Y=Y,
        k=k,
        distancia_max=distancia_max,
        num_pasos=200,
        titulo=f"Propagación Lateral del Láser (λ={lamda_laser*1e9:.1f} nm)"
    )
