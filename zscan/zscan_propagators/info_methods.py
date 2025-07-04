import types

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D



def plot_beam_profile(Ex, x, y):
    intensity = np.abs(Ex) ** 2
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(x * 1e3, y * 1e3, intensity, shading='auto', cmap='inferno')
    plt.title('Haz Gaussiano inicial (Carmel X-780)')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.colorbar(label='Intensidad [a.u.]')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_beam_propagation(phi_history, x, y, dz, cmap='inferno'):
    """
    Visualizes beam propagation as a 3D surface plot with a slider to control the z-position.
    Uses fixed z-axis limits to better visualize beam expansion/contraction.

    Parameters:
    ----------
    phi_history : numpy.ndarray
        3D array containing beam profiles at each z-step (shape: [Nz+1, Ny, Nx])
    x : numpy.ndarray
        1D array of x coordinates
    y : numpy.ndarray
        1D array of y coordinates
    dz : float
        Step size in z direction
    cmap : str, optional
        Colormap to use for the 3D surface plot, default is 'inferno'
    """
    Nz = phi_history.shape[0] - 1  # Number of z steps

    # Calculate maximum intensity across all frames for fixed z-axis limits
    max_intensity = 0
    for z_idx in range(Nz + 1):
        intensity = np.abs(phi_history[z_idx])**2
        max_intensity = max(max_intensity, np.max(intensity))

    # Create figure with 3D axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)  # Make room for slider

    # Convert to mm for display
    x_mm = x * 1e3
    y_mm = y * 1e3

    # Create meshgrid for 3D plot
    X_mm, Y_mm = np.meshgrid(x_mm, y_mm)

    # Calculate intensity for initial profile
    intensity = np.abs(phi_history[0])**2

    # Create the initial 3D surface plot
    surf = ax.plot_surface(X_mm, Y_mm, intensity, cmap=cmap, 
                          edgecolor='none', alpha=0.8)

    # Set fixed z-axis limits based on maximum intensity across all frames
    ax.set_zlim(0, max_intensity)

    # Set labels and title
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_zlabel('Intensity [a.u.]')
    ax.set_title(f'Beam Profile at z = 0.00 mm')

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Normalized Intensity')

    # Create slider axis and slider
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    z_slider = Slider(
        ax=ax_slider,
        label='z position [mm]',
        valmin=0,
        valmax=Nz * dz * 1e3,  # Convert to mm
        valinit=0,
        valstep=dz * 1e3  # Convert to mm
    )

    # Update function for slider
    def update(val):
        # Calculate the index in phi_history
        z_idx = int(round(val / (dz * 1e3)))
        z_idx = min(z_idx, Nz)  # Ensure index doesn't exceed array bounds

        # Update intensity data
        intensity = np.abs(phi_history[z_idx])**2

        # Clear the current plot
        ax.clear()

        # Create new surface plot with updated data
        surf = ax.plot_surface(X_mm, Y_mm, intensity, cmap=cmap, 
                              edgecolor='none', alpha=0.8)

        # Reset labels and title
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_zlabel('Intensity [a.u.]')
        ax.set_title(f'Beam Profile at z = {val:.2f} mm')

        # Maintain fixed z-axis limits
        ax.set_zlim(0, max_intensity)

        # Redraw the figure
        fig.canvas.draw_idle()

    # Register the update function with the slider
    z_slider.on_changed(update)

    # Don't use tight_layout for 3D plots as it's not fully compatible
    plt.show()

    return fig, ax, z_slider


def intensity_from_field(U: np.ndarray,
                         c: float = 3e8,
                         epsilon0: float = 8.854e-12) -> np.ndarray:
    """
    Convierte U(x,y) en I(x,y) = ½·c·ε0·|U|² (W/m²).
    """
    return 0.5 * c * epsilon0 * np.abs(U)**2


def assess_intensity_zscan(
        Ex: np.ndarray,
        sample
) -> float:
    """
    Evalúa si I_peak es suficiente para medir n2 por Z-scan.

    Parámetros:
    - Ex: perfil de campo complejo
    - sample: objeto de la clase Sample

    Returns:
    - I_peak: valor máximo de intensidad en W/m²
    """
    # 1) Intensidad local
    I = intensity_from_field(Ex)
    I_peak = I.max()

    # 2) Coeficiente n2 del medio
    n2 = sample.n2

    # 3) Umbral práctico para Z-scan (1e11–1e12 W/m²) :contentReference[oaicite:15]{index=15}
    # Comentado para usar formato mejorado en el script principal
    # if I_peak >= 1e11:
    #     print(f"I_peak = {I_peak:.2e} W/m² → intensidad suficiente para Z-scan cerrado (n2 ≃ {n2:.2e}).")
    # else:
    #     print(f"I_peak = {I_peak:.2e} W/m² → intensidad insuficiente; considerar enfoque más fuerte o pulso más corto.")

    return I_peak


def compute_transmitance(Ein, Eout, dx, dy):
    Ii = np.abs(Ein)**2
    Io = np.abs(Eout)**2

    Pi = np.sum(Ii) * dx * dy
    Po = np.sum(Io) * dx * dy

    T = Po/Pi

    return T, Pi, Po

def compute_transmitance_physical(Ein: np.ndarray,
                                  Eout: np.ndarray,
                                  dx: float,
                                  dy: float,
                                  c: float = 3e8,
                                  eps0: float = 8.854e-12
                                  ) -> tuple[float, float, float]:
    """
    Calcula la transmitancia y las potencias de entrada/salida en unidades físicas (W).

    Parámetros:
    ----------
    Ein : np.ndarray
        Campo eléctrico complejo de entrada (V/m).
    Eout : np.ndarray
        Campo eléctrico complejo de salida (V/m).
    dx : float
        Espaciado en x (m).
    dy : float
        Espaciado en y (m).
    c : float, opcional
        Velocidad de la luz en el medio (m/s). Por defecto 3e8.
    eps0 : float, opcional
        Permitividad del vacío (F/m). Por defecto 8.854e-12.

    Retorna:
    -------
    T : float
        Transmitancia (adimensional).
    Pi : float
        Potencia de entrada (W).
    Po : float
        Potencia de salida (W).
    """
    # Intensidad [W/m²] a partir de |E|²
    Ii = 0.5 * c * eps0 * np.abs(Ein)**2
    Io = 0.5 * c * eps0 * np.abs(Eout)**2

    # Integración sobre el área [m²] → potencia [W]
    Pi = np.sum(Ii) * dx * dy
    Po = np.sum(Io) * dx * dy

    # Transmitancia adimensional
    T = Po / Pi if Pi != 0 else np.nan

    return T, Pi, Po


def plot_transmitance(T, z):
    # Plot z-scan transmission
    plt.figure(figsize=(10, 6))
    plt.plot(z, T, 'b.-')
    plt.xlabel('z position (m)')
    plt.ylabel('Transmission')
    plt.title('Z-scan Transmission')
    plt.grid(True)
    plt.show()
