import types

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D


def gaussian_beam_profile_old(w0: float,
                          E0: float,
                          Nx: int,
                          Ny: int,
                          Lx: float = None,
                          Ly: float = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a 2D complex Gaussian beam profile suitable for initializing a BPM simulation.

    Parameters:
    ----------
    wavelength : float
        Wavelength of the laser in meters.
    w0 : float
        Beam waist (radius at z = 0) in meters.
    E0 : float
        Amplitude of the electric field.
    Nx : int
        Number of points in the x-direction (resolution).
    Ny : int
        Number of points in the y-direction (resolution).
    Lx : float, optional
        Physical size of the x-domain in meters. Defaults to 6 * w0.
    Ly : float, optional
        Physical size of the y-domain in meters. Defaults to 6 * w0.

    Returns:
    -------
    Ex, x, y, X, Y : tuple
        Complex electric field matrix (2D), 1D spatial coordinates x and y, and meshgrid X, Y.
    """

    # Set default domain sizes if not specified
    if Lx is None:
        Lx = 6 * w0
    if Ly is None:
        Ly = 6 * w0

    # Define spatial grids
    x = np.linspace(-Lx / 2, Lx / 2, Nx)
    y = np.linspace(-Ly / 2, Ly / 2, Ny)
    X, Y = np.meshgrid(x, y)

    # Calculate radial distance squared from beam center
    r2 = X**2 + Y**2

    # Initial Gaussian beam profile (z = 0)
    Ex = E0 * np.exp(-r2 / w0**2).astype(np.complex128)

    return Ex, x, y, X, Y


def gaussian_beam_profile_physical(
    wavelength: float,
    w0: float,
    E0: float,
    Nx: int,
    Ny: int,
    Lx: float = None,
    Ly: float = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera U(x,y) de un haz gaussiano TEM00 usando parámetros físicos.

    Parámetros:
    - wavelength: longitud de onda λ (m)
    - w0: cintura del haz (m)
    - E0: amplitud pico del campo (V/m)
    - Nx, Ny: puntos en x e y
    - Lx, Ly: tamaño de dominio (m). Por defecto 6·w0.
    """
    if Lx is None: Lx = 6 * w0
    if Ly is None: Ly = 6 * w0

    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    X, Y = np.meshgrid(x, y)
    r2 = X**2 + Y**2

    # Perfil gaussiano: E0·exp(–r²/w0²) (modo TEM00) :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
    Ex = E0 * np.exp(-r2 / w0**2).astype(np.complex128)
    return Ex, x, y, X, Y


def compute_E0(P_avg: float, f_rep: float, tau: float, w0: float) -> float:
    """
    Calcula la amplitud pico E0 del campo eléctrico (V/m) para un haz gaussiano.

    Parámetros:
    - P_avg: potencia promedio del láser (W)
    - f_rep: tasa de repetición (Hz)
    - tau: duración de pulso FWHM (s)
    - w0: radio de cintura del haz (m)

    Retorna:
    - E0: amplitud pico (V/m)
    """
    # Energía y potencia pico según Sheik-Bahae et al. :contentReference[oaicite:5]{index=5}
    E_pulse = P_avg / f_rep
    P_peak = E_pulse / tau

    # Intensidad pico en el foco de un haz gaussiano :contentReference[oaicite:6]{index=6}
    I0 = 2 * P_peak / (np.pi * w0 ** 2)

    # Relación I ↔ campo en vacío :contentReference[oaicite:7]{index=7}
    c = 3e8  # m/s
    eps0 = 8.854e-12  # F/m
    E0 = np.sqrt(2 * I0 / (c * eps0))

    return E0


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


def apply_lens(E_in: np.ndarray,
               X: np.ndarray,
               Y: np.ndarray,
               k: float,
               f: float,
               strength: float = 1.0,
               amp_gain: float = 1.0,
               aperture: float | None = None) -> np.ndarray:
    """
    Simula una lente delgada con parámetros extra:

    - f         : focal (m) donde quieres el foco
    - strength  : factor sobre la curvatura de fase (f_eff = f/strength)
    - amp_gain  : factor de escalado global de amplitud (Intensity → amp_gain^2)
    - aperture  : diámetro de la lente (m). Si no es None, todo r > aperture/2 se bloquea.
    """
    # 1) fase parabólica con “fuerza” ajustable
    phi = np.exp(-1j * k * strength / (2 * f) * (X ** 2 + Y ** 2))

    # 2) aplicamos la lente
    E_out = E_in * phi

    # 3) ganancia de amplitud fija (para lograr cualquier aumento de intensidad)
    E_out *= amp_gain

    # 4) máscara de diafragma (opcional)
    if aperture is not None:
        mask = (X ** 2 + Y ** 2) <= (aperture / 2) ** 2
        E_out *= mask

    return E_out


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


def apply_lens_abcd(
        E_in: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        wavelength: float,
        f: float
) -> np.ndarray:
    """
    Aplica una lente delgada basada en la ley ABCD (Saleh & Teich, Cap. 3.2).

    Parámetros:
    - U_in: campo complejo de entrada
    - X, Y: mallas espaciales
    - wavelength: λ en m
    - f: distancia focal de la lente en m

    Teoría: la q-parameter se transforma según
      q_out = (A·q_in + B) / (C·q_in + D)
    con matriz de lente delgada A=1, B=0, C=-1/f, D=1 :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}.
    """
    k = 2 * np.pi / wavelength
    # Fase parabólica: exp(-i·k/(2f)·(x²+y²))
    phi = np.exp(-1j * k / (2 * f) * (X ** 2 + Y ** 2))
    return E_in * phi


def intensity_from_field(U: np.ndarray,
                         c: float = 3e8,
                         epsilon0: float = 8.854e-12) -> np.ndarray:
    """
    Convierte U(x,y) en I(x,y) = ½·c·ε0·|U|² (W/m²).
    """
    return 0.5 * c * epsilon0 * np.abs(U)**2


def assess_intensity_zscan(
        Ex: np.ndarray,
        sample: types.SimpleNamespace
) -> None:
    """
    Evalúa si I_peak es suficiente para medir n2 por Z-scan.

    Parámetros:
    - U: perfil de campo complejo
    - material: clave en MATERIAL_PARAMS, e.g. "CS2"
    - sample: diccionario con n2, etc.
    """
    # 1) Intensidad local
    I = intensity_from_field(Ex)
    I_peak = I.max()

    # 2) Coeficiente n2 del medio
    n2 = sample.n2

    # 3) Umbral práctico para Z-scan (1e11–1e12 W/m²) :contentReference[oaicite:15]{index=15}
    if I_peak >= 1e11:
        print(f"I_peak = {I_peak:.2e} W/m² → intensidad suficiente para Z-scan cerrado (n2 ≃ {n2:.2e}).")
    else:
        print(f"I_peak = {I_peak:.2e} W/m² → intensidad insuficiente; considerar enfoque más fuerte o pulso más corto.")


