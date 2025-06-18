import numpy as np
from matplotlib import pyplot as plt


def gaussian_beam_profile(wavelength: float,
                          w0: float,
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
               f: float) -> np.ndarray:
    """
    Simula una lente delgada de focal f aplicando la fase cuadrática al frente de onda.

    Parámetros:
    -----------
    E_in : ndarray (Ny x Nx, complejo)
        Campo complejo antes de la lente.
    X, Y : ndarray (Ny x Nx)
        Coordenadas transversales en m (meshgrid).
    k    : float
        Número de onda 2π/λ·n (m^-1) en el medio donde está la lente.
    f    : float
        Distancia focal de la lente en m.

    Retorna:
    --------
    E_out : ndarray (Ny x Nx, complejo)
        Campo complejo justo después de la lente.
    """
    # fase parabólica de lente delgada
    phi_lens = np.exp(-1j * k / (2 * f) * (X**2 + Y**2))
    return E_in * phi_lens