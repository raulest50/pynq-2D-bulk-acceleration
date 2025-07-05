import numpy as np
from scipy.ndimage import gaussian_filter
from deep_tissue_imaging.elementos.plotting import plot_field_intensity, plot_field_phase

from matplotlib import pyplot as plt

from deep_tissue_imaging.elementos.tejidos import cerebro_emb_pez_cebra as tejido
import deep_tissue_imaging.elementos.domain as Domain
from deep_tissue_imaging.elementos.lasers import fuente_microscopia_1 as laser

def graficar_mascara_fase_aleatoria(X, Y, desviacion_fase=0.3, correlacion_m=5e-6, semilla=None):
    """
    Aplica una máscara de fase aleatoria suave al campo complejo phi.

    Parámetros:
        phi (ndarray): campo complejo original (E o phi).
        X, Y (ndarray): mallas espaciales 2D (en metros).
        desviacion_fase (float): desviación estándar de la fase en radianes.
        correlacion_m (float): longitud de correlación espacial en metros.
        semilla (int, opcional): semilla para reproducibilidad.

    Retorna:
        ndarray: campo complejo phi con fase aleatoria aplicada.
    """
    if semilla is not None:
        np.random.seed(semilla)

    shape = X.shape

    # Calcular dx y dy a partir de las mallas (en metros)
    dx = np.abs(X[0, 1] - X[0, 0])  # metros
    dy = np.abs(Y[1, 0] - Y[0, 0])  # metros

    # Longitud de correlación en número de píxeles
    sigma_x = correlacion_m / dx
    sigma_y = correlacion_m / dy

    # Ruido gaussiano con desviación deseada
    ruido = np.random.normal(loc=0.0, scale=desviacion_fase, size=shape)

    # Suavizado para imitar fluctuación estructural
    theta = gaussian_filter(ruido, sigma=(sigma_y, sigma_x), mode='reflect')
    mf = np.exp(1j * theta)

    # Graficar la parte real de la máscara de fase
    myplot(np.real(mf), X, Y, "parte real")

    # Graficar la parte imaginaria de la máscara de fase
    myplot(np.imag(mf), X, Y, "parte imaginaria")

    #magnitud
    myplot(np.abs(mf), X, Y, "magnitud")

def myplot(f, X, Y, title):
    """
    Plot the phase of phi1 field in a 2D color map with units in micrometers.

    Parameters:
    -----------
    phi : ndarray
        The complex field to plot (typically phi1 from propagation)
    X : ndarray
        The X coordinates meshgrid in meters
    Y : ndarray
        The Y coordinates meshgrid in meters

    Returns:
    --------
    fig, ax : tuple
        The figure and axis objects for further customization if needed
    """
    # Convert X and Y from meters to micrometers for display
    X_um = X * 1e6
    Y_um = Y * 1e6

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the phase as a 2D color map
    im = ax.pcolormesh(X_um, Y_um, f, cmap='twilight', shading='auto')

    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('f(x, y)')

    # Set labels with units in micrometers
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title(title)

    # Make the plot look nice
    plt.tight_layout()
    plt.show()
    return fig, ax


Lz = 241e-6 # 200um
Nz = 241
dz = Lz / Nz # 1um

Lx, Ly = 45e-6, 45e-6 # 45um x 45um
Nx, Ny = 128, 128
dx = Lx / Nx # 0.35um
dy = Ly / Ny # 0.35um

x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y)

k0 = 2*np.pi / laser.wavelength
k = k0 * tejido.n_0
sigma_phi = k * tejido.Dn * tejido.l_s
# Typical value for brain tissue (5 μm)
sigma_x = 5e-6

domain = Domain.Domain(X, Y, Nx, Ny, Nz, dx, dy, dz, 1e-12, k0, k, sigma_phi, sigma_x)

graficar_mascara_fase_aleatoria(X, Y, domain.sigma_phi*10, domain.sigma_x)
