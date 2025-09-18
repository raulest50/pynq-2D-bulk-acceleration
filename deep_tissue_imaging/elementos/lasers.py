import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def campo_tem00(X, Y, w0, I0, fase_inicial=0.0):
    """
    Generates a complex Gaussian TEM00 field E(x,y) ready to use in BPM.

    Parameters:
        X, Y (ndarray): 2D meshgrids with spatial coordinates in meters.
        w0 (float): beam waist radius in meters.
        I0 (float): peak intensity in W/m².
        fase_inicial (float): optional global phase (in radians).

    Returns:
        complex ndarray: complex electric field E(x,y)
    """
    R2 = X**2 + Y**2
    Ex = np.sqrt(np.float32(I0)) * np.exp(np.float32(-R2 / w0**2))
    fase = np.exp(np.complex64(-1j * fase_inicial))
    return np.complex64(Ex * fase)


class fuente_microscopia_1:
    wavelength = np.float32(800e-9)  # m wavelength
    w0 = np.float32(3e-6)  # m beam waist 3um
    I_peak = np.float32(1e10)  # W/m**2
    NA = np.float32(0.1)  # NA of the lens


if __name__ == "__main__":
    # Beam parameters
    w0 = fuente_microscopia_1.w0  # Beam radius in meters (3 µm)
    I0 = fuente_microscopia_1.I_peak   # Peak intensity in W/m²

    # Create coordinate mesh
    L = 45e-6  # Domain size (15 µm)
    N = 128     # Number of points in each dimension
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)

    # Generate the TEM00 field
    campo = campo_tem00(X, Y, w0, I0)

    # Calculate the intensity (|E|²)
    intensidad = np.abs(campo)**2

    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot 2D surface
    surf = ax.plot_surface(X*1e6, Y*1e6, intensidad/I0, 
                          cmap=cm.viridis, linewidth=0, antialiased=True)

    # Configure labels and title
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_zlabel('Normalized Intensity')
    ax.set_title('TEM00 Mode - Intensity Profile')

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Normalized Intensity')

    # Show plot
    plt.tight_layout()
    plt.show()

    # Also create a 2D view (contour)
    plt.figure(figsize=(8, 6))
    plt.contourf(X*1e6, Y*1e6, intensidad/I0, 50, cmap='viridis')
    plt.colorbar(label='Normalized Intensity')
    plt.xlabel('X (µm)')
    plt.ylabel('Y (µm)')
    plt.title('TEM00 Mode - Top View')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
