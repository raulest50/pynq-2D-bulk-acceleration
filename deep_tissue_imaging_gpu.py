"""
Deep Tissue Imaging GPU Script

This script performs deep tissue imaging simulation using a hybrid CPU-GPU approach.
It uses CPU for ADI operations and GPU for halfsteps, minimizing CPU-GPU transfers.
"""

import numpy as np
import time
import matplotlib.pyplot as plt

# Check if CuPy is available
try:
    import cupy as cp
    print("CuPy is available. GPU acceleration is enabled.")

    # Get GPU information
    print(f"Number of GPU devices: {cp.cuda.runtime.getDeviceCount()}")
    for i in range(cp.cuda.runtime.getDeviceCount()):
        device_props = cp.cuda.runtime.getDeviceProperties(i)
        print(f"\nDevice {i}: {device_props['name'].decode()}")
        print(f"  Compute Capability: {device_props['major']}.{device_props['minor']}")
        print(f"  Total Memory: {device_props['totalGlobalMem'] / (1024**3):.2f} GB")
        print(f"  CUDA Cores: {device_props['multiProcessorCount']}")
except ImportError:
    print("CuPy is not available. GPU acceleration is disabled.")
    print("Please install CuPy to use GPU acceleration.")
    print("Exiting...")
    exit(1)

from deep_tissue_imaging.elementos.lasers import fuente_microscopia_1 as laser, campo_tem00
from deep_tissue_imaging.elementos.plotting import plot_field_intensity, plot_field_intensity_history
from deep_tissue_imaging.elementos.tejidos import cerebro_emb_pez_cebra as tejido
import deep_tissue_imaging.elementos.domain as Domain
import deep_tissue_imaging_gpu.propagators.propagation as prop_gpu
from benchmark.phase_mask_manager import PhaseMaskManager
from benchmark.medir_psf_params import medir_psf_params

# Domain parameters
print("Setting up domain parameters...")
Lz = np.float32(361e-6)  # 361um
Nz = 361
dz = np.float32(Lz / Nz)  # 1um

Lx, Ly = np.float32(45e-6), np.float32(45e-6)  # 45um x 45um
Nx, Ny = 256, 256
dx = np.float32(Lx / Nx)  # 0.176um
dy = np.float32(Ly / Ny)  # 0.176um

x = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
y = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
X, Y = np.meshgrid(x, y)

k0 = np.float32(2*np.pi / laser.wavelength)
k = np.float32(k0 * tejido.n_0)
sigma_phi = np.float32(k * tejido.Dn * tejido.l_s)
# Typical value for brain tissue (5 μm)
sigma_x = np.float32(5e-6)

domain = Domain.Domain(X, Y, Nx, Ny, Nz, dx, dy, dz, np.float32(1e-12), k0, k, sigma_phi, sigma_x)

# Create initial field
print("Creating initial field...")
phi0 = campo_tem00(X, Y, laser.w0, laser.I_peak)

# Create a phase mask manager
print("Creating phase mask manager...")
mask_manager = PhaseMaskManager(save_dir="./phase_masks")

# Run GPU simulation
print("\nRunning GPU simulation...")
# Option 1: Store full history (more memory intensive but allows axial FWHM calculation)
phi_history = prop_gpu.full_propagation_within_tissue_hybrid(phi0, tejido, domain, mask_manager=mask_manager, store_history=True)

# Option 2: Store only initial and final states (less memory intensive)
# phi_initial, phi_final = prop_gpu.full_propagation_within_tissue_hybrid(phi0, tejido, domain, mask_manager=mask_manager, store_history=False)

# Measure PSF parameters
print("\nMeasuring PSF parameters...")
z_positions = np.linspace(0, Lz, Nz+1)
focal_plane = phi_history[-1]  # Last slice is the focal plane
psf_params = medir_psf_params(focal_plane, X, Y, phi_history, z_positions, plot=True)

# Plot field intensity history
print("\nPlotting field intensity history...")
plot_field_intensity_history(phi_history, X, Y)

# Alternative for Option 2: Plot only initial and final states
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plot_field_intensity(phi_initial, X, Y, title="Initial Beam Profile")
# plt.subplot(1, 2, 2)
# plot_field_intensity(phi_final, X, Y, title="Final Beam Profile")
# plt.tight_layout()
# plt.show()
