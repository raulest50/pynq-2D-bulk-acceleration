import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

# Estilo global más legible (no afecta tiempos de cómputo)
mpl.rcParams.update({
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

from deep_tissue_imaging.elementos.lasers import fuente_microscopia_1 as laser, campo_tem00
from deep_tissue_imaging.elementos.plotting import plot_field_intensity, plot_field_intensity_history
from deep_tissue_imaging.elementos.tejidos import cerebro_emb_pez_cebra as tejido
import deep_tissue_imaging.propagators.propagation as prop
import deep_tissue_imaging.elementos.domain as Domain
from benchmark.phase_mask_manager import PhaseMaskManager
from benchmark.medir_psf_params import medir_psf_params
from benchmark.system_info import print_system_info


def save_beam_profile(phi, X, Y, title, filename,
                      fs_title=20, fs_labels=16, fs_ticks=14):
    """Saves an intensity map |phi|^2 with improved style in PNG format."""
    X_um, Y_um = X * 1e6, Y * 1e6
    intensity = np.abs(phi) ** 2

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(X_um, Y_um, intensity, cmap='viridis', shading='auto')
    cbar = plt.colorbar(im, ax=ax)

    # Bold and larger labels
    ax.set_title(title, fontsize=fs_title, fontweight='bold')
    ax.set_xlabel('X (μm)', fontsize=fs_labels, fontweight='bold')
    ax.set_ylabel('Y (μm)', fontsize=fs_labels, fontweight='bold')
    cbar.set_label('Intensity (W/m²)', fontsize=fs_labels, fontweight='bold')

    # More readable and bold ticks
    ax.tick_params(axis='both', labelsize=fs_ticks)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight('bold')
    cbar.ax.tick_params(labelsize=fs_ticks)
    for lab in cbar.ax.get_yticklabels():
        lab.set_fontweight('bold')

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


# Parametros de dominio

Lz = np.float32(361e-6)  # 200um
Nz = 361
dz = np.float32(Lz / Nz)  # 1um

Lx, Ly = np.float32(45e-6), np.float32(45e-6)  # 45um x 45um
Nx, Ny = 32, 32
dx = np.float32(Lx / Nx)  # 0.35um
dy = np.float32(Ly / Ny)  # 0.35um

x = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
y = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
X, Y = np.meshgrid(x, y)

k0 = np.float32(2*np.pi / laser.wavelength)
k = np.float32(k0 * tejido.n_0)
sigma_phi = np.float32(k * tejido.Dn * tejido.l_s)
# Typical value for brain tissue (5 μm)
sigma_x = np.float32(5e-6)

domain = Domain.Domain(X, Y, Nx, Ny, Nz, dx, dy, dz, np.float32(1e-12), k0, k, sigma_phi, sigma_x)

phi0 = campo_tem00(X, Y, laser.w0, laser.I_peak)
# plot_field_intensity(phi0, X, Y)

# Create a phase mask manager
mask_manager = PhaseMaskManager(save_dir="./phase_masks")

# Print system information (does not affect timing)
print_system_info(save_json=True)

# Measure execution time of full_propagation_within_tissue
start_time = time.time()
phi_history = prop.full_propagation_within_tissue(phi0, tejido, domain, mask_manager=mask_manager)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")

# Save initial (step 0) and final (step Nz) beam profiles
save_beam_profile(phi_history[0], X, Y, 'Initial Beam Profile (Step 0)', 'beam_initial.png')
save_beam_profile(phi_history[-1], X, Y, f'Final Beam Profile (Step {Nz})', 'beam_final.png')

# Measure PSF parameters
z_positions = np.linspace(0, Lz, Nz+1)
focal_plane = phi_history[-1]  # Last slice is the focal plane
psf_params = medir_psf_params(focal_plane, X, Y, phi_history, z_positions, plot=True)

# Plot field intensity history
plot_field_intensity_history(phi_history, X, Y)
