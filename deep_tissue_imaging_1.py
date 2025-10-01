import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import datetime
import os
from pathlib import Path

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
from benchmark.system_info import print_system_info, collect_system_info


def save_performance_data(execution_time, domain, psf_params, output_dir="./performance_data"):
    """
    Save comprehensive performance data including execution time, system info,
    simulation parameters, and PSF measurements.

    Parameters:
        execution_time (float): Execution time in seconds
        domain (Domain): Domain object containing simulation parameters
        psf_params (dict): PSF parameters measured by medir_psf_params
        output_dir (str): Directory to save performance data
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Get system information
    system_info = collect_system_info()

    # Collect domain parameters
    domain_params = {
        "Lx": float(domain.X.shape[1] * domain.dx),
        "Ly": float(domain.X.shape[0] * domain.dy),
        "Lz": float(domain.Nz * domain.dz),
        "Nx": int(domain.Nx),
        "Ny": int(domain.Ny),
        "Nz": int(domain.Nz),
        "dx": float(domain.dx),
        "dy": float(domain.dy),
        "dz": float(domain.dz),
        "dt": float(domain.eps),
        "k0": float(domain.k0),
        "k": float(domain.k),
        "sigma_phi": float(domain.sigma_phi),
        "sigma_x": float(domain.sigma_x)
    }

    # Collect laser parameters
    laser_params = {
        "wavelength": float(laser.wavelength),
        "w0": float(laser.w0),
        "I_peak": float(laser.I_peak)
    }

    # Collect tissue parameters
    tissue_params = {
        "n_0": float(tejido.n_0),
        "Dn": float(tejido.Dn),
        "l_s": float(tejido.l_s),
        "alpha": float(tejido.alpha) if hasattr(tejido, 'alpha') else None,
        "beta": float(tejido.beta) if hasattr(tejido, 'beta') else None,
        "n2": float(tejido.n2) if hasattr(tejido, 'n2') else None
    }

    # Format PSF parameters for JSON serialization
    formatted_psf_params = {}
    for key, value in psf_params.items():
        if key == 'energia_encerrada':
            formatted_psf_params[key] = {str(float(k)): float(v) for k, v in value.items()}
        elif key == 'sidelobes':
            formatted_psf_params[key] = {
                'max_sidelobe_ratio': float(value['max_sidelobe_ratio']),
                'horizontal_sidelobes': [[int(pos), float(level)] for pos, level in value['horizontal_sidelobes']],
                'vertical_sidelobes': [[int(pos), float(level)] for pos, level in value['vertical_sidelobes']]
            }
        elif value is not None:
            formatted_psf_params[key] = float(value)

    # Compile all data
    performance_data = {
        "timestamp": timestamp,
        "execution_time_seconds": execution_time,
        "system_info": system_info,
        "domain_parameters": domain_params,
        "laser_parameters": laser_params,
        "tissue_parameters": tissue_params,
        "psf_parameters": formatted_psf_params
    }

    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, indent=2, ensure_ascii=False)

    print(f"\nPerformance data saved to: {filepath}")
    return filepath


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
Nx, Ny = 64, 64
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

# Save performance data
save_performance_data(execution_time, domain, psf_params)

# Plot field intensity history
plot_field_intensity_history(phi_history, X, Y)
