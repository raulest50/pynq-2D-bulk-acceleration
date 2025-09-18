import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# Global style for better readability (does not affect computation time)
mpl.rcParams.update({
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

from deep_tissue_imaging.elementos.lasers import fuente_microscopia_1 as laser, campo_tem00
from deep_tissue_imaging.elementos.plotting import plot_field_intensity, plot_field_intensity_history
from deep_tissue_imaging.elementos.tejidos import cerebro_emb_pez_cebra as tejido
import deep_tissue_imaging.propagators.propagation as prop
import deep_tissue_imaging.elementos.domain as Domain
from benchmark.phase_mask_manager import PhaseMaskManager
from benchmark.medir_psf_params import medir_psf_params
from benchmark.system_info import print_system_info
from scipy.ndimage import gaussian_filter


def save_beam_profile(phi, X, Y, title, filename, fs_title=16, fs_labels=12, fs_ticks=10):
    """
    Saves an intensity map |phi|^2 with improved style in PNG format.

    Parameters:
        phi (ndarray): Complex field
        X, Y (ndarray): Spatial meshgrids (in meters)
        title (str): Plot title
        filename (str): Output filename
        fs_title, fs_labels, fs_ticks (int): Font sizes
    """
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


def save_psf_analysis(phi, X, Y, phi_history, z_positions, title, filename, mask_manager=None):
    """
    Saves a comprehensive PSF analysis plot in PNG format with phase mask visualization.

    Parameters:
        phi (ndarray): Complex field at focal plane
        X, Y (ndarray): Spatial meshgrids (in meters)
        phi_history (ndarray): History of field propagation
        z_positions (ndarray): Z positions in meters
        title (str): Plot title
        filename (str): Output filename
        mask_manager (CustomPhaseMaskManager, optional): Phase mask manager for visualization
    """
    # Ensure we're working with intensity
    if np.iscomplexobj(phi):
        psf = np.abs(phi)**2
    else:
        psf = phi.copy()

    # Calculate PSF parameters
    psf_params = medir_psf_params(phi, X, Y, phi_history, z_positions, plot=False)

    # Create figure
    fig = plt.figure(figsize=(12, 10))

    # Main title
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Add FWHM as subtitle
    plt.figtext(0.5, 0.94, f"FWHM={psf_params['fwhm_lateral']*1e6:.2f} μm", 
               ha="center", fontsize=14)

    # Plot the PSF
    ax1 = plt.subplot(2, 2, 1)
    im = ax1.imshow(psf, extent=[X.min()*1e6, X.max()*1e6, Y.min()*1e6, Y.max()*1e6])
    plt.colorbar(im, ax=ax1, label='Intensity (W/m²)')
    ax1.set_title('PSF Intensity')
    ax1.set_xlabel('X (μm)')
    ax1.set_ylabel('Y (μm)')

    # Plot horizontal and vertical profiles
    center_y, center_x = np.unravel_index(np.argmax(psf), psf.shape)

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(X[center_y, :]*1e6, psf[center_y, :] / np.max(psf))
    ax2.axhline(0.5, color='r', linestyle='--', label='Half Maximum')
    ax2.set_title('Horizontal Profile')
    ax2.set_xlabel('X (μm)')
    ax2.set_ylabel('Normalized Intensity')
    ax2.grid(True)
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(Y[:, center_x]*1e6, psf[:, center_x] / np.max(psf))
    ax3.axhline(0.5, color='r', linestyle='--', label='Half Maximum')
    ax3.set_title('Vertical Profile')
    ax3.set_xlabel('Y (μm)')
    ax3.set_ylabel('Normalized Intensity')
    ax3.grid(True)
    ax3.legend()

    # Plot phase mask if available, otherwise plot axial profile
    if mask_manager is not None and 1 in mask_manager.masks:
        ax4 = plt.subplot(2, 2, 4)
        phase_mask = mask_manager.masks[1]  # Get the first phase mask

        # Plot the phase mask
        im = ax4.imshow(phase_mask, extent=[X.min()*1e6, X.max()*1e6, Y.min()*1e6, Y.max()*1e6],
                       cmap='coolwarm', vmin=-np.pi/2, vmax=np.pi/2)
        plt.colorbar(im, ax=ax4, label='Phase (rad)')
        ax4.set_title('Random Phase Mask (Tissue Heterogeneity)')
        ax4.set_xlabel('X (μm)')
        ax4.set_ylabel('Y (μm)')
    elif phi_history is not None and z_positions is not None:
        # Original axial profile code as fallback
        ax4 = plt.subplot(2, 2, 4)
        max_intensities = np.array([np.max(np.abs(psf_z)**2) for psf_z in phi_history])
        ax4.plot(z_positions*1e6, max_intensities / np.max(max_intensities))
        ax4.axhline(0.5, color='r', linestyle='--', label='Half Maximum')
        ax4.set_title('Axial Profile')
        ax4.set_xlabel('Z (μm)')
        ax4.set_ylabel('Normalized Intensity')
        ax4.grid(True)
        ax4.legend()

    # No text box at the bottom

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjusted for no text at bottom
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return psf_params


class CustomPhaseMaskManager(PhaseMaskManager):
    """
    Extension of PhaseMaskManager that allows generating masks with different random seeds.
    """

    def __init__(self, save_dir="./phase_masks", random_seed=None):
        """
        Initialize the CustomPhaseMaskManager.

        Parameters:
            save_dir (str): Directory to save phase masks for persistence
            random_seed (int, optional): Seed for random number generation
        """
        super().__init__(save_dir)
        self.random_seed = random_seed

    def generate_mask(self, shape, X, Y, desviacion_fase, correlacion_m, mask_index):
        """
        Generate a new phase mask with the given parameters and custom seed.

        Parameters:
            shape (tuple): Shape of the mask (Ny, Nx)
            X, Y (ndarray): Spatial meshgrids (in meters)
            desviacion_fase (float): Phase standard deviation in radians
            correlacion_m (float): Spatial correlation length in meters
            mask_index (int): Index of the mask (1, 2, or 3)

        Returns:
            ndarray: The generated phase mask (theta, not the complex exponential)
        """
        # Set a seed based on the mask index and the custom seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed + mask_index)  # Use custom seed + mask index
        else:
            np.random.seed(42 + mask_index)  # Use default seed + mask index

        # Calculate dx and dy from the meshgrids (in meters)
        dx = np.float32(np.abs(X[0, 1] - X[0, 0]))  # meters
        dy = np.float32(np.abs(Y[1, 0] - Y[0, 0]))  # meters

        # Correlation length in pixels
        sigma_x = np.float32(correlacion_m / dx)
        sigma_y = np.float32(correlacion_m / dy)

        # Generate Gaussian noise with desired standard deviation
        ruido = np.random.normal(loc=0.0, scale=desviacion_fase, size=shape).astype(np.float32)

        # Smooth to mimic structural fluctuation
        theta = gaussian_filter(ruido, sigma=(sigma_y, sigma_x), mode='reflect')

        # For on-the-fly masks, we don't save to disk
        if self.random_seed is not None:
            # Store in memory only
            self.masks[mask_index] = theta
            return theta

        # Save the mask (only for default masks)
        mask_file = self.get_mask_filename(mask_index)
        np.save(mask_file, theta)

        # Update metadata
        self.metadata[mask_index] = {
            'shape': shape,
            'desviacion_fase': desviacion_fase,
            'correlacion_m': correlacion_m,
            'created': np.datetime64('now')
        }
        self._save_metadata()

        return theta


def run_simulation(seed):
    """
    Runs a complete simulation with a specific random seed.

    Parameters:
        seed (int): Seed for random mask generation

    Returns:
        dict: Simulation results
    """
    # Domain parameters (same as in the table)
    Lz = np.float32(361e-6)  # 361um
    Nz = 361
    dz = np.float32(Lz / Nz)  # 1um

    Lx, Ly = np.float32(45e-6), np.float32(45e-6)  # 45um x 45um
    Nx, Ny = 64, 64
    dx = np.float32(Lx / Nx)  # 0.35um
    dy = np.float32(Ly / Ny)  # 0.35um

    x = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
    y = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    # Physical parameters from the table
    wavelength = np.float32(800e-9)  # 800 nm
    w0 = np.float32(3e-6)  # 3 μm
    I0 = np.float32(1e10)  # 10^10 W/m^2

    n0 = np.float32(1.36)
    Dn = np.float32(0.015)
    l_s = np.float32(120e-6)  # 120 μm
    alpha = np.float32(0.3)  # 0.3 mm^-1
    beta = np.float32(1e-11)  # 10^-11 m/W

    # Calculate derived parameters
    k0 = np.float32(2*np.pi / wavelength)
    k = np.float32(k0 * n0)
    sigma_phi = np.float32(k * Dn * l_s)
    sigma_x = np.float32(5e-6)  # 5 μm

    # Create domain
    domain = Domain.Domain(X, Y, Nx, Ny, Nz, dx, dy, dz, np.float32(1e-12), k0, k, sigma_phi, sigma_x)

    # Create initial field
    phi0 = campo_tem00(X, Y, w0, I0)

    # Create phase mask manager with custom seed
    mask_manager = CustomPhaseMaskManager(save_dir=f"./phase_masks_sim_{seed}", random_seed=seed)

    # Run propagation
    print(f"Running simulation with seed {seed}...")
    start_time = time.time()
    phi_history = prop.full_propagation_within_tissue(phi0, tejido, domain, mask_manager=mask_manager)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"  Execution time: {execution_time:.2f} seconds")

    # Calculate PSF parameters
    z_positions = np.linspace(0, Lz, Nz+1)
    focal_plane = phi_history[-1]
    psf_params = medir_psf_params(focal_plane, X, Y, phi_history, z_positions, plot=False)

    # Return results
    return {
        "seed": seed,
        "phi_history": phi_history,
        "X": X,
        "Y": Y,
        "execution_time": execution_time,
        "psf_params": psf_params,
        "mask_manager": mask_manager  # Include the mask_manager in the results
    }




# Print system information
print_system_info(save_json=True)

# Create output directory
output_dir = "output_beam_profiles"
os.makedirs(output_dir, exist_ok=True)

# Domain parameters (for initial beam)
Lz = np.float32(361e-6)  # 361um
Nz = 361
Lx, Ly = np.float32(45e-6), np.float32(45e-6)  # 45um x 45um
Nx, Ny = 64, 64
x = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
y = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
X, Y = np.meshgrid(x, y)

# Create initial field (same for all simulations)
phi0 = campo_tem00(X, Y, laser.w0, laser.I_peak)

# Save initial beam profile once
save_beam_profile(phi0, X, Y, 'Initial Beam Profile', f'{output_dir}/initial_beam.png')
print(f"Initial beam profile saved to {output_dir}/initial_beam.png")

# Run 4 simulations with different random seeds
seeds = [1000, 2000, 3000, 4000]  # Different seeds for each simulation
simulation_results = []

for i, seed in enumerate(seeds):
    # Run simulation
    result = run_simulation(seed)
    simulation_results.append(result)

    # Get final beam profile
    final_beam = result["phi_history"][-1]

    # Save final beam profile with PSF analysis
    z_positions = np.linspace(0, Lz, Nz+1)
    # Use the mask_manager from the simulation results
    psf_params = save_psf_analysis(
        final_beam, 
        result["X"], 
        result["Y"], 
        result["phi_history"], 
        z_positions,
        f"Output Beam Profile (Seed: {seed})",
        f"{output_dir}/output_beam_{i+1}.png",
        result["mask_manager"]  # Pass the mask manager from the results
    )

    print(f"Output beam profile {i+1} saved to {output_dir}/output_beam_{i+1}.png")
    print(f"  FWHM Lateral: {psf_params['fwhm_lateral']*1e6:.2f} μm")
    print(f"  FWHM Axial: {psf_params['fwhm_axial']*1e6:.2f} μm")
    print(f"  80% Energy Radius: {psf_params['radio_80']*1e6:.2f} μm")
    print(f"  Max Sidelobe: {psf_params['sidelobes']['max_sidelobe_ratio']*100:.1f}% of peak")
    print()

print("\nAll simulations completed and visualizations generated.")
