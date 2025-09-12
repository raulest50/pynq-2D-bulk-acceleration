import numpy as np
import time
import platform
import psutil

from deep_tissue_imaging.elementos.lasers import fuente_microscopia_1 as laser, campo_tem00
from deep_tissue_imaging.elementos.plotting import plot_field_intensity, plot_field_intensity_history
from deep_tissue_imaging.elementos.tejidos import cerebro_emb_pez_cebra as tejido
import deep_tissue_imaging.propagators.propagation as prop
import deep_tissue_imaging.elementos.domain as Domain
from benchmark.phase_mask_manager import PhaseMaskManager
from benchmark.medir_psf_params import medir_psf_params

# Parametros de dominio

Lz = np.float32(361e-6) # 361um - Total propagation distance
Nz = 361  # Number of BPM steps
dz = np.float32(Lz / Nz) # 1um - Step size

Lx, Ly = np.float32(45e-6), np.float32(45e-6) # 45um x 45um
Nx, Ny = 64, 64
dx = np.float32(Lx / Nx) # 0.35um
dy = np.float32(Ly / Ny) # 0.35um

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

# Print CPU information for comparison with FPGA
cpu_info = platform.processor()
cpu_cores = psutil.cpu_count(logical=False)
cpu_threads = psutil.cpu_count(logical=True)
memory = psutil.virtual_memory()
print(f"\nCPU Information:")
print(f"Processor: {cpu_info}")
print(f"Physical cores: {cpu_cores}, Logical cores: {cpu_threads}")
print(f"Memory: {memory.total / (1024**3):.2f} GB")
print(f"BPM Steps: {Nz} steps over {Lz*1e6:.1f} um")
print(f"Grid size: {Nx}x{Ny} pixels\n")

# Measure execution time of full_propagation_within_tissue using perf_counter for higher precision
print(f"Starting BPM propagation with {Nz} steps...")
start_time = time.perf_counter()
phi_history = prop.full_propagation_within_tissue(phi0, tejido, domain, mask_manager=mask_manager)
end_time = time.perf_counter()
execution_time = end_time - start_time
avg_step_time = execution_time / Nz
print(f"\nExecution time: {execution_time:.6f} seconds")
print(f"Average time per step: {avg_step_time*1000:.6f} ms")
print(f"Steps per second: {Nz/execution_time:.2f}")
print(f"Total BPM steps executed: {Nz}")

# Note: For additional performance comparisons, you can also use:
# 1. deep_tissue_imaging_gpu.py - GPU implementation using CUDA/CuPy
# 2. Your FPGA implementation on Kria KV260 (AMD Xilinx)

# Measure PSF parameters
z_positions = np.linspace(0, Lz, Nz+1)
focal_plane = phi_history[-1]  # Last slice is the focal plane
psf_params = medir_psf_params(focal_plane, X, Y, phi_history, z_positions, plot=True)

# Plot field intensity history
plot_field_intensity_history(phi_history, X, Y)
