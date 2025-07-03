"""
Example script demonstrating how to use the GPU-accelerated implementations
of the Z-scan simulation code.

This script shows how to replace the CPU implementations with the GPU-accelerated
versions in your existing code.
"""

import numpy as np
import time
import colorama
from colorama import Fore, Style

# Initialize colorama (needed on Windows)
colorama.init()

# Import from GPU-accelerated modules instead of the original ones
from zscan_custom.propagations_gpu import full_propagation_without_sample
from zscan_custom.info_methods import plot_beam_profile, plot_beam_propagation, compute_transmitance_physical

# The rest of the imports remain the same
from lasers.lasers import gaussian_beam_profile_physical, femto, print_laser_info, compute_E0
import materiales.materials as materials
from materiales.domain import Domain
from materiales.sample import Sample
import lentes.lente_ideal

# Print laser information
print_laser_info(femto)

# Calculate E0 to use in the rest of the code
E0 = compute_E0(femto.P_avg, femto.f_rep, femto.tau, femto.w0)

# Simulation parameters
Nx = 256                   # Resolution in x
Ny = 256                   # Resolution in y
Lx = 2e-3
Ly = 2e-3

# Generate Gaussian beam
Ex, x, y, X, Y = gaussian_beam_profile_physical(femto.w0, E0, Nx, Ny, Lx, Ly)

Lx = x[-1]
Ly = y[-1]

# Physical parameters for BPM model
n_air = materials.aire.n0
alpha_air = materials.aire.alpha
k_air = 2 * np.pi / femto.wavelength * n_air

# Z-scan simulation parameters
Nz = 1200  # Number of steps in z
Lz = 0.6   # 60 centimeters
z = np.linspace(0, Lz, Nz)

# Sample parameters
Nsz = 10   # Sample thickness expressed as integer number of z steps
dz = z[1] - z[0]

# Create Domain instance
domain = Domain(
    Nx=Nx,
    Ny=Ny,
    Nz=Nz,
    dx=x[1] - x[0],
    dy=y[1] - y[0],
    dz=dz,
    k_medium=k_air,
    alpha=alpha_air,
    eps=1e-12
)

# Print domain information
domain.print_domain_info(Lx, Ly, femto.wavelength, n_air)

# Sample material
material_name = "CS2"  # Carbon disulfide

sample_movs_step_size = 30
stops = np.arange(10, Nz, sample_movs_step_size)

# Get the corresponding material object
material_obj = getattr(materials, material_name)

# Create Sample instance
sample = Sample(
    material=material_name,
    material_obj=material_obj,
    thickness=Nsz * dz,
    thickness_units=Nsz,
    stops=stops,
    wavelength=femto.wavelength
)

# Print sample information
sample.print_sample_info()
sample.print_stops_info()

# Lens parameters
foco = 240e-3  # Focal length of the lens (240mm)

# Create phase mask for the lens
phase_mask = lentes.lente_ideal.get_mascara_fase(f=foco, λ=femto.wavelength, n=1.4, D=0, X=X, Y=Y)

# Apply the lens to the beam
Phi0 = Ex * phase_mask

# Run the simulation with GPU acceleration
print(f"\n{Fore.CYAN}{Style.BRIGHT}🚀 Running simulation with GPU acceleration...{Style.RESET_ALL}")
start_time = time.time()
phi, phi_history = full_propagation_without_sample(Phi0, domain)
end_time = time.time()
execution_time = (end_time - start_time) * 1000

# Print execution time with emoji
print(f"\n{Fore.GREEN}{Style.BRIGHT}⏱️ Execution time: {execution_time:.2f} ms{Style.RESET_ALL}")

# Print dimensions with emoji
print(f"{Fore.GREEN}{Style.BRIGHT}📊 Dimensions of phi_history: {phi_history.shape}{Style.RESET_ALL}")

# Compute transmittance
transmittance = compute_transmitance_physical(phi_history[0], phi_history[-1], domain.dx, domain.dy)
print(f"{Fore.GREEN}{Style.BRIGHT}📈 Transmittance: {transmittance}{Style.RESET_ALL}")

# Plot beam propagation
plot_beam_propagation(phi_history, x, y, dz)

"""
Note: To compare CPU vs GPU performance, you can run the same simulation with the CPU implementation:

from zscan_custom.propagations import full_propagation_without_sample as cpu_full_propagation_without_sample

# Run the simulation with CPU implementation
print(f"\n{Fore.CYAN}{Style.BRIGHT}🔄 Running simulation with CPU implementation...{Style.RESET_ALL}")
start_time = time.time()
phi_cpu, phi_history_cpu = cpu_full_propagation_without_sample(Phi0, domain)
end_time = time.time()
execution_time_cpu = (end_time - start_time) * 1000

print(f"\n{Fore.YELLOW}{Style.BRIGHT}⏱️ CPU Execution time: {execution_time_cpu:.2f} ms{Style.RESET_ALL}")
print(f"{Fore.GREEN}{Style.BRIGHT}🚀 Speedup: {execution_time_cpu / execution_time:.2f}x{Style.RESET_ALL}")
"""