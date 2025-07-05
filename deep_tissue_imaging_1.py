import numpy as np
import time

from deep_tissue_imaging.elementos.lasers import fuente_microscopia_1 as laser, campo_tem00
from deep_tissue_imaging.elementos.plotting import plot_field_intensity, plot_field_intensity_history
from deep_tissue_imaging.elementos.tejidos import cerebro_emb_pez_cebra as tejido
import deep_tissue_imaging.propagators.propagation as prop
import deep_tissue_imaging.elementos.domain as Domain

# Parametros de dominio

Lz = np.float32(361e-6) # 200um
Nz = 361
dz = np.float32(Lz / Nz) # 1um

Lx, Ly = np.float32(45e-6), np.float32(45e-6) # 45um x 45um
Nx, Ny = 256, 256
dx = np.float32(Lx / Nx) # 0.35um
dy = np.float32(Ly / Ny) # 0.35um

x = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
y = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
X, Y = np.meshgrid(x, y)

k0 = np.float32(2*np.pi / laser.wavelength)
k = np.float32(k0 * tejido.n_0)
sigma_phi = np.float32(k * tejido.Dn * tejido.l_s)
#sigma_x = 1600e-6 # 1 - 5 um
sigma_x = np.float32(1600000e-6*2)

domain = Domain.Domain(X, Y, Nx, Ny, Nz, dx, dy, dz, np.float32(1e-12), k0, k, sigma_phi, sigma_x)

phi0 = campo_tem00(X, Y, laser.w0, laser.I_peak)
# plot_field_intensity(phi0, X, Y)

# Measure execution time of full_step_within_tissue
start_time = time.time()
# phi1 = prop.full_step_within_tissue(phi0, tejido, domain)
phi_history = prop.full_propagation_within_tissue(phi0, tejido, domain)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time of full_step_within_tissue: {execution_time:.6f} seconds")

# plot_field_intensity(phi1, X, Y)
plot_field_intensity_history(phi_history, X, Y)
