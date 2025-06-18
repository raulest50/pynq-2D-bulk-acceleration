from helper_methods import gaussian_beam_profile
from zscan_custom.helper_methods import plot_beam_profile, apply_lens
import numpy as np
import materials
from types import SimpleNamespace as Namespace

from zscan_custom.operator_solvers import z_scan

# Parámetros del láser Carmel X-780
wavelength = 780e-9       # 780 nm
w0 = 0.625e-3              # Radio del haz en z = 0 (0.625 mm)
E0 = 1.0                   # Amplitud normalizada
Nx = 46                   # Resolución en x
Ny = 46                   # Resolución en y

# Generar haz gaussiano
Ex, x, y, X, Y = gaussian_beam_profile(wavelength, w0, E0, Nx, Ny)
Lx=x[-1]
Ly=y[-1]
print(f"ancho de dominio en x: {Lx*1e6} um \n ancho de dominio en y: {Ly*1e6} um")

# Graficar el perfil de intensidad
# plot_beam_profile(Ex, x, y)

# Parametros simulacion z-scan
Nz = 600
Lz = 0.6 # 60 centimetros
z = np.linspace(0, Lz, Nz)
n_air = 1.003
n_sample = materials.MATERIAL_PARAMS["CS2"]["n0"]
n2_sample = materials.MATERIAL_PARAMS["CS2"]["n2"]

# Parametros de la muestra
Nsz = 10
dz = z[1]-z[0]


print(f"grosor de la muestra: {Nsz*(dz)*1000} mm")
print(f"n0 muestra: {n_sample}, n2 muestra: {n2_sample}")

stops = np.arange(0, Nz, Nz // Nsz)
print(f"stops muestra (indices del vector z): {stops}")

# Parametros Fisicos Modelo BPM
k_air = 2 * 3.141592653589793 / wavelength * n_air
k_sample = 2 * 3.141592653589793 / wavelength * n_sample

domain = Namespace(
    Nx = Nx,
    Ny = Ny,
    Nz = Nz,
    dx = x[1] - x[0],
    dy = y[1] - y[0],
    dz = dz,
    k_medium = k_air,
    eps = 1e-12,
)

sample = Namespace(
    thickness = Nsz * dz,
    n0 = n_sample,
    n2 = n2_sample,
    stops = stops,
    k = k_sample
)

plot_beam_profile(Ex, x, y)
Phi0 = apply_lens(Ex, X, Y, domain.k_medium, 0.1)
plot_beam_profile(Phi0, x, y)

#T = z_scan(Ex, domain, sample)
