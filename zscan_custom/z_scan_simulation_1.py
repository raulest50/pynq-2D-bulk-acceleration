# Handle imports for both direct execution and module import
import sys
import os

# Try to import using absolute imports (when running as a module)
try:
    from zscan_custom.helper_methods import plot_beam_profile, plot_beam_propagation, apply_lens, compute_E0, assess_intensity_zscan, gaussian_beam_profile_physical, apply_lens_abcd
    import zscan_custom.materials as materials
    from zscan_custom.operator_solvers import z_scan, full_propagation_without_sample
# If that fails, try relative imports (when running directly)
except ModuleNotFoundError:
    # Add parent directory to path so Python can find the zscan_custom package
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from zscan_custom.helper_methods import plot_beam_profile, plot_beam_propagation, apply_lens, compute_E0, assess_intensity_zscan, gaussian_beam_profile_physical, apply_lens_abcd
    import zscan_custom.materials as materials
    from zscan_custom.operator_solvers import z_scan, full_propagation_without_sample



import numpy as np
from types import SimpleNamespace as Namespace

# Parámetros del láser Carmel X-780
wavelength = 780e-9    # m longitud onda
w0         = 0.625e-3  # m beam waist
P_avg      = 1.0       # W
f_rep      = 80e6      # Hz
tau        = 90e-15    # s

# 1) Calcula E0
E0 = compute_E0(P_avg, f_rep, tau, w0)
print(f"E0: {E0} V/m")


Nx = 64                   # Resolución en x
Ny = 64                   # Resolución en y

Lx = 2e-3
Ly = 2e-3

# Generar haz gaussiano
Ex, x, y, X, Y = gaussian_beam_profile_physical(wavelength, w0, E0, Nx, Ny, Lx, Ly)

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

assess_intensity_zscan(Ex, sample)

f = 0.15

# desired_gain = 1.5
# amp_gain = np.sqrt(desired_gain)    # ≈1.225

# Phi0 = Ex
# Phi0 = apply_lens_abcd(Ex, X, Y, wavelength, f)

Phi0 = apply_lens(
    Ex, X, Y,
    k       = domain.k_medium,
    f       = f,        # 25 cm
    strength= 0.5,         # focal efectiva = f/strength = 0.25 m
    amp_gain= 1.1,    # intensidad×1.5
    aperture=None          # o p.ej. 6*w0 para una lente de 6·waist de diámetro
)

#T = z_scan(Ex, domain, sample)

phi, phi_history = full_propagation_without_sample(Phi0, domain)

plot_beam_propagation(phi_history, x, y, dz)
print(f"dimensiones phi_history: {phi_history.shape}")
