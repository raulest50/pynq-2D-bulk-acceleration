
# from zscan_custom.helper_methods import plot_transmitance


from zscan_custom.helper_methods import plot_beam_profile, plot_beam_propagation, apply_lens, compute_E0, assess_intensity_zscan, gaussian_beam_profile_physical, apply_lens_abcd
import zscan_custom.materials as materials
from zscan_custom.propagations import z_scan, full_propagation_without_sample
import time
import colorama
from colorama import Fore, Back, Style
from tabulate import tabulate
import emoji


import numpy as np
from types import SimpleNamespace as Namespace

# Inicializar colorama (necesario en Windows)
colorama.init()

# Parámetros del láser Carmel X-780
wavelength = 780e-9    # m longitud onda
w0         = 0.625e-3  # m beam waist
P_avg      = 1.0       # W
f_rep      = 80e6      # Hz
tau        = 90e-15    # s

# 1) Calcula E0
E0 = compute_E0(P_avg, f_rep, tau, w0)

# Información del láser con emoji y color
print(f"\n{Fore.CYAN}{Style.BRIGHT}⚡ PARÁMETROS DEL LÁSER {Style.RESET_ALL}")
laser_data = [
    ["Campo eléctrico (E0)", f"{E0:.6e} V/m"],
    ["Beam waist (w0)", f"{w0*1e3:.3f} mm"],
    ["Potencia promedio", f"{P_avg:.1f} W"],
    ["Frecuencia de repetición", f"{f_rep/1e6:.0f} MHz"],
    ["Duración del pulso", f"{tau*1e15:.0f} fs"],
    ["Longitud de onda", f"{wavelength*1e9:.1f} nm"]
]
print(tabulate(laser_data, headers=["Parámetro", "Valor"], tablefmt="fancy_grid"))


Nx = 128                   # Resolución en x
Ny = 128                   # Resolución en y

Lx = 2e-3
Ly = 2e-3

# Generar haz gaussiano
Ex, x, y, X, Y = gaussian_beam_profile_physical(wavelength, w0, E0, Nx, Ny, Lx, Ly)

Lx=x[-1]
Ly=y[-1]

# Parametros Fisicos Modelo BPM
n_air = 1.003
k_air = 2 * 3.141592653589793 / wavelength * n_air

# Información del dominio como tabla
print(f"\n{Fore.CYAN}{Style.BRIGHT}📏 INFORMACIÓN DEL DOMINIO {Style.RESET_ALL}")
domain_data = [
    ["Ancho en X", f"{Lx*1e6:.1f} μm"],
    ["Ancho en Y", f"{Ly*1e6:.1f} μm"],
    ["Resolución en X (Nx)", f"{Nx}"],
    ["Resolución en Y (Ny)", f"{Ny}"],
    ["Espaciado en X (dx)", f"{x[1]-x[0]:.2e} m"],
    ["Espaciado en Y (dy)", f"{y[1]-y[0]:.2e} m"],
    ["Longitud de onda", f"{wavelength*1e9:.1f} nm"],
    ["Índice de refracción (aire)", f"{n_air:.4f}"],
    ["Número de onda (aire)", f"{k_air:.2e} rad/m"]
]
print(tabulate(domain_data, headers=["Parámetro", "Valor"], tablefmt="fancy_grid"))

# Graficar el perfil de intensidad
# plot_beam_profile(Ex, x, y)

# Parametros simulacion z-scan
Nz = 600
Lz = 0.6 # 60 centimetros
z = np.linspace(0, Lz, Nz)

# Material de la muestra
material_name = "CS2"  # Disulfuro de carbono
n_sample = materials.MATERIAL_PARAMS[material_name]["n0"]
n2_sample = materials.MATERIAL_PARAMS[material_name]["n2"]
alpha_sample = materials.MATERIAL_PARAMS[material_name]["alpha"]
beta_sample = materials.MATERIAL_PARAMS[material_name]["beta"]

# Parametros de la muestra
Nsz = 10 # sample thickness expressed as integer number of z steps
dz = z[1]-z[0]

# Información de la muestra como tabla
print(f"\n{Fore.CYAN}{Style.BRIGHT}🔍 INFORMACIÓN DE LA MUESTRA {Style.RESET_ALL}")
sample_data = [
    ["Material", f"{material_name}"],
    ["Grosor", f"{Nsz*(dz)*1000:.6f} mm"],
    ["Índice de refracción (n0)", f"{n_sample:.4f}"],
    ["Índice no lineal (n2)", f"{n2_sample:.2e}"],
    ["Coef. absorción lineal (α)", f"{alpha_sample:.2e} m⁻¹"],
    ["Coef. absorción de dos fotones (β)", f"{beta_sample:.2e} m/W"]
]
print(tabulate(sample_data, headers=["Parámetro", "Valor"], tablefmt="fancy_grid"))

sample_movs_step_size = 30
stops = np.arange(10, Nz, sample_movs_step_size)

# Stops de la muestra con formato más compacto
print(f"\n{Fore.CYAN}{Style.BRIGHT}🔢 STOPS DE LA MUESTRA {Style.RESET_ALL}")
# Dividir el array en grupos para mejor visualización
stops_groups = [stops[i:i+5] for i in range(0, len(stops), 5)]
for group in stops_groups:
    print(f"{Fore.YELLOW}{' '.join([f'{stop:4d}' for stop in group])}{Style.RESET_ALL}")

# Parámetros de la muestra - Número de onda
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
    material = material_name,
    thickness = Nsz * dz,
    thickness_units = Nsz,
    n0 = n_sample,
    n2 = n2_sample,
    alpha = alpha_sample,
    beta = beta_sample,
    stops = stops,
    k = k_sample
)

# Evaluar intensidad para Z-scan
I_peak = assess_intensity_zscan(Ex, sample)

# Mensaje con formato mejorado según el umbral
if I_peak >= 1e11:
    print(f"\n{Fore.CYAN}{Style.BRIGHT}💡 EVALUACIÓN DE INTENSIDAD {Style.RESET_ALL}")
    print(f"{Fore.YELLOW}I_peak: {Fore.WHITE}{I_peak:.2e} W/m² {Fore.GREEN}→ intensidad suficiente para Z-scan cerrado (n2 ≃ {n2_sample:.2e}){Style.RESET_ALL}")
else:
    print(f"\n{Fore.CYAN}{Style.BRIGHT}💡 EVALUACIÓN DE INTENSIDAD {Style.RESET_ALL}")
    print(f"{Fore.YELLOW}I_peak: {Fore.WHITE}{I_peak:.2e} W/m² {Fore.RED}→ intensidad insuficiente; considerar enfoque más fuerte o pulso más corto{Style.RESET_ALL}")

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

# T = z_scan(Phi0, sample, domain)
# print(f" T: {T}")
# plot_transmitance(T, z[sample.stops])

start_time = time.time()
phi, phi_history = full_propagation_without_sample(Phi0, domain)
end_time = time.time()
execution_time = (end_time-start_time)*1000

# Tiempo de ejecución con emoji
print(f"\n{Fore.GREEN}{Style.BRIGHT}⏱️ Tiempo de ejecución: {execution_time:.2f} ms{Style.RESET_ALL}")

# Dimensiones con emoji
print(f"{Fore.GREEN}{Style.BRIGHT}📊 Dimensiones phi_history: {phi_history.shape}{Style.RESET_ALL}")

plot_beam_propagation(phi_history, x, y, dz)
