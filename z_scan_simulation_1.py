from zscan.zscan_propagators.info_methods import plot_beam_propagation, \
    assess_intensity_zscan, compute_transmitance_physical

from zscan.lasers.lasers import gaussian_beam_profile_physical, femto, print_laser_info, compute_E0
from zscan import materiales as materials
from zscan.materiales.domain import Domain
from zscan.materiales.sample import Sample
from zscan.zscan_propagators.propagations import full_propagation_without_sample
from zscan.lentes.lente_ideal import get_mascara_fase

import time
import colorama
from colorama import Fore, Style

import numpy as np

# Inicializar colorama (necesario en Windows)
colorama.init()

# 1) Imprime información del láser
print_laser_info(femto)

# Calcula E0 para usar en el resto del código
E0 = compute_E0(femto.P_avg, femto.f_rep, femto.tau, femto.w0)


Nx = 256                   # Resolución en x
Ny = 256                   # Resolución en y

Lx = 2e-3
Ly = 2e-3

# Generar haz gaussiano
Ex, x, y, X, Y = gaussian_beam_profile_physical(femto.w0, E0, Nx, Ny, Lx, Ly)

Lx=x[-1]
Ly=y[-1]

# Parametros Fisicos Modelo BPM
n_air = materials.aire.n0
alpha_air = materials.aire.alpha
k_air = 2 * 3.141592653589793 / femto.wavelength * n_air

# Parametros simulacion z-scan
Nz = 1200 # Numero de pasos en z
Lz = 0.6 # 60 centimetros
z = np.linspace(0, Lz, Nz)

# Parametros de la muestra
Nsz = 10 # sample thickness expressed as integer number of z steps
dz = z[1]-z[0]

# Crear instancia de Domain
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

# Imprimir información del dominio
domain.print_domain_info(Lx, Ly, femto.wavelength, n_air)

# Graficar el perfil de intensidad
# plot_beam_profile(Ex, x, y)


# Material de la muestra
material_name = "CS2"  # Disulfuro de carbono


sample_movs_step_size = 30
stops = np.arange(10, Nz, sample_movs_step_size)

# Obtener el objeto de material correspondiente
material_obj = getattr(materials, material_name)

# Crear instancia de Sample
sample = Sample(
    material=material_name,
    material_obj=material_obj,
    thickness=Nsz * dz,
    thickness_units=Nsz,
    stops=stops,
    wavelength=femto.wavelength
)

# Imprimir información de la muestra
sample.print_sample_info()
sample.print_stops_info()

# Evaluar intensidad para Z-scan
I_peak = assess_intensity_zscan(Ex, sample)

# Mensaje con formato mejorado según el umbral
if I_peak >= 1e11:
    print(f"\n{Fore.CYAN}{Style.BRIGHT}💡 EVALUACIÓN DE INTENSIDAD {Style.RESET_ALL}")
    print(f"{Fore.YELLOW}I_peak: {Fore.WHITE}{I_peak:.2e} W/m² {Fore.GREEN}→ intensidad suficiente para Z-scan cerrado (n2 ≃ {sample.n2:.2e}){Style.RESET_ALL}")
else:
    print(f"\n{Fore.CYAN}{Style.BRIGHT}💡 EVALUACIÓN DE INTENSIDAD {Style.RESET_ALL}")
    print(f"{Fore.YELLOW}I_peak: {Fore.WHITE}{I_peak:.2e} W/m² {Fore.RED}→ intensidad insuficiente; considerar enfoque más fuerte o pulso más corto{Style.RESET_ALL}")

# Parámetros necesarios
w_min = 50e-6  # Cintura mínima 300mm
foco = 240e-3  # Distancia focal de la lente 150mm

# Instanciar la lente
#mi_lente = Lente(w_min, femto.wavelength, foco)
#phase_mask = mi_lente.get_mascara_fase(X, Y)

#Phi0 = Ex

phase_mask = get_mascara_fase(f=foco, λ=femto.wavelength, n=1.4, D=0, X=X, Y=Y)

Phi0 = Ex * phase_mask

#plot_beam_profile(Phi0, x, y)

# T = z_scan(Phi0, sample, domain)
# print(f" T: {T}")
# plot_transmitance(T, z[sample.stops])

start_time = time.time()
#phi, phi_history = full_propagation_with_sample_debug(Phi0, sample, 200, domain)
phi, phi_history = full_propagation_without_sample(Phi0, domain)
end_time = time.time()
execution_time = (end_time-start_time)*1000

# Tiempo de ejecución con emoji
print(f"\n{Fore.GREEN}{Style.BRIGHT}⏱️ Tiempo de ejecución: {execution_time:.2f} ms{Style.RESET_ALL}")

# Dimensiones con emoji
print(f"{Fore.GREEN}{Style.BRIGHT}📊 Dimensiones phi_history: {phi_history.shape}{Style.RESET_ALL}")

print(compute_transmitance_physical(phi_history[0], phi_history[-1], domain.dx, domain.dy))

plot_beam_propagation(phi_history, x, y, dz)
