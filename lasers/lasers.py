import numpy as np
import colorama
from colorama import Fore, Style
from tabulate import tabulate

# Inicializar colorama (necesario en Windows)
colorama.init()


class femto:
    # Parámetros del láser Carmel X-780
    wavelength = 780e-9  # m longitud onda
    w0 = 625e-6  # m beam waist 625um
    #w0 = 1e-3  # m beam waist
    P_avg = 0.3  # W
    f_rep = 80e6  # Hz
    tau = 90e-15  # s


def gaussian_beam_profile_physical(w0: float, E0: float, Nx: int, Ny: int, Lx: float = None, Ly: float = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera U(x,y) de un haz gaussiano TEM00 usando parámetros físicos.

    Parámetros:
    - wavelength: longitud de onda λ (m)
    - w0: cintura del haz (m)
    - E0: amplitud pico del campo (V/m)
    - Nx, Ny: puntos en x e y
    - Lx, Ly: tamaño de dominio (m). Por defecto 6·w0.
    """
    if Lx is None: Lx = 6 * w0
    if Ly is None: Ly = 6 * w0

    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    X, Y = np.meshgrid(x, y)
    r2 = X**2 + Y**2

    # Perfil gaussiano: E0·exp(–r²/w0²) (modo TEM00) :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
    Ex = E0 * np.exp(-r2 / w0**2).astype(np.complex128)
    return Ex, x, y, X, Y


def compute_E0(P_avg: float, f_rep: float, tau: float, w0: float) -> float:
    """
    Calcula la amplitud pico E0 del campo eléctrico (V/m) para un haz gaussiano.

    Parámetros:
    - P_avg: potencia promedio del láser (W)
    - f_rep: tasa de repetición (Hz)
    - tau: duración de pulso FWHM (s)
    - w0: radio de cintura del haz (m)

    Retorna:
    - E0: amplitud pico (V/m)
    """
    # Energía y potencia pico según Sheik-Bahae et al. :contentReference[oaicite:5]{index=5}
    E_pulse = P_avg / f_rep
    P_peak = E_pulse / tau

    # Intensidad pico en el foco de un haz gaussiano :contentReference[oaicite:6]{index=6}
    I0 = 2 * P_peak / (np.pi * w0 ** 2)

    # Relación I ↔ campo en vacío :contentReference[oaicite:7]{index=7}
    c = 3e8  # m/s
    eps0 = 8.854e-12  # F/m
    E0 = np.sqrt(2 * I0 / (c * eps0))

    return E0


def print_laser_info(laser_class):
    """
    Imprime información formateada de un láser con emojis y colores.

    Parámetros:
    - laser_class: Clase estática que contiene los parámetros del láser
                  (debe tener wavelength, w0, P_avg, f_rep, tau)
    """
    # Calcular E0 usando los parámetros del láser
    E0 = compute_E0(
        laser_class.P_avg, 
        laser_class.f_rep, 
        laser_class.tau, 
        laser_class.w0
    )

    # Información del láser con emoji y color
    print(f"\n{Fore.CYAN}{Style.BRIGHT}⚡ PARÁMETROS DEL LÁSER {Style.RESET_ALL}")
    laser_data = [
        ["Campo eléctrico (E0)", f"{E0:.6e} V/m"],
        ["Beam waist (w0)", f"{laser_class.w0*1e3:.3f} mm"],
        ["Potencia promedio", f"{laser_class.P_avg:.1f} W"],
        ["Frecuencia de repetición", f"{laser_class.f_rep/1e6:.0f} MHz"],
        ["Duración del pulso", f"{laser_class.tau*1e15:.0f} fs"],
        ["Longitud de onda", f"{laser_class.wavelength*1e9:.1f} nm"]
    ]
    print(tabulate(laser_data, headers=["Parámetro", "Valor"], tablefmt="fancy_grid"))


