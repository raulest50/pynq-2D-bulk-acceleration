import numpy as np
from tabulate import tabulate
import colorama
from colorama import Fore, Style

class Domain:
    """
    Clase que representa el dominio de simulación.
    """
    def __init__(self, Nx, Ny, Nz, dx, dy, dz, k_medium, alpha, eps=1e-12):
        """
        Constructor para la clase Domain.
        
        Parámetros:
        ----------
        Nx : int
            Resolución en dirección X
        Ny : int
            Resolución en dirección Y
        Nz : int
            Número de pasos en dirección Z
        dx : float
            Espaciado en dirección X (m)
        dy : float
            Espaciado en dirección Y (m)
        dz : float
            Espaciado en dirección Z (m)
        k_medium : float
            Número de onda en el medio (rad/m)
        alpha : float
            Coeficiente de absorción lineal (m^-1)
        eps : float, opcional
            Valor pequeño para evitar divisiones por cero, por defecto 1e-12
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.k_medium = k_medium
        self.alpha = alpha
        self.eps = eps
    
    def print_domain_info(self, Lx, Ly, wavelength, n_air):
        """
        Imprime información sobre el dominio en una tabla formateada.
        
        Parámetros:
        ----------
        Lx : float
            Ancho en dirección X (m)
        Ly : float
            Ancho en dirección Y (m)
        wavelength : float
            Longitud de onda (m)
        n_air : float
            Índice de refracción del aire
        """
        print(f"\n{Fore.CYAN}{Style.BRIGHT}📏 INFORMACIÓN DEL DOMINIO {Style.RESET_ALL}")
        domain_data = [
            ["Ancho en X", f"{Lx*1e6:.1f} μm"],
            ["Ancho en Y", f"{Ly*1e6:.1f} μm"],
            ["Resolución en X (Nx)", f"{self.Nx}"],
            ["Resolución en Y (Ny)", f"{self.Ny}"],
            ["Espaciado en X (dx)", f"{self.dx:.2e} m"],
            ["Espaciado en Y (dy)", f"{self.dy:.2e} m"],
            ["Longitud de onda", f"{wavelength*1e9:.1f} nm"],
            ["Índice de refracción (aire)", f"{n_air:.4f}"],
            ["Número de onda (aire)", f"{self.k_medium:.2e} rad/m"]
        ]
        print(tabulate(domain_data, headers=["Parámetro", "Valor"], tablefmt="fancy_grid"))