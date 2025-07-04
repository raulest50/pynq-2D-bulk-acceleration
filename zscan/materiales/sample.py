import numpy as np
from tabulate import tabulate
import colorama
from colorama import Fore, Style
import importlib

class Sample:
    """
    Clase que representa una muestra para simulaciones Z-scan.
    """
    def __init__(self, material, material_obj, thickness, thickness_units, stops, wavelength):
        """
        Constructor para la clase Sample.

        Parámetros:
        ----------
        material : str
            Nombre del material
        material_obj : object
            Objeto de la clase de material (CS2, aire, etc.)
        thickness : float
            Grosor de la muestra (m)
        thickness_units : int
            Grosor en unidades de pasos de simulación
        stops : numpy.ndarray
            Array con las posiciones de parada para Z-scan
        wavelength : float
            Longitud de onda (m)
        """
        self.material = material
        self.thickness = thickness
        self.thickness_units = thickness_units
        self.n0 = material_obj.n0
        self.n2 = material_obj.n2
        self.alpha = material_obj.alpha
        self.beta = material_obj.beta
        self.stops = stops
        # Calcular el número de onda para el material
        self.k = 2 * np.pi / wavelength * self.n0

    def print_sample_info(self):
        """
        Imprime información sobre la muestra en una tabla formateada.
        """
        print(f"\n{Fore.CYAN}{Style.BRIGHT}🔍 INFORMACIÓN DE LA MUESTRA {Style.RESET_ALL}")
        sample_data = [
            ["Material", f"{self.material}"],
            ["Grosor", f"{self.thickness*1000:.6f} mm"],
            ["Índice de refracción (n0)", f"{self.n0:.4f}"],
            ["Índice no lineal (n2)", f"{self.n2:.2e}"],
            ["Coef. absorción lineal (α)", f"{self.alpha:.2e} m⁻¹"],
            ["Coef. absorción de dos fotones (β)", f"{self.beta:.2e} m/W"]
        ]
        print(tabulate(sample_data, headers=["Parámetro", "Valor"], tablefmt="fancy_grid"))

    def print_stops_info(self):
        """
        Imprime información sobre los stops de la muestra en formato formateado.
        """
        print(f"\n{Fore.CYAN}{Style.BRIGHT}🔢 STOPS DE LA MUESTRA {Style.RESET_ALL}")
        # Dividir el array en grupos para mejor visualización
        stops_groups = [self.stops[i:i+5] for i in range(0, len(self.stops), 5)]
        for group in stops_groups:
            print(f"{Fore.YELLOW}{' '.join([f'{stop:4d}' for stop in group])}{Style.RESET_ALL}")
