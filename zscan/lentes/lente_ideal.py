
import numpy as np


def get_mascara_fase(f:float, λ:float, n:float, D:float, X:np.ndarray, Y:np.ndarray,):
    k = (2 *np.pi) / λ  # Número de onda
    R = (X**2 + Y**2)  # Distancia radial desde el centro
    Phase = - ( k * D * n ) + ( R * k / (2*f) )
    z0 = np.pi * (110e-6)**2 /λ
    ap = np.exp( - k * R / (2 * z0))
    return np.exp(-1j * Phase)  # Retorna la máscara de fase como un array complejo
