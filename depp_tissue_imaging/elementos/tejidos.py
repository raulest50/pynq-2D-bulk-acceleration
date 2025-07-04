import numpy as np
from scipy.ndimage import gaussian_filter


def generar_mascara_fase_aleatoria(shape, dx, desviacion_fase=0.3, correlacion_um=2.0, semilla=None):
    """
    Genera una máscara de fase aleatoria suave (en radianes) para simulación de dispersión en tejido biológico.

    Parámetros:
        shape (tuple): dimensiones (ny, nx) de la máscara (igual al tamaño del campo ψ).
        dx (float): tamaño del pixel o paso espacial en micras (µm).
        desviacion_fase (float): desviación estándar de la fase (en radianes).
        correlacion_um (float): longitud de correlación espacial en micras (determina suavidad).
        semilla (int, optional): valor de semilla para reproducibilidad.

    Retorna:
        np.ndarray: matriz 2D de fase aleatoria suavizada en radianes.
    """
    if semilla is not None:
        np.random.seed(semilla)

    # Paso espacial a metros
    dx_m = dx * 1e-6

    # Longitud de correlación en píxeles
    sigma_pix = correlacion_um / dx

    # Ruido gaussiano blanco (media 0, sigma = desviación de fase deseada)
    ruido = np.random.normal(loc=0.0, scale=desviacion_fase, size=shape)

    # Suavizado con filtro gaussiano (para simular anisotropía)
    fase = gaussian_filter(ruido, sigma=sigma_pix, mode='reflect')

    return fase


class cerebro_emb_pez_cebra:
    n_0 = 1.36 # 1.33 – 1.38	Agua, citoplasma, matriz extracelular
    Dn = 0.015 # ±0.01 – ±0.05	Heterogeneidad estructural (núcleo, orgánulos)
    l_s = 120e-6 # 	50 – 150 μm	Para medios como cerebro o piel
    alpha = 0.3 # mm*-1
    beta = 1e-11 # m/W
    n2 = 3e-20 # m²/W


