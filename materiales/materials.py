# material_constants.py

# Parámetros de materiales para simulaciones BPM Z-scan
# Unidades en el SI:
#  - n0: índice de refracción (adimensional)
#  - n2: coeficiente de Kerr, en m^2/W
#  - alpha: coeficiente de absorción lineal, en m^-1
#  - beta: coeficiente de absorción de dos fotones (TPA), en m/W


class aire:
    n0 = 1.0003  # índice de refracción del aire a 800 nm (adimensional)
    n2 = 5e-23   # índice de Kerr típico para aire (m^2/W)
    alpha = 1.5e-5  # absorción lineal (m^-1)
    beta = 0.0   # TPA despreciable en aire (m/W)

class CS2:
    n0 = 1.6276  # índice lineal de CS2 (adimensional)
    n2 = 2.7e-19  # índice de Kerr de CS2 (m^2/W)
    alpha = 0.01  # absorción lineal (m^-1)
    beta = 0.0   # TPA despreciable a 800 nm (m/W)

class FusedSilica:
    n0 = 1.4585  # índice lineal de sílice fundida (adimensional)
    n2 = 2.74e-20  # índice de Kerr de sílice fundida (m^2/W)
    alpha = 1e-5  # absorción lineal (m^-1)
    beta = 1e-12  # coeficiente TPA típico (~0.1 cm/GW) (m/W)

class BK7:
    n0 = 1.5168  # índice lineal de BK7 (adimensional)
    n2 = 2.7e-20  # índice de Kerr de BK7 (m^2/W)
    alpha = 0.4   # absorción lineal (m^-1)
    beta = 1e-13  # TPA muy pequeño (m/W)

class As2S3:
    n0 = 2.4     # índice lineal de As2S3 (adimensional)
    n2 = 6.8e-18  # índice de Kerr de As2S3 (m^2/W)
    alpha = 0.01  # absorción lineal (m^-1)
    beta = 1e-11  # coeficiente TPA típico (m/W)

class Water:
    n0 = 1.333   # índice lineal del agua (adimensional)
    n2 = 8e-20   # índice de Kerr del agua (m^2/W)
    alpha = 0.0046  # absorción lineal (m^-1)
    beta = 5e-12  # coeficiente TPA (~0.005 cm/GW) (m/W)



