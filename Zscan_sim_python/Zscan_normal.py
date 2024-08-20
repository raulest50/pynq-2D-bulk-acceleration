

import numpy as np
from Functions import Sellmeir_Fcy_Response

# Physical constants
c = 3e8  # speed of light [m/s]
Mu_Zero = 4 * np.pi * 1e-7
Epsilon_Zero = 1 / (Mu_Zero * c**2)

# Frequency of the pulse centered at
Lambda0 = 800e-9  # wavelength in meters
ko = 2 * np.pi / Lambda0

f0 = c / Lambda0  # frequency in Hz
omega_0 = 2 * np.pi * f0  # angular frequency

# Refractive index using Sellmeir equation
n_omega0 = Sellmeir_Fcy_Response(c, f0)

# Non-linear refractive index
n2 = 2.5e-20

# Printing results similar to MATLAB's `fprintf`
print('----------Media Properties at central Frequency ------------')
print(f'Lambda0  > {Lambda0 * 1e9:.2f} [nm]')
print(f'f0 > {f0 * 1e-12:.2f} [THz]')
print(f'n_omega0 > {n_omega0:.5f}')
print(f'n2 > {n2 * 1e20:.2f} x10^(-20)')
print('----------Critical Power for Self Trapping------------')

PcriticalSelfTrap = (np.pi * (0.61**2) * Lambda0**2) / (8 * n_omega0 * n2)
print(f'Optical Critical Power Self-trap PcriticalSelfTrap > {PcriticalSelfTrap * 1e-9:.2f} GW')
print('------------------------------------------------  ')

# Laser Power characteristics
print('----------Laser Power Characteristics ------------')
waist_measured = 0.29  # spot in cm
waist_mts = waist_measured * 1e-2  # spot in meters

# Optical Peak density
I0 = 87e9  # W/cm^2
I0_mts = I0 * 1e4  # W/m^2
print('----------Using Gaussian power definition for optical power ------------')
print(f'Optical_Peak_Intensity I0 > {I0 * 1e-9:.2f} GW/cm^2')

Optical_Power = (1/2) * I0 * np.pi * waist_measured**2
print(f'Optical_Power > {Optical_Power * 1e-9:.2f} GW')

E0_peak = np.sqrt(I0_mts)
print(f'E0_peak > {E0_peak:.2f}')

# Self-focusing lengths
Zsf = ((2 * n_omega0 * waist_mts**2) / Lambda0) * (1 / np.sqrt(Optical_Power / PcriticalSelfTrap))
Zsf2 = waist_measured * np.sqrt(n_omega0 / (2 * n2 * I0_mts))  # in cm

print(f'Self-focusing length Zsf > {Zsf * 1e2:.2f} [cm]')
print(f'Self-focusing length Zsf2 > {Zsf2:.2f} [cm]')
print('------------------------------------------------  ')

# Imposing the amplitude of the electric field
print('----- Imposing the amplitude of the electric field-------')
print(f'E0_peak > {E0_peak:.2f}')
print('------------------------------------------------  ')


# Gaussian Beam Shape characteristics
# Recalculating with the new beam waist
FWHM_beam = 70e-6
wo = FWHM_beam / 2.355  # beam waist Gaussian

# Recalculating the optical power with the new waist
I0_peak = I0_mts  # same optical Intensity with this new waist size
Optical_Power = (1/2) * I0_peak * np.pi * (wo)**2

# Amplitude (Using W/m^2) since we are using n2 in m^2/W
E0_Amplitude = np.sqrt(I0_peak)

# Print out the results
print(f'Optical_Power > {Optical_Power * 1e-6:.2f} MW')
print(f'Critical Power Self-trap PcriticalSelfTrap > {PcriticalSelfTrap * 1e-6:.2f} MW')
print(f'E0_peak > {E0_Amplitude:.2f}')
print(f'DeltaN = n2 * E0_Amplitude**2 > {n2 * E0_Amplitude**2:.2e}')

if Optical_Power >= PcriticalSelfTrap:
    # Self-focusing can occur
    ZRayleigh = np.pi * wo**2 / Lambda0
    Zsf_NewWo1 = abs(((2 * n_omega0 * wo**2) / Lambda0) * (1 / np.sqrt((Optical_Power / PcriticalSelfTrap) - 1)))
    Zsf_NewWo = wo * np.sqrt(n_omega0 / (2 * n2 * I0_mts))  # in cm, because the waist is in cm

    print(f'FWHM_beam > {FWHM_beam * 1e6:.2f} [um]')
    print(f'wo > {wo * 1e6:.2f} [um]')
    print(f'ZRayleigh > {ZRayleigh * 1e2:.2f} [cm]')
    print(f'Self focusing conditions > {Zsf_NewWo1:.2f}')
    print(f'Zsf_NewWo1 > {Zsf_NewWo1:.2f} [m]')
    print(f'Zsf_NewWo > {Zsf_NewWo:.2f} [m]')

