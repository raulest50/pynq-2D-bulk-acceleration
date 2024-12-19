
import numpy as np
from matplotlib import pyplot as plt
from Functions import Gaussian_BEAM_Solution_Saleh, Sellmeir_Fcy_Response

Eo = 1
c = 3e8  # speed of light [m/s]
Lambda0 = 800e-9  # wavelength in meters
ko = 2 * np.pi / Lambda0
f0 = c / Lambda0  # frequency in Hz
n_omega0 = Sellmeir_Fcy_Response(c, f0)


# Gaussian Beam Shape characteristics
# Recalculating with the new beam waist
FWHM_beam = 70e-6
wo = FWHM_beam / 2.355  # beam waist Gaussian

N = 20

x = np.linspace(-400e-6, 400e-6, N)
y = np.linspace(-400e-6, 400e-6, N)

XX, YY = np.meshgrid(x, y)
Zo = 0
Lz = 6e-2  # max. Lz to simulate

waist_loc = 0.5 * Lz  # waist location
Eout, w_z =  Gaussian_BEAM_Solution_Saleh(Eo, wo, ko * n_omega0, XX, YY, Zo - waist_loc)

print(f"w_z : {w_z}")
print(f"len(XX) : {len(XX)}")
print(f"len(YY) : {len(YY)}")
print(f"len(Eg) : {len(Eout)}")
print(f"Eg[0] : \n {Eout[0]}")
print(f"len(Eg[0]) : {len(Eout[0])}")
print(f"XX : \n {XX}")

Iout = np.abs(Eout)**2

plt.figure()
plt.imshow(Iout, extent=(XX.min()*1e6, XX.max()*1e6, YY.min()*1e6, YY.max()*1e6), cmap='inferno', origin='lower')
plt.colorbar(label='Intensity [a.u.]')
plt.xlabel('x [μm]')
plt.ylabel('y [μm]')
plt.title('Gaussian Beam Intensity Profile at Z')
plt.show()
