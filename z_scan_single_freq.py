import numpy as np
import pandas as pd

from FunctionsSimplified import Sellmeir_Fcy_Response, BPM_2D_Prop_NL_var_alongZ,\
    Gaussian_BEAM_Solution_Saleh, Gaussian_BEAM_Solution_Saleh1D
import time

import matplotlib.pyplot as plt

# Physical constants
c = 3e8  # speed of light [m/s]
Mu_Zero = 4 * np.pi * 1e-7
Epsilon_Zero = 1 / (Mu_Zero * c**2)

# Central frequency parameters (only one frequency f0)
Lambda0 = 800e-9         # wavelength in meters
ko = 2 * np.pi / Lambda0
f0 = c / Lambda0         # frequency in Hz
omega_0 = 2 * np.pi * f0   # angular frequency

# Calculate the refractive index at f0 using Sellmeir equation
n_omega = Sellmeir_Fcy_Response(c, f0)

# Non-linear refractive index
n2 = 2.5e-20

# Print media properties
print('----------Media Properties at central Frequency ------------')
print(f'Lambda0  > {Lambda0 * 1e9:.2f} [nm]')
print(f'f0 > {f0 * 1e-12:.2f} [THz]')
print(f'n_omega > {n_omega:.5f}')
print(f'n2 > {n2 * 1e20:.2f} x10^(-20)')
print('----------Critical Power for Self Trapping------------')

PcriticalSelfTrap = (np.pi * (0.61**2) * Lambda0**2) / (8 * n_omega * n2)
print(f'Optical Critical Power Self-trap PcriticalSelfTrap > {PcriticalSelfTrap * 1e-9:.2f} GW')
print('------------------------------------------------  ')

# Laser Power characteristics
print('----------Laser Power Characteristics ------------')
waist_measured = 0.29  # in cm
waist_mts = waist_measured * 1e-2  # in meters

I0 = 87e9         # Optical peak intensity in W/cm^2
I0_mts = I0 * 1e4  # in W/m^2
print('----------Using Gaussian power definition for optical power ------------')
print(f'Optical_Peak_Intensity I0 > {I0 * 1e-9:.2f} GW/cm^2')

Optical_Power = (1/2) * I0 * np.pi * waist_measured**2
print(f'Optical_Power > {Optical_Power * 1e-9:.2f} GW')

E0_peak = np.sqrt(I0_mts)
print(f'E0_peak > {E0_peak:.2f}')

# Self-focusing lengths
Zsf = ((2 * n_omega * waist_mts**2) / Lambda0) * (1 / np.sqrt(Optical_Power / PcriticalSelfTrap))
Zsf2 = waist_measured * np.sqrt(n_omega / (2 * n2 * I0_mts))
print(f'Self-focusing length Zsf > {Zsf * 1e2:.2f} [cm]')
print(f'Self-focusing length Zsf2 > {Zsf2:.2f} [cm]')
print('------------------------------------------------  ')

print('----- Imposing the amplitude of the electric field-------')
print(f'E0_peak > {E0_peak:.2f}')
print('------------------------------------------------  ')

# Gaussian Beam Shape characteristics
FWHM_beam = 70e-6
wo = FWHM_beam / 2.355  # Gaussian beam waist

I0_peak = I0_mts  # Optical intensity remains the same with the new waist
Optical_Power = (1/2) * I0_peak * np.pi * wo**2
E0_Amplitude = np.sqrt(I0_peak)

print(f'Optical_Power > {Optical_Power * 1e-6:.2f} MW')
print(f'Critical Power Self-trap PcriticalSelfTrap > {PcriticalSelfTrap * 1e-6:.2f} MW')
print(f'E0_peak > {E0_Amplitude:.2f}')
print(f'DeltaN = n2 * E0_Amplitude**2 > {n2 * E0_Amplitude**2:.2e}')

if Optical_Power >= PcriticalSelfTrap:
    ZRayleigh = np.pi * wo**2 / Lambda0
    Zsf_NewWo1 = abs(((2 * n_omega * wo**2) / Lambda0) * (1 / np.sqrt((Optical_Power / PcriticalSelfTrap) - 1)))
    Zsf_NewWo = wo * np.sqrt(n_omega / (2 * n2 * I0_mts))
    print(f'FWHM_beam > {FWHM_beam * 1e6:.2f} [um]')
    print(f'wo > {wo * 1e6:.2f} [um]')
    print(f'ZRayleigh > {ZRayleigh * 1e2:.2f} [cm]')
    print(f'Self focusing conditions > {Zsf_NewWo1:.2f}')
    print(f'Zsf_NewWo1 > {Zsf_NewWo1:.2f} [m]')
    print(f'Zsf_NewWo > {Zsf_NewWo:.2f} [m]')

# Spatial Discretization
DX = 15e-6
DY = 15e-6
DZ = 100e-6

# For faster test runs, using smaller Lx and Ly (originally 700e-6 scaled down by 8)
Lx = 700e-6/8
Ly = 700e-6/8
Lz = 6e-2  # total propagation distance

NDX = int(np.floor(Lx / DX))
NDY = int(np.floor(Ly / DY))
NDZ = int(np.ceil(Lz / DZ))

X = np.linspace(-Lx/2, Lx/2, NDX + 1)
Y = np.linspace(-Ly/2, Ly/2, NDY + 1)
DX = X[1] - X[0]
DY = Y[1] - Y[0]
Z = np.linspace(0, Lz, NDZ)
DZ = Z[1] - Z[0]

print('-----------Propagation Length------------------')
print(f'Lx > {Lx * 1e6:.2f} um')
print(f'Ly > {Ly * 1e6:.2f} um')
print(f'DX > {DX * 1e6:.2f} um')
print(f'DY > {DY * 1e6:.2f} um')
print('------------------')
print(f'Lz > {Lz * 1e2:.2f} cm')
print(f'DZ > {DZ * 1e2:.2f} cm')
print('------------------')
print(f'NDX > {NDX}')
print(f'NDY > {NDY}')
print(f'NDZ > {NDZ}')
print('------------------------------------------------')

XX, YY = np.meshgrid(X, Y)
Eo = 1
waist_loc = 0.5 * Lz
SourceProf, _ = Gaussian_BEAM_Solution_Saleh(Eo, wo, ko * n_omega, XX, YY, 0 - waist_loc)

# Define aperture mask for optical power calculation
S = 2 * wo
maskS = ((XX)**2 + (YY)**2) <= (S / 2)**2

radius_vect = XX[int((NDX + 1) / 2), :]
SourceX = SourceProf[int((NDX + 1) / 2), :]

# Analytical diffraction (1D)
RdAnaly = X
Eout_Analytic, _ = Gaussian_BEAM_Solution_Saleh1D(Eo, wo, ko * n_omega, RdAnaly, Lz - waist_loc)

plt.figure()
plt.plot(RdAnaly * 1e6, np.abs(SourceX))  # type: ignore[arg-type]
plt.plot(RdAnaly * 1e6, np.abs(Eout_Analytic))  # type: ignore[arg-type]
plt.grid(True)
plt.rcParams.update({'font.size': 14})
plt.legend(['input at z=0', 'output analytical'])
plt.xlabel('x [μm]')
plt.ylabel('E [p.u.]')
plt.show()

# Initial beam profile for numeric propagation
PHI_m0 = np.transpose(SourceProf)

plt.figure()
ax1 = plt.subplot(2, 2, (1, 3), projection='3d')  # type: ignore[arg-type]
ax1.plot_surface(XX * 1e6, YY * 1e6, np.abs(PHI_m0.T))  # type: ignore[arg-type]
ax1.set_xlabel('x [μm]')
ax1.set_ylabel('y [μm]')
ax1.set_zlabel('E(x,y) [p.u.]')
ax1.set_aspect('auto')
plt.rcParams.update({'font.size': 14})
plt.subplot(222)
plt.contour(XX * 1e6, YY * 1e6, np.abs(PHI_m0.T))  # type: ignore[arg-type]
plt.xlabel('x [μm]')
plt.ylabel('y [μm]')
plt.title('Initial Beam profile')
plt.grid(True)
plt.axis('square')
plt.subplot(224)
plt.plot(radius_vect * 1e6, np.abs(PHI_m0[int((NDX + 1) / 2), :]))
plt.plot(radius_vect * 1e6, 0.5 * np.ones_like(radius_vect))
plt.xlabel('r [μm]')
plt.ylabel('E0 [p.u.]')
plt.axis('square')
plt.title('Initial Beam profile')
plt.grid(True)
plt.show()

# Propagation parameters for f0 only
omega_f = 2 * np.pi * f0
k = omega_f / c


LeffSample = 1e-3
Lsample_ini = 0.5e-2
Lpath_length = 5.5e-2
Lsample_end = Lsample_ini + Lpath_length
zSampleLocs = np.linspace(Lsample_ini, Lsample_end, 100)

start_time = time.time()
Tout = np.zeros(len(zSampleLocs))

# Loop over each sample location (600 propagation steps overall)
for lzSample in range(len(zSampleLocs)):
    zSample = zSampleLocs[lzSample]
    Z_n_profile = np.zeros(len(Z))
    Z_n_profile[(Z >= (zSample - LeffSample/2)) & (Z <= (zSample + LeffSample/2))] = 1

    # Solve the propagation equation for f0 only (no frequency loop)
    AmpNL = E0_Amplitude
    PHI_m0_freq = AmpNL * PHI_m0

    n_air = 1
    n2_air = 0
    nalongZ = n_air + (n_omega - n_air) * Z_n_profile
    n2alongZ = n2_air + n2 * Z_n_profile

    print(f"lzSample > {lzSample} / {len(zSampleLocs)-1}")

    # Optional: visualize the refractive index profile
    # if(lzSample % 10 == 0):
    #     plt.plot(nalongZ)
    #     plt.show()
    #     plt.plot(n2alongZ)  # type: ignore[arg-type]
    #     plt.show()

    # Propagate the beam along z
    # PHI_m, PHI_m_alongZ, z_to_save = BPM_2D_Prop_NL_var_alongZ(
    #     PHI_m0_freq, k, NDX, NDY, NDZ, DX, DY, DZ,
    #     Npoints_Z_to_save, nalongZ, n2alongZ
    # )


    """
    THIS IMPLEMENTS A SINGLE Z-SCAN.
    Z-SCAN EXPERIMENT IS SIMULATED len(zSampleLocs) TIMES AT EACH zSampleLocs LOCATIONS FOR THE SAMPLE
    """
    PHI_m = BPM_2D_Prop_NL_var_alongZ(
        PHI_m0_freq, k,
        NDX, NDY, NDZ, DX, DY, DZ,
        nalongZ, n2alongZ
    )

    # Calculate the transmitted optical power through the aperture
    Tout[lzSample] = np.sum(np.abs(PHI_m[maskS])**2) * np.pi * (S/2)**2

    # Optionally, process volumetric data (for visualization or further analysis)
    # normalized = 1
    # VolData, XX, YY, zLL = Create_Volumetric_Data(PHI_m_alongZ, XX, YY, z_to_save, Npoints_Z_to_save, normalized)

end_time = time.time()
print(f'Time elapsed: {end_time - start_time:.2f} seconds')

plt.figure()
plt.plot(zSampleLocs * 1e2, Tout)  # type: ignore[arg-type]
plt.xlabel('Zscan Sample locs [cm]')
plt.ylabel('Transmittance [p.u.]')
plt.grid(True)
plt.rcParams.update({'font.size': 14})
plt.show()

df = pd.DataFrame({
    'z': zSampleLocs * 1e2,
    'T': Tout
})
df.to_csv('T_s.csv', index=False)