

import numpy as np
from Functions import Sellmeir_Fcy_Response, Create_Volumetric_Data, BPM_2D_Prop_NL_var_alongZ
import time
import cv2
import matplotlib.pyplot as plt

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



# Spatial Discretization
DX = 15e-6
DY = 15e-6
DZ = 100e-6

Lx = 700e-6
Ly = 700e-6
Lz = 6e-2  # max. Lz to simulate

NDX = int(np.floor(Lx / DX))  # discretization along X
NDY = int(np.floor(Ly / DY))  # discretization along Y
NDZ = int(np.ceil(Lz / DZ))  # discretization along Z

X = np.linspace(-Lx/2, Lx/2, NDX + 1)
Y = np.linspace(-Ly/2, Ly/2, NDY + 1)

DX = X[1] - X[0]
DY = Y[1] - Y[0]

Z = np.linspace(0, Lz, NDZ)
DZ = Z[1] - Z[0]

# Print out the propagation length details
print('-----------Propagation Length------------------  ')
print(f'Lx > {Lx * 1e6:.2f} um')
print(f'Ly > {Ly * 1e6:.2f} um')
print(f'DX > {DX * 1e6:.2f} um')
print(f'DY > {DY * 1e6:.2f} um')
print('------------------  ')
print(f'Lz > {Lz * 1e2:.2f} cm')
print(f'DZ > {DZ * 1e2:.2f} cm')
print('------------------  ')
print(f'NDX > {NDX}')
print(f'NDY > {NDY}')
print(f'NDZ > {NDZ}')
print('------------------------------------------------ \n')

Xo = 0
Yo = 0
Zo = 0  # Reference point Z=0
XX, YY = np.meshgrid(X, Y)

Eo = 1
waist_loc = 0.5 * Lz  # waist location
SourceProf, _ = Gaussian_BEAM_Solution_Saleh(Eo, wo, ko * n_omega0, XX, YY, Zo - waist_loc)

# Calculating the optical power through an aperture S with no sample
S = 2 * wo
maskS = ((XX)**2 + (YY)**2) <= (S / 2)**2

radius_vect = XX[int((NDX + 1) / 2), :]
SourceX = SourceProf[int((NDX + 1) / 2), :]

# Analytical diffraction ONLY
RdAnaly = X
Eout_Analytic, _ = Gaussian_BEAM_Solution_Saleh1D(Eo, wo, ko * n_omega0, RdAnaly, Lz - waist_loc)

# Plotting the input and analytical output profiles
plt.figure()
plt.plot(RdAnaly * 1e6, np.abs(SourceX))
plt.plot(RdAnaly * 1e6, np.abs(Eout_Analytic))
plt.grid(True)
plt.gca().set_fontsize(14)
plt.legend(['input at z=0', 'output analytical'])
plt.xlabel('x [\mu m]')
plt.ylabel('E[p.u.]')
plt.show()

# Initial profile for Numeric propagation
PHI_m0 = np.transpose(SourceProf)

# Plotting the initial beam profile
plt.figure()
ax1 = plt.subplot(2, 2, (1, 3), projection='3d')
ax1.plot_surface(XX * 1e6, YY * 1e6, np.abs(PHI_m0.T))
ax1.set_xlabel('x [\mu m]')
ax1.set_ylabel('y [\mu m]')
ax1.set_zlabel('E(x,y) [p.u.]')
ax1.set_aspect('auto')
plt.gca().set_fontsize(14)

plt.subplot(222)
plt.contour(XX * 1e6, YY * 1e6, np.abs(PHI_m0.T))
plt.xlabel('x [\mu m]')
plt.ylabel('y [\mu m]')
plt.title('Initial Beam profile')
plt.grid(True)
plt.axis('square')

plt.subplot(224)
plt.plot(radius_vect * 1e6, np.abs(PHI_m0[int((NDX + 1) / 2), :]))
plt.plot(radius_vect * 1e6, 0.5 * np.ones_like(radius_vect))
plt.xlabel('r [\mu m]')
plt.ylabel('E_0 [p.u.]')
plt.axis('square')
plt.title('Initial Beam profile')
plt.grid(True)
plt.show()



# For each frequency, solve the 2D-BPM
# In this case, there is only the central frequency
AW0 = E0_Amplitude
Freq = np.array([f0])  # Freq is defined as an array with the central frequency
LambdaVec = np.array([Lambda0])
PHI_m_frq = np.zeros((NDX + 1, NDY + 1, len(Freq)), dtype=complex)

# Checking Frequency and Sellmeir Equation
lambdaV = np.zeros(len(Freq))
n_omega_sellmeir = np.zeros(len(Freq))

for freq_i in range(len(Freq)):
    # Calculating the corresponding linear refractive index
    fq = Freq[freq_i]
    lambdaV[freq_i] = c / fq
    n_omega_sellmeir[freq_i] = Sellmeir_Fcy_Response(c, fq)

# Finding the closest index to the central frequency
valDif = np.abs(Freq - f0)
index_f0 = np.argmin(valDif)
Npoints_Z_to_save = 15

LeffSample = 1e-3
Lsample_ini = 0.5e-2
Lpath_length = 5.5e-2
Lsample_end = Lsample_ini + Lpath_length
zSampleLocs = np.linspace(Lsample_ini, Lsample_end, 100)

# Create the video writer
video_filename = 'myVideoZscan6cm.avi'
frame_rate = 5
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define codec and create VideoWriter object
video_writer = cv2.VideoWriter(video_filename, fourcc, frame_rate, (640, 480))

# Now you can use video_writer to write frames to the video file
# Remember to release the video writer after writing all frames

# open(writerObj) in MATLAB is analogous to creating the video_writer object in Python.


# Start the timer
start_time = time.time()

# Initialize the output arrays
Tout = np.zeros(len(zSampleLocs))
PHI_out_Numeric = np.zeros((NDX + 1, NDY + 1, len(Freq)), dtype=complex)

# Loop over each sample location
for lzSample in range(len(zSampleLocs)):

    zSample = zSampleLocs[lzSample]
    Z_n_profile = np.zeros(len(Z))
    Z_n_profile[(Z >= (zSample - LeffSample / 2)) & (Z <= (zSample + LeffSample / 2))] = 1

    # Solve the propagation equation for each frequency
    for freq_i in range(len(Freq)):

        # Take the amplitude for each frequency
        AmpNL = AW0[freq_i]
        # Each amplitude has the same spatial profile
        PHI_m0_freq = AmpNL * PHI_m0
        fq = Freq[freq_i]
        Lambda_i = LambdaVec[freq_i]

        omega_f = 2 * np.pi * fq
        print(f'Lambda_i > {Lambda_i * 1e9:.2f} [nm]')

        # Calculate the corresponding linear refractive index
        n_air = 1
        n2_air = 0

        n_omega = Sellmeir_Fcy_Response(c, fq)
        k = omega_f / c
        n_omega_sellmeir[freq_i] = n_omega

        nalongZ = n_air + (n_omega - n_air) * Z_n_profile
        n2alongZ = n2_air + n2 * Z_n_profile

        if index_f0 != freq_i:
            # Saving the propagation is not required
            PHI_m = BPM_2D_Prop_NL_var_alongZ(PHI_m0_freq, k, NDX, NDY, NDZ, DX, DY, DZ, 0, nalongZ, n2alongZ)
        else:
            # Saving the propagation at this frequency is required
            PHI_m, PHI_m_alongZ, z_to_save = BPM_2D_Prop_NL_var_alongZ(PHI_m0_freq, k, NDX, NDY, NDZ, DX, DY, DZ,
                                                                       Npoints_Z_to_save, nalongZ, n2alongZ)

        PHI_out_Numeric[:, :, freq_i] = PHI_m

    PHI_OUT = PHI_out_Numeric[:, :, 0]
    PHI_OUT_norm = PHI_OUT / E0_Amplitude

    # Calculate the optical power through an aperture S
    Tout[lzSample] = np.sum(np.abs(PHI_OUT[maskS]) ** 2) * np.pi * (S / 2) ** 2

    normalized = 1
    VolData, XX, YY, zLL = Create_Volumetric_Data(PHI_m_alongZ, XX, YY, z_to_save, Npoints_Z_to_save, normalized)

    # Slice Interpolation
    XX3, YY3, ZZ3 = np.meshgrid(X, Y, z_to_save)
    xslice = 0
    yslice = 0
    zslice = []

    fig = plt.figure(1245, figsize=(12, 8))

    plt.subplot(2, 3, (1, 4))
    plt.plot(Z * 1e2, nalongZ)
    plt.xlabel('z [cm]')
    plt.ylabel('n')
    plt.gca().invert_xaxis()

    plt.subplot(2, 3, (2, 5))
    plt.contourf(XX3 * 1e6, YY3 * 1e6, VolData[:, :, 0], cmap='viridis')
    plt.xlabel('x [\mu m]')
    plt.ylabel('y [\mu m]')
    plt.zlabel('z [cm]')
    plt.title(f'z sample {zSample * 1e2:.2f} [cm]')

    plt.subplot(2, 3, 3)
    plt.contourf(XX * 1e6, YY * 1e6, np.abs(PHI_OUT_norm.T), cmap='viridis')
    plt.colorbar()
    plt.xlabel('x [\mu m]')
    plt.ylabel('y [\mu m]')
    plt.title(f'z sample {zSample * 1e2:.2f} [cm]')

    plt.subplot(2, 3, 6)
    plt.plot(zSampleLocs[:lzSample + 1], Tout[:lzSample + 1], 'o-')
    plt.xlabel('Zscan Sample locs [cm]')
    plt.ylabel('Transmittance[p.u]')
    plt.grid(True)

    fig.tight_layout()

    # Save frame to video
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    video_writer.write(frame)

    plt.close(fig)

# Stop the timer
end_time = time.time()
print(f'Time elapsed: {end_time - start_time:.2f} seconds')

# Release the video writer object
video_writer.release()

# Plot the transmittance vs. Z-scan sample locations
plt.figure()
plt.plot(zSampleLocs * 1e2, Tout)
plt.xlabel('Zscan Sample locs [cm]')
plt.ylabel('Transmittance[p.u]')
plt.grid(True)
plt.gca().set_fontsize(14)
plt.show()



