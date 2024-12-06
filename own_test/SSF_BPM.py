import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift

# Physical parameters
wavelength = 800e-9  # 800 nm
k0 = 2 * np.pi / wavelength
n0 = 1.45  # reference index
dz = 1e-6  # propagation step

# Spatial grid
Lx = 50e-6
Ly = 50e-6
Nx = 256
Ny = 256
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
dx = x[1]-x[0]
dy = y[1]-y[0]
XX, YY = np.meshgrid(x, y)

# Input field: Gaussian beam
w0 = 10e-6
Psi = np.exp(-(XX**2+YY**2)/w0**2)

# Fourier coordinates
kx = 2*np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2*np.pi * np.fft.fftfreq(Ny, d=dy)
kX, kY = np.meshgrid(kx, ky)

# Define the refractive index distribution n(x,y)
# For simplicity, assume uniform medium n=n0
n = n0 * np.ones_like(XX)

# Split-step: For each step, we apply:
# Psi(z+dz) = exp(dz*N) * FFT^-1{exp(dz*D(kx,ky)) * FFT{Psi(z)}}
# Where D = i/(2*k0*n0)*Laplace and N = -i*k0*(n^2-n0^2)/(2*n0)

# For linear medium (n = n0), N=0. Let's just show diffraction step.

def propagate_ssfm(Psi, dz, kX, kY, k0, n0):
    # Diffraction operator in frequency domain
    D = (1j/(2*k0*n0))*( - (kX**2 + kY**2))
    # Apply half-step diffraction
    Psi_k = fft2(Psi)
    Psi_k = Psi_k * np.exp(D * dz)  # one full step of diffraction
    Psi_out = ifft2(Psi_k)
    return Psi_out

plt.imshow(np.abs(Psi)**2, extent=(x[0]*1e6,x[-1]*1e6,y[0]*1e6,y[-1]*1e6))
plt.xlabel('x (um)')
plt.ylabel('y (um)')
plt.title('Intensity at z=0m')
plt.colorbar()
plt.show()

# Propagate for a given number of steps
num_steps = 100
for step in range(num_steps):
    Psi = propagate_ssfm(Psi, dz, kX, kY, k0, n0)

# Plot final result
plt.imshow(np.abs(Psi)**2, extent=(x[0]*1e6,x[-1]*1e6,y[0]*1e6,y[-1]*1e6))
plt.xlabel('x (um)')
plt.ylabel('y (um)')
plt.title('Intensity at z={:.2e}m'.format(num_steps*dz))
plt.colorbar()
plt.show()
