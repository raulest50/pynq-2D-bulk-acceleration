import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
wavelength = 800e-9
k0 = 2 * np.pi / wavelength
n0 = 1.45
dz = 1e-6

Lx = 50e-6
Ly = 50e-6
Nx = 128
Ny = 128
x = np.linspace(-Lx / 2, Lx / 2, Nx)
y = np.linspace(-Ly / 2, Ly / 2, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
XX, YY = np.meshgrid(x, y)

# Input field: Gaussian
w0 = 10e-6
Psi = np.exp(-(XX ** 2 + YY ** 2) / w0 ** 2)


# Operators for Crank-Nicolson
# We have D = i/(2*k0*n0)*∇_T^2.
# Discretize second derivative in x and y using central differences:
# ∂²/∂x² Psi_j = (Psi_{j+1} - 2*Psi_j + Psi_{j-1}) / dx²

# Create 1D second-derivative operators for x and y:
def second_deriv_operator(N, d):
    # Second derivative with Dirichlet boundary conditions (or TBC)
    diag_main = -2.0 * np.ones(N)
    diag_off = np.ones(N - 1)
    Lap = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(N, N))
    return Lap / (d ** 2)


Lx_op = second_deriv_operator(Nx, dx)
Ly_op = second_deriv_operator(Ny, dy)

# Crank-Nicolson split-step (ADI):
# Step half in x-direction:
# (I - (dz/2)*D_x)*Psi_{m+1/2} = (I + (dz/2)*D_x)*Psi_m
# where D_x = i/(2*k0*n0)*∂²/∂x²
# Similarly for y-direction.

Dx_factor = (dz / 2) * (1j / (2 * k0 * n0)) * Lx_op
Dy_factor = (dz / 2) * (1j / (2 * k0 * n0)) * Ly_op

Ix = np.eye(Nx, dtype=complex)
Iy = np.eye(Ny, dtype=complex)

# For ADI:
# 1) Solve in x-direction with y fixed
# For each row in y:
# (Ix - Dx_factor)*Psi_{m+1/2}(x,:) = (Ix + Dx_factor)*Psi_m(x,:)
# 2) Solve in y-direction with x fixed
# For each column in x:
# (Iy - Dy_factor)*Psi_{m+1}(y,:) = (Iy + Dy_factor)*Psi_{m+1/2}(y,:)

A_plus_x = (Ix + Dx_factor)
A_minus_x = (Ix - Dx_factor)
A_plus_y = (Iy + Dy_factor)
A_minus_y = (Iy - Dy_factor)

# Convert to sparse for efficiency (already done by construction)
# We'll do a few propagation steps:
num_steps = 50

for step in range(num_steps):
    # X-direction half step
    Psi_half = np.zeros_like(Psi, dtype=complex)
    for j in range(Ny):
        # Solve along x for each fixed y
        rhs = A_plus_x @ Psi[j, :]
        Psi_half[j, :] = spsolve(A_minus_x, rhs)
    # Y-direction half step
    Psi_new = np.zeros_like(Psi, dtype=complex)
    for i in range(Nx):
        # Solve along y for each fixed x
        rhs = A_plus_y @ Psi_half[:, i]
        Psi_new[:, i] = spsolve(A_minus_y, rhs)

    Psi = Psi_new

# Plot final field intensity
plt.imshow(np.abs(Psi) ** 2, extent=(x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6))
plt.xlabel('x (um)')
plt.ylabel('y (um)')
plt.title('Intensity at z={:.2e}m'.format(num_steps * dz))
plt.colorbar()
plt.show()
