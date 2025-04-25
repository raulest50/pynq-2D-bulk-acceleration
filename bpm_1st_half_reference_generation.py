import numpy as np
from FunctionsSimplified import BPM_First_half_TBC

# --- PARAMETERS (must match tb and C++ TB) ---
DX = 15e-6
DY = 15e-6
DZ = 100e-6
Lx = 700e-6
Ly = 700e-6
Lz = 6e-2

NDX = int(np.floor(Lx / DX))
NDY = int(np.floor(Ly / DY))

c = 3e8
Lambda0 = 800e-9
ko = 2 * np.pi / Lambda0
f0 = c / Lambda0
omega_f = 2 * np.pi * f0
k = omega_f / c

n0 = 1.0
n2 = 2.5e-20

# Generate test data
np.random.seed(0)
PHI_m        = (np.random.rand(NDX+1, NDY+1) + 1j * np.random.rand(NDX+1, NDY+1)).astype(np.complex64)
PHI_m_auxNL  = PHI_m.copy()

# Compute reference half-step
PHI_half_ref = BPM_First_half_TBC(PHI_m, PHI_m_auxNL, k, n0, NDX, NDY, DX, DY, 0.5*DZ, n2)

# Utility to write ASCII .dat
def write_ascii(path, arr):
    with open(path, 'w') as f:
        for j in range(arr.shape[1]):
            for i in range(arr.shape[0]):
                re = arr[i,j].real
                im = arr[i,j].imag
                f.write(f"{re:.8e} {im:.8e}\n")

base = './bpm_1st_h_testfiles'
write_ascii(f'{base}/phi_m0.dat',       PHI_m)
write_ascii(f'{base}/phi_aux.dat',      PHI_m_auxNL)
write_ascii(f'{base}/phi_half_ref.dat', PHI_half_ref)

print("Wrote ASCII test files:")
print(" - phi_m0.dat")
print(" - phi_aux.dat")
print(" - phi_half_ref.dat")
