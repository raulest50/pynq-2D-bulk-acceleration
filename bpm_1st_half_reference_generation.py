# bpm_1st_half_reference_generation.py
import numpy as np
from FunctionsSimplified import BPM_First_half_TBC

# --- PARAMETERS (must match tb_bpm_first_half.cpp) ---
DX = 15e-6
DY = 15e-6
DZ = 100e-6
Lx = 700e-6
Ly = 700e-6
Lz = 6e-2

NDX = int(np.floor(Lx / DX))
NDY = int(np.floor(Ly / DY))
NDZ = int(np.ceil(Lz / DZ))

print(f" NDX > {NDX} NDY > {NDY} NDZ > {NDZ}")

c = 3e8
Lambda0 = 800e-9
ko = 2 * np.pi / Lambda0
f0 = c / Lambda0
omega_f = 2 * np.pi * f0
k = omega_f / c

n0 = 1.0
n2 = 2.5e-20

# --- GENERATE TEST DATA ---
np.random.seed(0)
PHI_m = (np.random.rand(NDX + 1, NDY + 1) + 1j*np.random.rand(NDX + 1, NDY + 1)).astype(np.complex64)
PHI_m_auxNL = PHI_m.copy()

print(type(PHI_m))

# --- RUN REFERENCE HALF-STEP ---
PHI_half_ref = BPM_First_half_TBC(PHI_m, PHI_m_auxNL, k, n0, NDX, NDY, DX, DY, 0.5 * DZ, n2)

# --- DUMP EACH COMPLEX ARRAY TO A SINGLE .dat FILE ---
PHI_m       .tofile('./bpm_1st_h_testfiles/phi_m0.dat')
PHI_m_auxNL .tofile('./bpm_1st_h_testfiles/phi_aux.dat')
PHI_half_ref.tofile('./bpm_1st_h_testfiles/phi_half_ref.dat')

