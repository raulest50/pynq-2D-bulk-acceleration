import numpy as np
from depp_tissue_imaging.elementos.lasers import fuente_microscopia_1 as laser
from depp_tissue_imaging.elementos.tejidos import cerebro_emb_pez_cebra as tejido

# Parametros de dominio

Lz = 200e-6 # 200um
Nz = 200
dz = Lz / Nz # 1um

Lx, Ly = 45e-6, 45e-3 # 2mm x 2mm
Nx, Ny = 128, 128
dx = Lx / Nx # 0.35um
dy = Ly / Ny # 0.35um

x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y)

