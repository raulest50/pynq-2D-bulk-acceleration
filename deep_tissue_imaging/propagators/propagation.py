import numpy as np
import deep_tissue_imaging.propagators.step_operators as so

def full_step_within_tissue(phi, tejido, d):
    phi = so.adi_x(phi, d.Ny, d.eps, d.k, d.dz, d.dx)
    phi = so.half_2photon_absorption(phi, tejido.beta, d.dz)
    phi = so.half_nonlinear(phi, d.k, tejido.n2, d.dz)
    phi = so.half_linear_absorption(phi, tejido.alpha, d.dz)

    phi = so.adi_y(phi, d.Nx, d.eps, d.k, d.dz, d.dy)
    phi = so.half_2photon_absorption(phi, tejido.beta, d.dz)
    phi = so.half_nonlinear(phi, d.k, tejido.n2, d.dz)
    phi = so.half_linear_absorption(phi, tejido.alpha, d.dz)
    return phi

def full_propagation_within_tissue(phi, tejido, d):
    spm = int(tejido.l_s/d.dz)
    phi_history = np.zeros((d.Nz + 1, *phi.shape), dtype=np.complex64)
    phi_history[0] = phi
    for k in range(0, d.Nz):
        phi = full_step_within_tissue(phi, tejido, d)
        if k % spm == 0 and k != 0:
            phi = so.aplicar_mascara_fase_aleatoria(phi, d.X, d.Y, d.sigma_phi, d.sigma_x)
            print(f"aplicada mascara aleatoria en z = {k}")
        phi_history[k + 1] = phi

    return phi_history

def compute_psf(phi, tejido, dominio):
    pass
