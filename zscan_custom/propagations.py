import numpy as np

from zscan_custom.operadores_solvers import single_bpm_step_only_linear_medium, \
    single_bpm_step_within_sample


def z_scan(phi0, sample, domain):
    phi = np.copy(phi0)
    phi_at_aperture_stops = np.zeros((len(sample.stops), *phi0.shape), dtype=complex)
    T = np.zeros(len(sample.stops))
    for v in range(0, len(sample.stops)):
        phi = full_propagation_with_sample(phi, sample, sample.stops[v], domain)
        #phi_at_aperture_stops[v] = phi
        #T[v] = compute_transmitance(phi0, phi)
        print(f"terminada propagacion en z = {sample.stops[v]}")
    return T


def full_propagation_without_sample(phi0, domain):
    """
    Propagates a beam through a medium without a sample, storing the beam profile at each z-step.

    Parameters:
    ----------
    phi0 : numpy.ndarray
        Initial complex field
    domain : object
        Domain object containing simulation parameters

    Returns:
    -------
    phi : numpy.ndarray
        Final complex field after propagation
    phi_history : numpy.ndarray
        3D array containing the beam profile at each z-step
    """
    phi = np.copy(phi0)
    # Create a 3D array to store all beam profiles
    phi_history = np.zeros((domain.Nz + 1, *phi0.shape), dtype=complex)
    # Store initial beam profile
    phi_history[0] = phi

    for k in range(0, domain.Nz):
        phi = single_bpm_step_only_linear_medium(phi, domain.k_medium, domain.dz, domain.dx, domain.dy)
        # Store beam profile at this step
        phi_history[k + 1] = phi

    return phi, phi_history


def full_propagation_with_sample_debug(phi0, sample, sample_current_position,domain):
    phi = np.copy(phi0)

    for k in range(0, sample_current_position):
        phi = single_bpm_step_only_linear_medium(phi, domain.k_medium, domain.dz, domain.dx, domain.dy)

    for k in range(sample_current_position, sample_current_position + sample.thickness_units + 1):
        phi = single_bpm_step_within_sample(phi, sample.k, sample.n2, domain.dz, domain.dx, domain.dy, domain.eps)

    for k in range(sample_current_position + sample.thickness_units + 1, domain.Nz):
        phi = single_bpm_step_only_linear_medium(phi, domain.k_medium, domain.dz, domain.dx, domain.dy)

    return phi


def full_propagation_with_sample(phi0, sample, sample_current_position,domain):
    phi = np.copy(phi0)

    for k in range(0, sample_current_position):
        phi = single_bpm_step_only_linear_medium(phi, domain.k_medium, domain.dz, domain.dx, domain.dy)

    for k in range(sample_current_position, sample_current_position + sample.thickness_units + 1):
        phi = single_bpm_step_within_sample(phi, sample.k, sample.n2, domain.dz, domain.dx, domain.dy, domain.eps)

    for k in range(sample_current_position + sample.thickness_units + 1, domain.Nz):
        phi = single_bpm_step_only_linear_medium(phi, domain.k_medium, domain.dz, domain.dx, domain.dy)

    return phi
