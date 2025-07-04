import numpy as np

from zscan.zscan_propagators.operadores_solvers import single_bpm_step_only_linear_medium, \
    single_bpm_step_within_sample, single_bpm_step_only_linear_medium_wa


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

def apply_lens_truncated(E_in: np.ndarray,
                         X: np.ndarray,
                         Y: np.ndarray,
                         k: float,
                         f: float,
                         amp_gain: float = 1.0,
                         aperture: float | None = None
                         ) -> np.ndarray:
    """
    Lente delgada con f fija + diafragma circular.
    """
    # fase de lente delgada pura (strength=1)
    phi = np.exp(-1j * k / (2 * f) * (X**2 + Y**2))

    E_out = E_in * phi
    E_out *= amp_gain

    # si se especifica diafragma, bloquear r > aperture/2
    if aperture is not None:
        mask = (X**2 + Y**2) <= (aperture/2)**2
        E_out *= mask

    return E_out


def full_propagation_without_sample_wa(phi0, domain):
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
        phi = single_bpm_step_only_linear_medium_wa(phi, domain.k_medium, domain.dz, domain.dx, domain.dy, 1e-12,domain.alpha)
        # Store beam profile at this step
        phi_history[k + 1] = phi

    return phi, phi_history


def full_propagation_with_sample_debug(phi0, sample, sample_current_position, domain):
    phi = np.copy(phi0)
    phi_history = np.zeros((domain.Nz + 1, *phi0.shape), dtype=complex)
    phi_history[0] = phi

    for k in range(0, sample_current_position):
        phi = single_bpm_step_only_linear_medium(phi, domain.k_medium, domain.dz, domain.dx, domain.dy)
        phi_history[k + 1] = phi

    for k in range(sample_current_position, sample_current_position + sample.thickness_units + 1):
        phi = single_bpm_step_within_sample(phi, sample.k, sample.n2, domain.dz, domain.dx, domain.dy, domain.eps)
        phi_history[k + 1] = phi

    for k in range(sample_current_position + sample.thickness_units + 1, domain.Nz):
        phi = single_bpm_step_only_linear_medium(phi, domain.k_medium, domain.dz, domain.dx, domain.dy)
        phi_history[k + 1] = phi

    return phi, phi_history


def full_propagation_with_sample(phi0, sample, sample_current_position, domain):
    phi = np.copy(phi0)

    for k in range(0, sample_current_position):
        phi = single_bpm_step_only_linear_medium(phi, domain.k_medium, domain.dz, domain.dx, domain.dy)

    for k in range(sample_current_position, sample_current_position + sample.thickness_units + 1):
        phi = single_bpm_step_within_sample(phi, sample.k, sample.n2, domain.dz, domain.dx, domain.dy, domain.eps)

    for k in range(sample_current_position + sample.thickness_units + 1, domain.Nz):
        phi = single_bpm_step_only_linear_medium(phi, domain.k_medium, domain.dz, domain.dx, domain.dy)

    return phi
