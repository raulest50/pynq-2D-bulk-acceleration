import numpy as np


def custom_thomas_solver(dp, dp1, dp2, do, b):
    """
    Solves a tridiagonal system with special structure using Thomas algorithm.

    Parameters:
    ----------
    dp : float
        Value for all elements in the main diagonal except first and last
    dp1 : float
        Value for the first element in the main diagonal [0,0]
    dp2 : float
        Value for the last element in the main diagonal [-1,-1]
    do : float
        Value for all elements in the off-diagonals
    b : numpy.ndarray
        Right-hand side vector

    Returns:
    -------
    x : numpy.ndarray
        Solution vector
    """
    n = len(b)

    # Create arrays for the modified coefficients
    c_prime = np.zeros(n-1, dtype=b.dtype)  # Upper diagonal
    d_prime = np.zeros(n, dtype=b.dtype)    # Modified right-hand side

    # Forward elimination
    # First row
    c_prime[0] = do / dp1
    d_prime[0] = b[0] / dp1

    # Middle rows
    for i in range(1, n-1):
        denominator = dp - do * c_prime[i-1]
        c_prime[i] = do / denominator
        d_prime[i] = (b[i] - do * d_prime[i-1]) / denominator

    # Last row
    d_prime[n-1] = (b[n-1] - do * d_prime[n-2]) / (dp2 - do * c_prime[n-2])

    # Back substitution
    x = np.zeros(n, dtype=b.dtype)
    x[n-1] = d_prime[n-1]

    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x


def compute_b_vector(dp, dp1, dp2, do, x0):
    """
    Multiplies a tridiagonal matrix with special structure by a vector.
    Optimized for vectors with dimensions > 3, but handles all cases.

    Parameters:
    ----------
    dp : float
        Value for all elements in the main diagonal except first and last
    dp1 : float
        Value for the first element in the main diagonal [0,0]
    dp2 : float
        Value for the last element in the main diagonal [-1,-1]
    do : float
        Value for all elements in the off-diagonals
    x0 : numpy.ndarray
        Input vector to be multiplied

    Returns:
    -------
    b : numpy.ndarray
        Result of the matrix-vector multiplication
    """
    n = len(x0)
    b = np.zeros(n, dtype=x0.dtype)

    # For n > 2, use the implementation with a for loop
    # First row: b[0] = dp1 * x0[0] + do * x0[1]
    b[0] = dp1 * x0[0] + do * x0[1]

    # Middle rows: b[i] = do * x0[i-1] + dp * x0[i] + do * x0[i+1]
    # Using a for loop as requested
    for i in range(1, n-1):
        b[i] = do * x0[i-1] + dp * x0[i] + do * x0[i+1]

    # Last row: b[n-1] = do * x0[n-2] + dp2 * x0[n-1]
    b[n-1] = do * x0[n-2] + dp2 * x0[n-1]

    return b


def adi_x(phi, Ny, eps, k, dz, dx):
    ung = 1j * dz / (4 * k * dx**2)
    phi_inter = np.zeros_like(phi, dtype=complex)
    for j in range(Ny):

        if abs(phi[1, j]) < eps:
            ratio_x0 = 1.0
        else:
            ratio_x0 = phi[0, j] / phi[1, j]

        if abs(phi[-2, j]) < eps:
            ratio_xn = 1.0
        else:
            ratio_xn = phi[-1, j] / phi[-2, j]

        dp1_B = -2 * ung + 1 + ung * ratio_x0
        dp2_B = -2 * ung + 1 + ung * ratio_xn
        dp_B = -2 * ung + 1
        do_B = ung

        b = compute_b_vector(dp_B, dp1_B, dp2_B, do_B, phi[:, j])

        dp1_A = 2 * ung + 1 - ung * ratio_x0
        dp2_A = 2 * ung + 1 - ung * ratio_xn
        dp_A = 2 * ung + 1
        do_A = -ung

        phi_inter[:, j] = custom_thomas_solver(dp_A, dp1_A, dp2_A, do_A, b)

    return phi_inter


def adi_y(phi, Nx, eps, k, dz, dy):
    ung = 1j * dz / (4 * k * dy**2)
    phi_inter = np.zeros_like(phi, dtype=complex)
    for i in range(Nx):

        if abs(phi[i, 1]) < eps:
            ratio_y0 = 1.0
        else:
            ratio_y0 = phi[i, 0] / phi[i, 1]

        if abs(phi[i, -2]) < eps:
            ratio_yn = 1.0
        else:
            ratio_yn = phi[i, -1] / phi[i, -2]

        dp1_B = -2 * ung + 1 + ung * ratio_y0
        dp2_B = -2 * ung + 1 + ung * ratio_yn
        dp_B = -2 * ung + 1
        do_B = ung

        b = compute_b_vector(dp_B, dp1_B, dp2_B, do_B, phi[i, :])

        dp1_A = 2 * ung + 1 - ung * ratio_y0
        dp2_A = 2 * ung + 1 - ung * ratio_yn
        dp_A = 2 * ung + 1
        do_A = -ung

        phi_inter[i, :] = custom_thomas_solver(dp_A, dp1_A, dp2_A, do_A, b)

    return phi_inter


def half_nonlinear(phi, k_sample, n2_sample, dz):
   phase = np.exp( 1j * k_sample * n2_sample * dz/2 *np.abs(phi)**2 )
   return phase * phi


def single_bpm_step_within_sample(phi, k_sample, n2_sample, dz, dx, dy, eps=1e-12):
    Ny, Nx = phi.shape
    phi_inter = adi_x(phi, Ny, eps, k_sample, dz, dx)
    phi_inter = half_nonlinear(phi_inter, k_sample, n2_sample, dz)
    phi_inter = adi_y(phi_inter, Nx, eps, k_sample, dz, dy)
    phi_inter = half_nonlinear(phi_inter, k_sample, n2_sample, dz)
    return phi_inter


def single_bpm_step_only_linear_medium(phi, k_medium, dz, dx, dy, eps=1e-12):
    """
    Performs a single BPM step in a linear medium (without nonlinear effects).

    Parameters:
    ----------
    phi : numpy.ndarray
        Input complex field
    k_medium : float
        Wave number in the medium
    dz : float
        Step size in the propagation direction
    dx : float
        Step size in the x direction
    dy : float
        Step size in the y direction
    eps : float, optional
        Small value to avoid division by zero, default is 1e-12

    Returns:
    -------
    phi_out : numpy.ndarray
        Output complex field after propagation
    """
    Ny, Nx = phi.shape
    phi_inter = adi_x(phi, Ny, eps, k_medium, dz, dx)
    phi_out = adi_y(phi_inter, Nx,   eps, k_medium, dz, dy)
    return phi_out

def full_propagation_with_sample(phi0, sample, domain):
    phi = np.copy(phi0)

    for k in range(0, sample.position):
        phi = single_bpm_step_only_linear_medium(phi, domain.k_medium, domain.dz, domain.dx, domain.dy)

    for k in range(sample.position, sample.position + sample.thickness + 1):
        phi = single_bpm_step_within_sample(phi, sample.k, sample.n2, domain.dz, domain.dx, domain.dy, domain.eps)

    for k in range(sample.position + sample.thickness + 1, domain.Nz):
        phi = single_bpm_step_only_linear_medium(phi, domain.k_medium, domain.dz, domain.dx, domain.dy)

    return phi

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

def z_scan(phi0, sample, domain):
    phi = np.copy(phi0)
    phi_stops = np.zeros((domain.Nz + 1, *phi0.shape), dtype=complex)
    for sample_position in sample.stops:
        phi = full_propagation_with_sample(phi, sample, domain)
        phi_stops[sample_position] = phi
