import cupy as cp
import numpy as np
from scipy.ndimage import gaussian_filter

# GPU-accelerated step operators for deep tissue imaging

## Operador Dispersion

def custom_thomas_solver_gpu(dp, dp1, dp2, do, b):
    """
    Solves a tridiagonal system with special structure using Thomas algorithm on GPU.

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
    b : cupy.ndarray
        Right-hand side vector

    Returns:
    -------
    x : cupy.ndarray
        Solution vector
    """
    # Ensure inputs are on GPU
    if isinstance(dp, np.ndarray):
        dp = cp.asarray(dp)
    if isinstance(dp1, np.ndarray):
        dp1 = cp.asarray(dp1)
    if isinstance(dp2, np.ndarray):
        dp2 = cp.asarray(dp2)
    if isinstance(do, np.ndarray):
        do = cp.asarray(do)
    if isinstance(b, np.ndarray):
        b = cp.asarray(b)

    n = len(b)

    # Create arrays for the modified coefficients
    c_prime = cp.zeros(n-1, dtype=b.dtype)  # Upper diagonal
    d_prime = cp.zeros(n, dtype=b.dtype)    # Modified right-hand side

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
    x = cp.zeros(n, dtype=b.dtype)
    x[n-1] = d_prime[n-1]

    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x


def compute_b_vector_gpu(dp, dp1, dp2, do, x0):
    """
    Multiplies a tridiagonal matrix with special structure by a vector on GPU.
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
    x0 : cupy.ndarray
        Input vector to be multiplied

    Returns:
    -------
    b : cupy.ndarray
        Result of the matrix-vector multiplication
    """
    # Ensure inputs are on GPU
    if isinstance(dp, np.ndarray):
        dp = cp.asarray(dp)
    if isinstance(dp1, np.ndarray):
        dp1 = cp.asarray(dp1)
    if isinstance(dp2, np.ndarray):
        dp2 = cp.asarray(dp2)
    if isinstance(do, np.ndarray):
        do = cp.asarray(do)
    if isinstance(x0, np.ndarray):
        x0 = cp.asarray(x0)

    n = len(x0)
    b = cp.zeros(n, dtype=x0.dtype)

    # First row: b[0] = dp1 * x0[0] + do * x0[1]
    b[0] = dp1 * x0[0] + do * x0[1]

    # Middle rows: b[i] = do * x0[i-1] + dp * x0[i] + do * x0[i+1]
    # Using a for loop as requested
    for i in range(1, n-1):
        b[i] = do * x0[i-1] + dp * x0[i] + do * x0[i+1]

    # Last row: b[n-1] = do * x0[n-2] + dp2 * x0[n-1]
    b[n-1] = do * x0[n-2] + dp2 * x0[n-1]

    return b


def adi_x_gpu(phi, Ny, eps, k, dz, dx):
    """
    GPU-accelerated ADI operator for x-direction.

    Parameters:
    ----------
    phi : cupy.ndarray
        Complex field
    Ny : int
        Number of points in y direction
    eps : float
        Small value to avoid division by zero
    k : float
        Wave number
    dz : float
        Step size in z direction
    dx : float
        Step size in x direction

    Returns:
    -------
    phi_inter : cupy.ndarray
        Field after applying ADI operator
    """
    # Ensure inputs are on GPU
    if isinstance(phi, np.ndarray):
        phi = cp.asarray(phi)
    if isinstance(eps, np.ndarray):
        eps = cp.asarray(eps)
    if isinstance(k, np.ndarray):
        k = cp.asarray(k)
    if isinstance(dz, np.ndarray):
        dz = cp.asarray(dz)
    if isinstance(dx, np.ndarray):
        dx = cp.asarray(dx)

    ung = 1j * dz / (4 * k * dx**2)
    phi_inter = cp.zeros_like(phi, dtype=cp.complex64)

    for j in range(Ny):
        if cp.abs(phi[1, j]) < eps:
            ratio_x0 = 1.0
        else:
            ratio_x0 = phi[0, j] / phi[1, j]

        if cp.abs(phi[-2, j]) < eps:
            ratio_xn = 1.0
        else:
            ratio_xn = phi[-1, j] / phi[-2, j]

        dp1_B = -2 * ung + 1.0 + ung * ratio_x0
        dp2_B = -2 * ung + 1.0 + ung * ratio_xn
        dp_B = -2 * ung + 1.0
        do_B = ung

        b = compute_b_vector_gpu(dp_B, dp1_B, dp2_B, do_B, phi[:, j])

        dp1_A = 2 * ung + 1.0 - ung * ratio_x0
        dp2_A = 2 * ung + 1.0 - ung * ratio_xn
        dp_A = 2 * ung + 1.0
        do_A = -ung

        phi_inter[:, j] = custom_thomas_solver_gpu(dp_A, dp1_A, dp2_A, do_A, b)

    return phi_inter


def adi_y_gpu(phi, Nx, eps, k, dz, dy):
    """
    GPU-accelerated ADI operator for y-direction.

    Parameters:
    ----------
    phi : cupy.ndarray
        Complex field
    Nx : int
        Number of points in x direction
    eps : float
        Small value to avoid division by zero
    k : float
        Wave number
    dz : float
        Step size in z direction
    dy : float
        Step size in y direction

    Returns:
    -------
    phi_inter : cupy.ndarray
        Field after applying ADI operator
    """
    # Ensure inputs are on GPU
    if isinstance(phi, np.ndarray):
        phi = cp.asarray(phi)
    if isinstance(eps, np.ndarray):
        eps = cp.asarray(eps)
    if isinstance(k, np.ndarray):
        k = cp.asarray(k)
    if isinstance(dz, np.ndarray):
        dz = cp.asarray(dz)
    if isinstance(dy, np.ndarray):
        dy = cp.asarray(dy)

    ung = 1j * dz / (4 * k * dy**2)
    phi_inter = cp.zeros_like(phi, dtype=cp.complex64)

    for i in range(Nx):
        if cp.abs(phi[i, 1]) < eps:
            ratio_y0 = 1.0
        else:
            ratio_y0 = phi[i, 0] / phi[i, 1]

        if cp.abs(phi[i, -2]) < eps:
            ratio_yn = 1.0
        else:
            ratio_yn = phi[i, -1] / phi[i, -2]

        dp1_B = -2 * ung + 1.0 + ung * ratio_y0
        dp2_B = -2 * ung + 1.0 + ung * ratio_yn
        dp_B = -2 * ung + 1.0
        do_B = ung

        b = compute_b_vector_gpu(dp_B, dp1_B, dp2_B, do_B, phi[i, :])

        dp1_A = 2 * ung + 1.0 - ung * ratio_y0
        dp2_A = 2 * ung + 1.0 - ung * ratio_yn
        dp_A = 2 * ung + 1.0
        do_A = -ung

        phi_inter[i, :] = custom_thomas_solver_gpu(dp_A, dp1_A, dp2_A, do_A, b)

    return phi_inter


## Operador N - Kerr

def half_nonlinear_gpu(phi, k_sample, n2_sample, dz):
    """
    GPU-accelerated nonlinear operator for Kerr effect.

    Parameters:
    ----------
    phi : cupy.ndarray
        Complex field
    k_sample : float or numpy.float32
        Wave number in the sample
    n2_sample : float or numpy.float32
        Nonlinear refractive index
    dz : float or numpy.float32
        Step size in z direction

    Returns:
    -------
    phi : cupy.ndarray
        Field after applying nonlinear operator
    """
    # Ensure inputs are on GPU
    if isinstance(phi, np.ndarray):
        phi = cp.asarray(phi)

    # Convert scalar values to CuPy scalars
    if isinstance(k_sample, (np.number, np.ndarray)):
        k_sample = cp.asarray(k_sample, dtype=cp.float32)
    else:
        k_sample = cp.float32(k_sample)

    if isinstance(n2_sample, (np.number, np.ndarray)):
        n2_sample = cp.asarray(n2_sample, dtype=cp.float32)
    else:
        n2_sample = cp.float32(n2_sample)

    if isinstance(dz, (np.number, np.ndarray)):
        dz = cp.asarray(dz, dtype=cp.float32)
    else:
        dz = cp.float32(dz)

    phase = cp.exp(1j * k_sample * n2_sample * dz/2 * cp.abs(phi)**2)
    return phase * phi


## Operadores de absorcion

# absorcion lineal
def half_linear_absorption_gpu(phi, alpha, dz):
    """
    GPU-accelerated linear absorption operator.

    Parameters:
    ----------
    phi : cupy.ndarray
        Complex field
    alpha : float or numpy.float32
        Linear absorption coefficient
    dz : float or numpy.float32
        Step size in z direction

    Returns:
    -------
    phi : cupy.ndarray
        Field after applying linear absorption
    """
    # Ensure inputs are on GPU
    if isinstance(phi, np.ndarray):
        phi = cp.asarray(phi)

    # Convert scalar values to CuPy scalars
    if isinstance(alpha, (np.number, np.ndarray)):
        alpha = cp.asarray(alpha, dtype=cp.float32)
    else:
        alpha = cp.float32(alpha)

    if isinstance(dz, (np.number, np.ndarray)):
        dz = cp.asarray(dz, dtype=cp.float32)
    else:
        dz = cp.float32(dz)

    return cp.exp(-alpha * dz/4) * phi

# absorcion de 2 fotones
def half_2photon_absorption_gpu(phi, beta, dz):
    """
    GPU-accelerated two-photon absorption operator.

    Parameters:
    ----------
    phi : cupy.ndarray
        Complex field
    beta : float or numpy.float32
        Two-photon absorption coefficient
    dz : float or numpy.float32
        Step size in z direction

    Returns:
    -------
    phi : cupy.ndarray
        Field after applying two-photon absorption
    """
    # Ensure inputs are on GPU
    if isinstance(phi, np.ndarray):
        phi = cp.asarray(phi)

    # Convert scalar values to CuPy scalars
    # This is the key fix - we need to handle NumPy scalar types
    if isinstance(beta, (np.number, np.ndarray)):
        beta = cp.asarray(beta, dtype=cp.float32)
    else:
        beta = cp.float32(beta)

    if isinstance(dz, (np.number, np.ndarray)):
        dz = cp.asarray(dz, dtype=cp.float32)
    else:
        dz = cp.float32(dz)

    # Now all values are CuPy types, so this expression will work
    return cp.exp(-beta * dz/4 * cp.abs(phi)**2) * phi

# Note: We do NOT implement a GPU version of aplicar_mascara_fase_aleatoria
# as per the requirements. Instead, we'll use the PhaseMaskManager.
