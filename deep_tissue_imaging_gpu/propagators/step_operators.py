"""
Step operators for deep tissue imaging GPU package.

This module provides GPU-accelerated step operators for deep tissue imaging.
Only the halfsteps are accelerated with GPU, while the ADI operations remain on CPU.
"""

import numpy as np
import cupy as cp
from scipy.ndimage import gaussian_filter

# Import CPU operators for reference and for operations that remain on CPU
import deep_tissue_imaging.propagators.step_operators as cpu_so

# GPU-accelerated halfstep operators

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

    # Explicitly convert scalar values to CuPy scalars
    beta_gpu = beta if isinstance(beta, cp.ndarray) else cp.float32(beta)
    dz_gpu = dz if isinstance(dz, cp.ndarray) else cp.float32(dz)

    # Now all values are CuPy types, so this expression will work
    return cp.exp(-beta_gpu * dz_gpu/4 * cp.abs(phi)**2) * phi

def half_nonlinear_gpu(phi, k_sample, n2_sample, dz):
    """
    GPU-accelerated nonlinear Kerr effect operator.

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
        Field after applying nonlinear Kerr effect
    """
    # Ensure inputs are on GPU
    if isinstance(phi, np.ndarray):
        phi = cp.asarray(phi)

    # Explicitly convert scalar values to CuPy scalars
    k_gpu = k_sample if isinstance(k_sample, cp.ndarray) else cp.float32(k_sample)
    n2_gpu = n2_sample if isinstance(n2_sample, cp.ndarray) else cp.float32(n2_sample)
    dz_gpu = dz if isinstance(dz, cp.ndarray) else cp.float32(dz)

    # Apply nonlinear phase shift
    phase = cp.exp(1j * k_gpu * n2_gpu * dz_gpu/2 * cp.abs(phi)**2)
    return phase * phi

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

    # Explicitly convert scalar values to CuPy scalars
    alpha_gpu = alpha if isinstance(alpha, cp.ndarray) else cp.float32(alpha)
    dz_gpu = dz if isinstance(dz, cp.ndarray) else cp.float32(dz)

    # Apply linear absorption
    return cp.exp(-alpha_gpu * dz_gpu/4) * phi

# CPU functions are used directly from the original module
# These are included here for reference

# ADI operators
adi_x = cpu_so.adi_x
adi_y = cpu_so.adi_y

# Phase mask application
aplicar_mascara_fase_aleatoria = cpu_so.aplicar_mascara_fase_aleatoria