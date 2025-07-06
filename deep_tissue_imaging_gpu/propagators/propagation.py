"""
Propagation functions for deep tissue imaging GPU package.

This module provides hybrid CPU-GPU propagation functions for deep tissue imaging.
It uses CPU for ADI operations and GPU for halfsteps, minimizing CPU-GPU transfers.
"""

import numpy as np
import cupy as cp
import time

import deep_tissue_imaging.propagators.step_operators as cpu_so
import deep_tissue_imaging_gpu.propagators.step_operators as gpu_so

def full_step_within_tissue_hybrid(phi, tejido, d):
    """
    Hybrid CPU-GPU implementation of a full step within tissue.

    Uses CPU for ADI operations and GPU for halfsteps.

    Parameters:
    ----------
    phi : ndarray or cupy.ndarray
        Complex field
    tejido : object
        Tissue properties
    d : object
        Domain properties

    Returns:
    -------
    phi : cupy.ndarray
        Field after applying one full step
    """
    # Convert to CPU for ADI x-direction
    if isinstance(phi, cp.ndarray):
        phi_cpu = cp.asnumpy(phi).astype(np.complex64)  # Explicitly set data type
    else:
        phi_cpu = phi.astype(np.complex64) if phi.dtype != np.complex64 else phi

    # ADI x-direction on CPU
    phi_cpu = cpu_so.adi_x(phi_cpu, d.Ny, d.eps, d.k, d.dz, d.dx)

    # Convert to GPU for halfsteps
    phi_gpu = cp.asarray(phi_cpu)

    # Apply halfsteps on GPU
    phi_gpu = gpu_so.half_2photon_absorption_gpu(phi_gpu, tejido.beta, d.dz)
    phi_gpu = gpu_so.half_nonlinear_gpu(phi_gpu, d.k, tejido.n2, d.dz)
    phi_gpu = gpu_so.half_linear_absorption_gpu(phi_gpu, tejido.alpha, d.dz)

    # Convert back to CPU for ADI y-direction
    phi_cpu = cp.asnumpy(phi_gpu).astype(np.complex64)  # Explicitly set data type

    # ADI y-direction on CPU
    phi_cpu = cpu_so.adi_y(phi_cpu, d.Nx, d.eps, d.k, d.dz, d.dy)

    # Convert back to GPU for final halfsteps
    phi_gpu = cp.asarray(phi_cpu)

    # Apply final halfsteps on GPU
    phi_gpu = gpu_so.half_2photon_absorption_gpu(phi_gpu, tejido.beta, d.dz)
    phi_gpu = gpu_so.half_nonlinear_gpu(phi_gpu, d.k, tejido.n2, d.dz)
    phi_gpu = gpu_so.half_linear_absorption_gpu(phi_gpu, tejido.alpha, d.dz)

    return phi_gpu

def full_propagation_within_tissue_hybrid(phi, tejido, d, mask_manager=None, store_history=True):
    """
    Hybrid CPU-GPU implementation of full propagation within tissue.

    Uses CPU for ADI operations and GPU for halfsteps, with optional history storage.

    Parameters:
    ----------
    phi : ndarray or cupy.ndarray
        Initial complex field
    tejido : object
        Tissue properties
    d : object
        Domain properties
    mask_manager : PhaseMaskManager, optional
        Phase mask manager for consistent masks
    store_history : bool, optional
        Whether to store the entire propagation history (default: True)

    Returns:
    -------
    If store_history is True:
        ndarray: History of the field propagation (on CPU)
    If store_history is False:
        tuple: (phi_initial, phi_final) - Initial and final fields (on CPU)
    """
    # Ensure phi is on GPU
    if isinstance(phi, np.ndarray):
        phi_gpu = cp.asarray(phi)
    else:
        phi_gpu = phi

    # Store initial state
    phi_initial_gpu = phi_gpu.copy()
    phi_initial_cpu = cp.asnumpy(phi_initial_gpu).astype(np.complex64)  # Explicitly set data type

    # Pre-compute parameters
    spm = int(tejido.l_s/d.dz)

    # Initialize history if needed
    if store_history:
        phi_history = np.zeros((d.Nz + 1, *phi_initial_cpu.shape), dtype=np.complex64)
        phi_history[0] = phi_initial_cpu

    # Initialize masks at the beginning if mask_manager is provided
    if mask_manager is not None:
        print("Initializing phase masks...")
        mask_manager.initialize_masks(phi_initial_cpu.shape, d.X, d.Y, d.sigma_phi, d.sigma_x)

    # Track which mask to use (1, 2, 3)
    mask_counter = 0

    # Warm up GPU with a small example before timing
    print("Warming up GPU...")
    small_phi = cp.ones((32, 32), dtype=cp.complex64)
    _ = gpu_so.half_2photon_absorption_gpu(small_phi, tejido.beta, d.dz)
    _ = gpu_so.half_nonlinear_gpu(small_phi, d.k, tejido.n2, d.dz)
    _ = gpu_so.half_linear_absorption_gpu(small_phi, tejido.alpha, d.dz)
    cp.cuda.Stream.null.synchronize()

    # Main propagation loop
    print("\nRunning hybrid CPU-GPU simulation...")
    start_time = time.time()

    for k in range(0, d.Nz):
        # Propagate one step (hybrid CPU-GPU)
        phi_gpu = full_step_within_tissue_hybrid(phi_gpu, tejido, d)

        # Apply phase mask if needed (only 3 times in 361 steps)
        if k % spm == 0 and k != 0:
            # Increment mask counter (1, 2, 3, 1, 2, 3, ...)
            mask_counter = (mask_counter % 3) + 1

            # Convert to CPU for phase mask application
            phi_cpu = cp.asnumpy(phi_gpu).astype(np.complex64)  # Explicitly set data type

            if mask_manager is not None:
                # Use the mask manager with the current mask index
                phi_cpu = mask_manager.apply_mask(phi_cpu, mask_counter)
                print(f"aplicada mascara aleatoria {mask_counter} en z = {k}")
            else:
                # Use the original function if no mask manager is provided
                phi_cpu = cpu_so.aplicar_mascara_fase_aleatoria(phi_cpu, d.X, d.Y, d.sigma_phi, d.sigma_x)
                print(f"aplicada mascara aleatoria en z = {k}")

            # Convert back to GPU
            phi_gpu = cp.asarray(phi_cpu)

        # Store history if needed
        if store_history:
            phi_history[k + 1] = cp.asnumpy(phi_gpu).astype(np.complex64)  # Explicitly set data type

    # Ensure all GPU operations are complete
    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Hybrid CPU-GPU execution time: {execution_time:.6f} seconds")

    # Return appropriate result based on store_history flag
    if store_history:
        return phi_history
    else:
        return phi_initial_cpu, cp.asnumpy(phi_gpu).astype(np.complex64)  # Explicitly set data type
