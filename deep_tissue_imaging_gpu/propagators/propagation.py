import cupy as cp
import numpy as np
import deep_tissue_imaging_gpu.propagators.step_operators as so

def full_step_within_tissue_gpu(phi, tejido, d):
    """
    GPU-accelerated full step within tissue.
    
    Parameters:
    ----------
    phi : cupy.ndarray
        Complex field
    tejido : object
        Tissue properties
    d : object
        Domain properties
        
    Returns:
    -------
    phi : cupy.ndarray
        Field after applying full step
    """
    # Ensure phi is on GPU
    if isinstance(phi, np.ndarray):
        phi = cp.asarray(phi)
        
    phi = so.adi_x_gpu(phi, d.Ny, d.eps, d.k, d.dz, d.dx)
    phi = so.half_2photon_absorption_gpu(phi, tejido.beta, d.dz)
    phi = so.half_nonlinear_gpu(phi, d.k, tejido.n2, d.dz)
    phi = so.half_linear_absorption_gpu(phi, tejido.alpha, d.dz)

    phi = so.adi_y_gpu(phi, d.Nx, d.eps, d.k, d.dz, d.dy)
    phi = so.half_2photon_absorption_gpu(phi, tejido.beta, d.dz)
    phi = so.half_nonlinear_gpu(phi, d.k, tejido.n2, d.dz)
    phi = so.half_linear_absorption_gpu(phi, tejido.alpha, d.dz)
    
    return phi

def full_propagation_within_tissue_gpu(phi, tejido, d, mask_manager=None):
    """
    GPU-accelerated full propagation within tissue with optional phase mask management.
    
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
        
    Returns:
    -------
    phi_history : cupy.ndarray
        History of the field propagation
    """
    # Ensure phi is on GPU
    if isinstance(phi, np.ndarray):
        phi = cp.asarray(phi)
        
    spm = int(tejido.l_s/d.dz)
    phi_history = cp.zeros((d.Nz + 1, *phi.shape), dtype=cp.complex64)
    phi_history[0] = phi
    
    # Initialize masks at the beginning if mask_manager is provided
    if mask_manager is not None:
        # Convert X and Y to CPU if they're on GPU
        X = cp.asnumpy(d.X) if isinstance(d.X, cp.ndarray) else d.X
        Y = cp.asnumpy(d.Y) if isinstance(d.Y, cp.ndarray) else d.Y
        mask_manager.initialize_masks(phi.shape, X, Y, d.sigma_phi, d.sigma_x)
    
    # Track which mask to use (1, 2, 3)
    mask_counter = 0
    
    for k in range(0, d.Nz):
        phi = full_step_within_tissue_gpu(phi, tejido, d)
        if k % spm == 0 and k != 0:
            # Increment mask counter (1, 2, 3, 1, 2, 3, ...)
            mask_counter = (mask_counter % 3) + 1
            
            if mask_manager is not None:
                # Convert phi to CPU, apply mask, then convert back to GPU
                # This is because we're not implementing a GPU version of the phase mask
                phi_cpu = cp.asnumpy(phi)
                phi_cpu = mask_manager.apply_mask(phi_cpu, mask_counter)
                phi = cp.asarray(phi_cpu)
                print(f"aplicada mascara aleatoria {mask_counter} en z = {k}")
            else:
                # If no mask manager is provided, we need to convert to CPU,
                # apply the original function, then convert back to GPU
                X = cp.asnumpy(d.X) if isinstance(d.X, cp.ndarray) else d.X
                Y = cp.asnumpy(d.Y) if isinstance(d.Y, cp.ndarray) else d.Y
                sigma_phi = cp.asnumpy(d.sigma_phi) if isinstance(d.sigma_phi, cp.ndarray) else d.sigma_phi
                sigma_x = cp.asnumpy(d.sigma_x) if isinstance(d.sigma_x, cp.ndarray) else d.sigma_x
                
                from deep_tissue_imaging.propagators.step_operators import aplicar_mascara_fase_aleatoria
                phi_cpu = cp.asnumpy(phi)
                phi_cpu = aplicar_mascara_fase_aleatoria(phi_cpu, X, Y, sigma_phi, sigma_x)
                phi = cp.asarray(phi_cpu)
                print(f"aplicada mascara aleatoria en z = {k}")
                
        phi_history[k + 1] = phi
    
    return phi_history