import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import warnings

# Flag to check if CUDA is available
CUDA_AVAILABLE = True
try:
    cuda.Device(0)  # Try to access the first CUDA device
except:
    CUDA_AVAILABLE = False
    warnings.warn("CUDA device not found. Falling back to CPU implementation.")

# Import GPU-accelerated operators and solvers
from zscan.zscan_propagators.operadores_solvers_gpu import (
    single_bpm_step_only_linear_medium,
    single_bpm_step_within_sample,
    single_bpm_step_only_linear_medium_wa
)

# Import original CPU implementations as fallback
from zscan.zscan_propagators.propagations import (
    z_scan as cpu_z_scan,
    full_propagation_without_sample as cpu_full_propagation_without_sample,
    apply_lens_truncated as cpu_apply_lens_truncated,
    full_propagation_without_sample_wa as cpu_full_propagation_without_sample_wa,
    full_propagation_with_sample_debug as cpu_full_propagation_with_sample_debug,
    full_propagation_with_sample as cpu_full_propagation_with_sample
)

def z_scan(phi0, sample, domain):
    """
    GPU-accelerated version of z_scan.
    Falls back to CPU implementation if CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        return cpu_z_scan(phi0, sample, domain)
    
    phi = np.copy(phi0)
    phi_at_aperture_stops = np.zeros((len(sample.stops), *phi0.shape), dtype=complex)
    T = np.zeros(len(sample.stops))
    for v in range(0, len(sample.stops)):
        phi = full_propagation_with_sample(phi, sample, sample.stops[v], domain)
        print(f"terminada propagacion en z = {sample.stops[v]}")
    return T

def full_propagation_without_sample(phi0, domain):
    """
    GPU-accelerated version of full_propagation_without_sample.
    Falls back to CPU implementation if CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        return cpu_full_propagation_without_sample(phi0, domain)
    
    phi = np.copy(phi0)
    # Create a 3D array to store all beam profiles
    phi_history = np.zeros((domain.Nz + 1, *phi0.shape), dtype=complex)
    # Store initial beam profile
    phi_history[0] = phi
    
    # If GPU is available, transfer phi to GPU once
    if CUDA_AVAILABLE:
        phi_gpu = gpuarray.to_gpu(phi)
        
        for k in range(0, domain.Nz):
            # Process on GPU
            phi_gpu = single_bpm_step_only_linear_medium(
                phi_gpu, domain.k_medium, domain.dz, domain.dx, domain.dy
            )
            # Transfer result back to CPU for storage
            phi = phi_gpu.get()
            phi_history[k + 1] = phi
    else:
        # Fallback to CPU implementation
        for k in range(0, domain.Nz):
            phi = single_bpm_step_only_linear_medium(
                phi, domain.k_medium, domain.dz, domain.dx, domain.dy
            )
            phi_history[k + 1] = phi
    
    return phi, phi_history

def apply_lens_truncated(E_in, X, Y, k, f, amp_gain=1.0, aperture=None):
    """
    GPU-accelerated version of apply_lens_truncated.
    Falls back to CPU implementation if CUDA is not available.
    """
    # This function is simple enough that GPU acceleration may not provide significant benefits
    # We'll use the CPU implementation for now
    return cpu_apply_lens_truncated(E_in, X, Y, k, f, amp_gain, aperture)

def full_propagation_without_sample_wa(phi0, domain):
    """
    GPU-accelerated version of full_propagation_without_sample_wa.
    Falls back to CPU implementation if CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        return cpu_full_propagation_without_sample_wa(phi0, domain)
    
    phi = np.copy(phi0)
    # Create a 3D array to store all beam profiles
    phi_history = np.zeros((domain.Nz + 1, *phi0.shape), dtype=complex)
    # Store initial beam profile
    phi_history[0] = phi
    
    # If GPU is available, transfer phi to GPU once
    if CUDA_AVAILABLE:
        phi_gpu = gpuarray.to_gpu(phi)
        
        for k in range(0, domain.Nz):
            # Process on GPU
            phi_gpu = single_bpm_step_only_linear_medium_wa(
                phi_gpu, domain.k_medium, domain.dz, domain.dx, domain.dy, 
                1e-12, domain.alpha
            )
            # Transfer result back to CPU for storage
            phi = phi_gpu.get()
            phi_history[k + 1] = phi
    else:
        # Fallback to CPU implementation
        for k in range(0, domain.Nz):
            phi = single_bpm_step_only_linear_medium_wa(
                phi, domain.k_medium, domain.dz, domain.dx, domain.dy, 
                1e-12, domain.alpha
            )
            phi_history[k + 1] = phi
    
    return phi, phi_history

def full_propagation_with_sample_debug(phi0, sample, sample_current_position, domain):
    """
    GPU-accelerated version of full_propagation_with_sample_debug.
    Falls back to CPU implementation if CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        return cpu_full_propagation_with_sample_debug(phi0, sample, sample_current_position, domain)
    
    phi = np.copy(phi0)
    phi_history = np.zeros((domain.Nz + 1, *phi0.shape), dtype=complex)
    phi_history[0] = phi
    
    # If GPU is available, transfer phi to GPU once
    if CUDA_AVAILABLE:
        phi_gpu = gpuarray.to_gpu(phi)
        
        # First propagation through air
        for k in range(0, sample_current_position):
            phi_gpu = single_bpm_step_only_linear_medium(
                phi_gpu, domain.k_medium, domain.dz, domain.dx, domain.dy
            )
            phi = phi_gpu.get()
            phi_history[k + 1] = phi
        
        # Propagation through sample
        for k in range(sample_current_position, sample_current_position + sample.thickness_units + 1):
            phi_gpu = single_bpm_step_within_sample(
                phi_gpu, sample.k, sample.n2, domain.dz, domain.dx, domain.dy, domain.eps
            )
            phi = phi_gpu.get()
            phi_history[k + 1] = phi
        
        # Final propagation through air
        for k in range(sample_current_position + sample.thickness_units + 1, domain.Nz):
            phi_gpu = single_bpm_step_only_linear_medium(
                phi_gpu, domain.k_medium, domain.dz, domain.dx, domain.dy
            )
            phi = phi_gpu.get()
            phi_history[k + 1] = phi
    else:
        # Fallback to CPU implementation
        for k in range(0, sample_current_position):
            phi = single_bpm_step_only_linear_medium(
                phi, domain.k_medium, domain.dz, domain.dx, domain.dy
            )
            phi_history[k + 1] = phi
        
        for k in range(sample_current_position, sample_current_position + sample.thickness_units + 1):
            phi = single_bpm_step_within_sample(
                phi, sample.k, sample.n2, domain.dz, domain.dx, domain.dy, domain.eps
            )
            phi_history[k + 1] = phi
        
        for k in range(sample_current_position + sample.thickness_units + 1, domain.Nz):
            phi = single_bpm_step_only_linear_medium(
                phi, domain.k_medium, domain.dz, domain.dx, domain.dy
            )
            phi_history[k + 1] = phi
    
    return phi, phi_history

def full_propagation_with_sample(phi0, sample, sample_current_position, domain):
    """
    GPU-accelerated version of full_propagation_with_sample.
    Falls back to CPU implementation if CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        return cpu_full_propagation_with_sample(phi0, sample, sample_current_position, domain)
    
    phi = np.copy(phi0)
    
    # If GPU is available, transfer phi to GPU once
    if CUDA_AVAILABLE:
        phi_gpu = gpuarray.to_gpu(phi)
        
        # First propagation through air
        for k in range(0, sample_current_position):
            phi_gpu = single_bpm_step_only_linear_medium(
                phi_gpu, domain.k_medium, domain.dz, domain.dx, domain.dy
            )
        
        # Propagation through sample
        for k in range(sample_current_position, sample_current_position + sample.thickness_units + 1):
            phi_gpu = single_bpm_step_within_sample(
                phi_gpu, sample.k, sample.n2, domain.dz, domain.dx, domain.dy, domain.eps
            )
        
        # Final propagation through air
        for k in range(sample_current_position + sample.thickness_units + 1, domain.Nz):
            phi_gpu = single_bpm_step_only_linear_medium(
                phi_gpu, domain.k_medium, domain.dz, domain.dx, domain.dy
            )
        
        # Transfer final result back to CPU
        phi = phi_gpu.get()
    else:
        # Fallback to CPU implementation
        for k in range(0, sample_current_position):
            phi = single_bpm_step_only_linear_medium(
                phi, domain.k_medium, domain.dz, domain.dx, domain.dy
            )
        
        for k in range(sample_current_position, sample_current_position + sample.thickness_units + 1):
            phi = single_bpm_step_within_sample(
                phi, sample.k, sample.n2, domain.dz, domain.dx, domain.dy, domain.eps
            )
        
        for k in range(sample_current_position + sample.thickness_units + 1, domain.Nz):
            phi = single_bpm_step_only_linear_medium(
                phi, domain.k_medium, domain.dz, domain.dx, domain.dy
            )
    
    return phi