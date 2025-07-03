import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.cumath import exp
import warnings

# Flag to check if CUDA is available
CUDA_AVAILABLE = True
try:
    cuda.Device(0)  # Try to access the first CUDA device
except:
    CUDA_AVAILABLE = False
    warnings.warn("CUDA device not found. Falling back to CPU implementation.")

# Import original CPU implementations as fallback
from zscan_custom.operadores_solvers import (
    custom_thomas_solver as cpu_custom_thomas_solver,
    compute_b_vector as cpu_compute_b_vector,
    adi_x as cpu_adi_x,
    adi_y as cpu_adi_y,
    adi_x_wa as cpu_adi_x_wa,
    adi_y_wa as cpu_adi_y_wa,
    half_nonlinear as cpu_half_nonlinear,
    single_bpm_step_within_sample as cpu_single_bpm_step_within_sample,
    single_bpm_step_only_linear_medium as cpu_single_bpm_step_only_linear_medium,
    single_bpm_step_only_linear_medium_wa as cpu_single_bpm_step_only_linear_medium_wa
)

# CUDA kernel for Thomas algorithm
thomas_solver_kernel = """
extern "C" {
    __global__ void thomas_forward_elimination(
        cuDoubleComplex* c_prime, cuDoubleComplex* d_prime, 
        double dp, double dp1, double dp2, double do_val,
        cuDoubleComplex* b, int n) {
        
        // First row
        c_prime[0] = make_cuDoubleComplex(do_val / dp1, 0);
        d_prime[0] = make_cuDoubleComplex(b[0].x / dp1, b[0].y / dp1);
        
        // Middle rows
        for (int i = 1; i < n-1; i++) {
            double denominator_real = dp - do_val * c_prime[i-1].x;
            double denominator_imag = -do_val * c_prime[i-1].y;
            double denom_squared = denominator_real * denominator_real + denominator_imag * denominator_imag;
            
            // c_prime[i] = do_val / denominator
            c_prime[i] = make_cuDoubleComplex(
                (do_val * denominator_real) / denom_squared,
                (-do_val * denominator_imag) / denom_squared
            );
            
            // d_prime[i] = (b[i] - do_val * d_prime[i-1]) / denominator
            cuDoubleComplex temp;
            temp.x = b[i].x - do_val * d_prime[i-1].x;
            temp.y = b[i].y - do_val * d_prime[i-1].y;
            
            d_prime[i] = make_cuDoubleComplex(
                (temp.x * denominator_real + temp.y * denominator_imag) / denom_squared,
                (temp.y * denominator_real - temp.x * denominator_imag) / denom_squared
            );
        }
        
        // Last row
        cuDoubleComplex temp;
        temp.x = b[n-1].x - do_val * d_prime[n-2].x;
        temp.y = b[n-1].y - do_val * d_prime[n-2].y;
        
        double denominator_real = dp2 - do_val * c_prime[n-2].x;
        double denominator_imag = -do_val * c_prime[n-2].y;
        double denom_squared = denominator_real * denominator_real + denominator_imag * denominator_imag;
        
        d_prime[n-1] = make_cuDoubleComplex(
            (temp.x * denominator_real + temp.y * denominator_imag) / denom_squared,
            (temp.y * denominator_real - temp.x * denominator_imag) / denom_squared
        );
    }
    
    __global__ void thomas_back_substitution(
        cuDoubleComplex* x, cuDoubleComplex* c_prime, cuDoubleComplex* d_prime, int n) {
        
        x[n-1] = d_prime[n-1];
        
        for (int i = n-2; i >= 0; i--) {
            x[i].x = d_prime[i].x - c_prime[i].x * x[i+1].x + c_prime[i].y * x[i+1].y;
            x[i].y = d_prime[i].y - c_prime[i].x * x[i+1].y - c_prime[i].y * x[i+1].x;
        }
    }
    
    __global__ void compute_b_vector_kernel(
        cuDoubleComplex* b, double dp, double dp1, double dp2, double do_val,
        cuDoubleComplex* x0, int n) {
        
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (i == 0) {
            // First row
            b[0].x = dp1 * x0[0].x + do_val * x0[1].x;
            b[0].y = dp1 * x0[0].y + do_val * x0[1].y;
        }
        else if (i == n-1) {
            // Last row
            b[n-1].x = do_val * x0[n-2].x + dp2 * x0[n-1].x;
            b[n-1].y = do_val * x0[n-2].y + dp2 * x0[n-1].y;
        }
        else if (i < n) {
            // Middle rows
            b[i].x = do_val * x0[i-1].x + dp * x0[i].x + do_val * x0[i+1].x;
            b[i].y = do_val * x0[i-1].y + dp * x0[i].y + do_val * x0[i+1].y;
        }
    }
    
    __global__ void half_nonlinear_kernel(
        cuDoubleComplex* phi_out, cuDoubleComplex* phi_in, 
        double k_sample, double n2_sample, double dz, int size) {
        
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (i < size) {
            double abs_phi_squared = phi_in[i].x * phi_in[i].x + phi_in[i].y * phi_in[i].y;
            double phase = k_sample * n2_sample * dz/2 * abs_phi_squared;
            
            double cos_phase = cos(phase);
            double sin_phase = sin(phase);
            
            phi_out[i].x = cos_phase * phi_in[i].x - sin_phase * phi_in[i].y;
            phi_out[i].y = sin_phase * phi_in[i].x + cos_phase * phi_in[i].y;
        }
    }
}
"""

# Compile CUDA kernels if CUDA is available
if CUDA_AVAILABLE:
    try:
        mod = SourceModule(thomas_solver_kernel)
        thomas_forward_elimination = mod.get_function("thomas_forward_elimination")
        thomas_back_substitution = mod.get_function("thomas_back_substitution")
        compute_b_vector_kernel = mod.get_function("compute_b_vector_kernel")
        half_nonlinear_kernel = mod.get_function("half_nonlinear_kernel")
    except Exception as e:
        CUDA_AVAILABLE = False
        warnings.warn(f"Failed to compile CUDA kernels: {e}. Falling back to CPU implementation.")

def custom_thomas_solver(dp, dp1, dp2, do, b):
    """
    GPU-accelerated version of the Thomas algorithm for tridiagonal systems.
    Falls back to CPU implementation if CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        return cpu_custom_thomas_solver(dp, dp1, dp2, do, b)
    
    n = len(b)
    
    # Allocate memory on GPU
    c_prime_gpu = gpuarray.zeros(n-1, dtype=np.complex128)
    d_prime_gpu = gpuarray.zeros(n, dtype=np.complex128)
    b_gpu = gpuarray.to_gpu(b)
    x_gpu = gpuarray.zeros(n, dtype=np.complex128)
    
    # Forward elimination
    thomas_forward_elimination(
        c_prime_gpu, d_prime_gpu, 
        np.float64(dp), np.float64(dp1), np.float64(dp2), np.float64(do),
        b_gpu, np.int32(n),
        block=(1, 1, 1), grid=(1, 1)
    )
    
    # Back substitution
    thomas_back_substitution(
        x_gpu, c_prime_gpu, d_prime_gpu, np.int32(n),
        block=(1, 1, 1), grid=(1, 1)
    )
    
    # Transfer result back to CPU
    return x_gpu.get()

def compute_b_vector(dp, dp1, dp2, do, x0):
    """
    GPU-accelerated version of compute_b_vector.
    Falls back to CPU implementation if CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        return cpu_compute_b_vector(dp, dp1, dp2, do, x0)
    
    n = len(x0)
    
    # Allocate memory on GPU
    x0_gpu = gpuarray.to_gpu(x0)
    b_gpu = gpuarray.zeros(n, dtype=np.complex128)
    
    # Calculate block and grid dimensions
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    
    # Execute kernel
    compute_b_vector_kernel(
        b_gpu, np.float64(dp), np.float64(dp1), np.float64(dp2), np.float64(do),
        x0_gpu, np.int32(n),
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )
    
    # Transfer result back to CPU
    return b_gpu.get()

def adi_x(phi, Ny, eps, k, dz, dx):
    """
    GPU-accelerated version of adi_x.
    Falls back to CPU implementation if CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        return cpu_adi_x(phi, Ny, eps, k, dz, dx)
    
    ung = 1j * dz / (4 * k * dx**2)
    phi_inter = np.zeros_like(phi, dtype=complex)
    
    # Process each row on CPU for now
    # This could be further optimized to process multiple rows in parallel on GPU
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
    """
    GPU-accelerated version of adi_y.
    Falls back to CPU implementation if CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        return cpu_adi_y(phi, Nx, eps, k, dz, dy)
    
    ung = 1j * dz / (4 * k * dy**2)
    phi_inter = np.zeros_like(phi, dtype=complex)
    
    # Process each column on CPU for now
    # This could be further optimized to process multiple columns in parallel on GPU
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

def adi_x_wa(phi, Ny, eps, k, dz, dx, alpha):
    """
    GPU-accelerated version of adi_x_wa.
    Falls back to CPU implementation if CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        return cpu_adi_x_wa(phi, Ny, eps, k, dz, dx, alpha)
    
    ung = 1j * dz / (4 * k * dx**2)
    a = dz * alpha / 8
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
        
        dp1_B = -2 * ung + 1 + ung * ratio_x0 - a
        dp2_B = -2 * ung + 1 + ung * ratio_xn - a
        dp_B = -2 * ung + 1 - a
        do_B = ung
        
        b = compute_b_vector(dp_B, dp1_B, dp2_B, do_B, phi[:, j])
        
        dp1_A = 2 * ung + 1 - ung * ratio_x0 + a
        dp2_A = 2 * ung + 1 - ung * ratio_xn + a
        dp_A = 2 * ung + 1 + a
        do_A = -ung
        
        phi_inter[:, j] = custom_thomas_solver(dp_A, dp1_A, dp2_A, do_A, b)
    
    return phi_inter

def adi_y_wa(phi, Nx, eps, k, dz, dy, alpha):
    """
    GPU-accelerated version of adi_y_wa.
    Falls back to CPU implementation if CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        return cpu_adi_y_wa(phi, Nx, eps, k, dz, dy, alpha)
    
    ung = 1j * dz / (4 * k * dy**2)
    a = dz * alpha / 8
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
        
        dp1_B = -2 * ung + 1 + ung * ratio_y0 - a
        dp2_B = -2 * ung + 1 + ung * ratio_yn - a
        dp_B = -2 * ung + 1 - a
        do_B = ung
        
        b = compute_b_vector(dp_B, dp1_B, dp2_B, do_B, phi[i, :])
        
        dp1_A = 2 * ung + 1 - ung * ratio_y0 + a
        dp2_A = 2 * ung + 1 - ung * ratio_yn + a
        dp_A = 2 * ung + 1 + a
        do_A = -ung
        
        phi_inter[i, :] = custom_thomas_solver(dp_A, dp1_A, dp2_A, do_A, b)
    
    return phi_inter

def half_nonlinear(phi, k_sample, n2_sample, dz):
    """
    GPU-accelerated version of half_nonlinear.
    Falls back to CPU implementation if CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        return cpu_half_nonlinear(phi, k_sample, n2_sample, dz)
    
    # Reshape phi to 1D array for GPU processing
    original_shape = phi.shape
    phi_flat = phi.flatten()
    size = len(phi_flat)
    
    # Allocate memory on GPU
    phi_in_gpu = gpuarray.to_gpu(phi_flat)
    phi_out_gpu = gpuarray.zeros(size, dtype=np.complex128)
    
    # Calculate block and grid dimensions
    block_size = 256
    grid_size = (size + block_size - 1) // block_size
    
    # Execute kernel
    half_nonlinear_kernel(
        phi_out_gpu, phi_in_gpu,
        np.float64(k_sample), np.float64(n2_sample), np.float64(dz), np.int32(size),
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )
    
    # Transfer result back to CPU and reshape
    return phi_out_gpu.get().reshape(original_shape)

def single_bpm_step_within_sample(phi, k_sample, n2_sample, dz, dx, dy, eps=1e-12):
    """
    GPU-accelerated version of single_bpm_step_within_sample.
    Falls back to CPU implementation if CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        return cpu_single_bpm_step_within_sample(phi, k_sample, n2_sample, dz, dx, dy, eps)
    
    Ny, Nx = phi.shape
    phi_inter = adi_x(phi, Ny, eps, k_sample, dz, dx)
    phi_inter = half_nonlinear(phi_inter, k_sample, n2_sample, dz)
    phi_inter = adi_y(phi_inter, Nx, eps, k_sample, dz, dy)
    phi_inter = half_nonlinear(phi_inter, k_sample, n2_sample, dz)
    return phi_inter

def single_bpm_step_only_linear_medium(phi, k_medium, dz, dx, dy, eps=1e-12):
    """
    GPU-accelerated version of single_bpm_step_only_linear_medium.
    Falls back to CPU implementation if CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        return cpu_single_bpm_step_only_linear_medium(phi, k_medium, dz, dx, dy, eps)
    
    Ny, Nx = phi.shape
    phi_inter = adi_x(phi, Ny, eps, k_medium, dz, dx)
    phi_out = adi_y(phi_inter, Nx, eps, k_medium, dz, dy)
    return phi_out

def single_bpm_step_only_linear_medium_wa(phi, k_medium, dz, dx, dy, eps, alpha):
    """
    GPU-accelerated version of single_bpm_step_only_linear_medium_wa.
    Falls back to CPU implementation if CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        return cpu_single_bpm_step_only_linear_medium_wa(phi, k_medium, dz, dx, dy, eps, alpha)
    
    Ny, Nx = phi.shape
    phi_inter = adi_x_wa(phi, Ny, eps, k_medium, dz, dx, alpha)
    phi_out = adi_y_wa(phi_inter, Nx, eps, k_medium, dz, dy, alpha)
    return phi_out