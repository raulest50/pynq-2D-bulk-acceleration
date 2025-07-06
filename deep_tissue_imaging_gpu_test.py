import numpy as np
import time

from deep_tissue_imaging.elementos.lasers import fuente_microscopia_1 as laser, campo_tem00
from deep_tissue_imaging.elementos.plotting import plot_field_intensity, plot_field_intensity_history
from deep_tissue_imaging.elementos.tejidos import cerebro_emb_pez_cebra as tejido
import deep_tissue_imaging.propagators.propagation as prop_cpu
import deep_tissue_imaging_gpu.propagators.propagation as prop_gpu
import deep_tissue_imaging.elementos.domain as Domain
from benchmark.phase_mask_manager import PhaseMaskManager
from benchmark.medir_psf_params import medir_psf_params

def run_comparison():
    """
    Run a comparison between CPU and GPU implementations of deep tissue imaging.
    """
    print("Setting up domain parameters...")
    # Domain parameters
    Lz = np.float32(361e-6)  # 361um
    Nz = 361
    dz = np.float32(Lz / Nz)  # 1um

    Lx, Ly = np.float32(45e-6), np.float32(45e-6)  # 45um x 45um
    Nx, Ny = 256, 256
    dx = np.float32(Lx / Nx)  # 0.35um
    dy = np.float32(Ly / Ny)  # 0.35um

    x = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
    y = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    k0 = np.float32(2*np.pi / laser.wavelength)
    k = np.float32(k0 * tejido.n_0)
    sigma_phi = np.float32(k * tejido.Dn * tejido.l_s)
    sigma_x = np.float32(5e-6)  # Typical value for brain tissue (5 μm)

    domain = Domain.Domain(X, Y, Nx, Ny, Nz, dx, dy, dz, np.float32(1e-12), k0, k, sigma_phi, sigma_x)

    # Create initial field
    print("Creating initial field...")
    phi0 = campo_tem00(X, Y, laser.w0, laser.I_peak)

    # Create a phase mask manager
    print("Creating phase mask manager...")
    mask_manager = PhaseMaskManager(save_dir="./phase_masks")

    # Run GPU simulation
    print("\nRunning GPU simulation...")
    start_time = time.time()
    phi_history_gpu = prop_gpu.full_propagation_within_tissue_gpu(phi0, tejido, domain, mask_manager=mask_manager)
    # Ensure all GPU operations are complete
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start_time
    print(f"GPU execution time: {gpu_time:.6f} seconds")

    # Run CPU simulation
    print("\nRunning CPU simulation...")
    start_time = time.time()
    phi_history_cpu = prop_cpu.full_propagation_within_tissue(phi0, tejido, domain, mask_manager=mask_manager)
    cpu_time = time.time() - start_time
    print(f"CPU execution time: {cpu_time:.6f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")

    # Convert GPU results to CPU for comparison and visualization
    print("\nComparing results...")
    phi_history_gpu_cpu = cp.asnumpy(phi_history_gpu)

    # Compare results
    max_diff = np.max(np.abs(phi_history_cpu - phi_history_gpu_cpu))
    mean_diff = np.mean(np.abs(phi_history_cpu - phi_history_gpu_cpu))
    print(f"Maximum absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")

    # Measure PSF parameters for GPU results
    print("\nMeasuring PSF parameters for GPU results...")
    z_positions = np.linspace(0, Lz, Nz+1)
    focal_plane_gpu = phi_history_gpu_cpu[-1]  # Last slice is the focal plane
    psf_params_gpu = medir_psf_params(focal_plane_gpu, X, Y, phi_history_gpu_cpu, z_positions, plot=True)

    # Plot field intensity history for GPU results
    print("\nPlotting field intensity history for GPU results...")
    plot_field_intensity_history(phi_history_gpu_cpu, X, Y)

    return phi_history_cpu, phi_history_gpu_cpu, cpu_time, gpu_time

if __name__ == "__main__":
    print("=== Deep Tissue Imaging: CPU vs GPU Comparison ===")
    
    # Check if CuPy is available
    try:
        import cupy as cp
        print("CuPy is available. GPU acceleration is enabled.")
        
        # Print GPU information
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"Number of GPU devices: {device_count}")
        
        for device_id in range(device_count):
            cp.cuda.Device(device_id).use()
            device_props = cp.cuda.runtime.getDeviceProperties(device_id)
            print(f"\nDevice {device_id}: {device_props['name'].decode()}")
            print(f"  Compute Capability: {device_props['major']}.{device_props['minor']}")
            print(f"  Total Memory: {device_props['totalGlobalMem'] / (1024**3):.2f} GB")
            print(f"  CUDA Cores: {device_props['multiProcessorCount']}")
        
        # Set back to device 0 for the rest of the script
        cp.cuda.Device(0).use()
        
    except ImportError:
        print("CuPy is not available. Please install it with:")
        print("pip install cupy-cuda12x  # Replace with your CUDA version")
        exit(1)
    
    # Run the comparison
    phi_history_cpu, phi_history_gpu, cpu_time, gpu_time = run_comparison()
    
    print("\n=== Summary ===")
    print(f"CPU execution time: {cpu_time:.6f} seconds")
    print(f"GPU execution time: {gpu_time:.6f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    if cpu_time > gpu_time:
        print("\nGPU acceleration was successful!")
        print("The GPU implementation is faster than the CPU implementation.")
    else:
        print("\nGPU acceleration was not successful.")
        print("The CPU implementation is faster than the GPU implementation.")
        print("This could be due to the overhead of transferring data between CPU and GPU for the phase mask application.")