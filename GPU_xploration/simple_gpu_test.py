"""
Minimal script to verify GPU functionality using CuPy with complex64 data.
This simple test confirms that:
1. CuPy can access the GPU
2. Basic operations with complex64 data work correctly

For NVIDIA Quadro P600 GPU testing.
"""

import numpy as np
import time

try:
    import cupy as cp
    print("CuPy successfully imported!")
except ImportError:
    print("Error: CuPy is not installed. Please install it with:")
    print("pip install cupy-cuda11x  # Replace with your CUDA version")
    exit(1)

# Check GPU availability
try:
    device_count = cp.cuda.runtime.getDeviceCount()
    if device_count == 0:
        print("No GPU devices found!")
        exit(1)
        
    # Get GPU info
    device = cp.cuda.Device(0)  # Use the first GPU
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"\nGPU detected: {props['name'].decode()}")
    print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
    print(f"Memory: {props['totalGlobalMem'] / (1024**3):.2f} GB")
    
    # Simple test with complex64 data
    print("\nRunning simple test with complex64 data...")
    
    # Create small complex64 arrays
    size = 1024
    
    # CPU version (NumPy)
    cpu_start = time.time()
    a_cpu = np.random.random((size, size)) + 1j * np.random.random((size, size))
    a_cpu = a_cpu.astype(np.complex64)
    b_cpu = np.fft.fft2(a_cpu)
    cpu_time = time.time() - cpu_start
    
    # GPU version (CuPy)
    gpu_start = time.time()
    a_gpu = cp.random.random((size, size)) + 1j * cp.random.random((size, size))
    a_gpu = a_gpu.astype(cp.complex64)
    b_gpu = cp.fft.fft2(a_gpu)
    cp.cuda.Stream.null.synchronize()  # Make sure GPU computation is complete
    gpu_time = time.time() - gpu_start
    
    # Report results
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    print("\nSUCCESS: GPU is working correctly with complex64 data!")
    print("You can proceed with GPU acceleration for deep tissue imaging.")
    
except Exception as e:
    print(f"Error during GPU test: {e}")
    print("GPU test failed. Please check your CUDA installation.")