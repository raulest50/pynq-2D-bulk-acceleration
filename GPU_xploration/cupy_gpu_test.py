"""
Simple script to test GPU functionality using CuPy with complex64 data type.
This script verifies that:
1. CuPy can access the GPU
2. Complex64 operations work correctly
3. There's a performance improvement compared to CPU

For NVIDIA Quadro P600 GPU acceleration of deep tissue imaging.
"""

import numpy as np
import time
import sys

# Try importing CuPy and handle import errors gracefully
try:
    import cupy as cp
    print("CuPy successfully imported!")
except ImportError:
    print("Error: CuPy is not installed. Please install it with:")
    print("pip install cupy-cuda11x  # Replace with your CUDA version")
    sys.exit(1)

def print_gpu_info():
    """Print information about available GPU devices."""
    try:
        print("\n=== GPU Information ===")
        print(f"CuPy version: {cp.__version__}")
        print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")

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
        print("\nGPU is working correctly!")
        return True
    except Exception as e:
        print(f"Error accessing GPU information: {e}")
        return False

def test_complex64_operations():
    """Test complex64 operations on both CPU and GPU."""
    print("\n=== Testing Complex64 Operations ===")

    # Create complex64 test data
    size = 2048
    iterations = 10  # Number of iterations for better benchmarking
    print(f"Creating {size}x{size} complex64 arrays...")

    # Generate data on CPU first (not timed)
    print("Generating test data...")
    cpu_data = np.random.random((size, size)) + 1j * np.random.random((size, size))
    cpu_data = cpu_data.astype(np.complex64)

    # Transfer data to GPU (not timed)
    gpu_data = cp.asarray(cpu_data)

    # Warm-up run for GPU (not timed) to eliminate JIT compilation overhead
    print("Performing GPU warm-up...")
    _ = cp.fft.fft2(gpu_data)
    _ = cp.abs(_)**2
    _ = cp.fft.ifft2(_)
    cp.cuda.Stream.null.synchronize()  # Ensure warm-up is complete

    print("Running benchmarks...")

    # CPU (NumPy) implementation - timed
    cpu_times = []
    for i in range(iterations):
        cpu_start = time.time()
        # Perform FFT (common in deep tissue imaging)
        cpu_fft = np.fft.fft2(cpu_data)
        cpu_power = np.abs(cpu_fft)**2
        cpu_result = np.fft.ifft2(cpu_power)
        cpu_times.append(time.time() - cpu_start)

    cpu_time = sum(cpu_times) / iterations

    # GPU (CuPy) implementation - timed
    gpu_times = []
    for i in range(iterations):
        gpu_start = time.time()
        # Perform the same operations on GPU
        gpu_fft = cp.fft.fft2(gpu_data)
        gpu_power = cp.abs(gpu_fft)**2
        gpu_result = cp.fft.ifft2(gpu_power)
        cp.cuda.Stream.null.synchronize()  # Proper synchronization
        gpu_times.append(time.time() - gpu_start)

    gpu_time = sum(gpu_times) / iterations

    # Report results
    print(f"CPU time (average of {iterations} runs): {cpu_time:.4f} seconds")
    print(f"GPU time (average of {iterations} runs): {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")

    # Verify results are similar (not exact due to floating point differences)
    # Transfer a small portion to CPU for comparison
    gpu_sample = cp.asnumpy(gpu_result[:10, :10])
    cpu_sample = cpu_result[:10, :10]

    # Check if results are close enough
    max_diff = np.max(np.abs(gpu_sample - cpu_sample))
    print(f"Maximum difference between CPU and GPU results: {max_diff}")

    if max_diff < 1e-5:
        print("Results match! Complex64 operations working correctly on GPU.")
        return True
    else:
        print("Warning: Results differ significantly between CPU and GPU.")
        return False

def test_batch_processing():
    """Test batch processing to demonstrate GPU's strength with parallel workloads."""
    print("\n=== Testing Batch Processing ===")

    # Create smaller arrays but process multiple at once
    size = 1024  # Smaller size for batch processing
    batch_size = 8  # Number of arrays to process simultaneously
    iterations = 5  # Number of iterations for benchmarking

    print(f"Creating {batch_size} arrays of size {size}x{size}...")

    # Generate batch data on CPU (not timed)
    print("Generating batch data...")
    cpu_batch_data = []
    for i in range(batch_size):
        data = np.random.random((size, size)) + 1j * np.random.random((size, size))
        cpu_batch_data.append(data.astype(np.complex64))

    # Transfer batch data to GPU (not timed)
    gpu_batch_data = [cp.asarray(data) for data in cpu_batch_data]

    # Warm-up run for GPU
    print("Performing GPU warm-up...")
    for data in gpu_batch_data:
        _ = cp.fft.fft2(data)
        _ = cp.abs(_)**2
        _ = cp.fft.ifft2(_)
    cp.cuda.Stream.null.synchronize()

    print("Running batch processing benchmarks...")

    # CPU batch processing (sequential)
    cpu_batch_times = []
    for i in range(iterations):
        cpu_start = time.time()
        cpu_results = []
        for data in cpu_batch_data:
            fft = np.fft.fft2(data)
            power = np.abs(fft)**2
            result = np.fft.ifft2(power)
            cpu_results.append(result)
        cpu_batch_times.append(time.time() - cpu_start)

    cpu_batch_time = sum(cpu_batch_times) / iterations

    # GPU batch processing
    gpu_batch_times = []
    for i in range(iterations):
        gpu_start = time.time()
        gpu_results = []
        for data in gpu_batch_data:
            fft = cp.fft.fft2(data)
            power = cp.abs(fft)**2
            result = cp.fft.ifft2(power)
            gpu_results.append(result)
        cp.cuda.Stream.null.synchronize()
        gpu_batch_times.append(time.time() - gpu_start)

    gpu_batch_time = sum(gpu_batch_times) / iterations

    # Report batch processing results
    print(f"CPU batch time (average of {iterations} runs): {cpu_batch_time:.4f} seconds")
    print(f"GPU batch time (average of {iterations} runs): {gpu_batch_time:.4f} seconds")
    print(f"Batch processing speedup: {cpu_batch_time/gpu_batch_time:.2f}x")

    return cpu_batch_time/gpu_batch_time > 1.0  # Return True if GPU is faster

def test_large_dataset():
    """Test performance with a large dataset to demonstrate GPU's strength with large workloads."""
    print("\n=== Testing Large Dataset Performance ===")

    # Use a larger size to better utilize GPU parallelism
    # Only if system has enough memory
    try:
        size = 4096  # 4K x 4K array (16M elements)
        iterations = 3  # Fewer iterations due to larger dataset

        print(f"Creating large {size}x{size} complex64 array...")

        # Generate data on CPU (not timed)
        print("Generating large test data...")
        cpu_data = np.random.random((size, size)) + 1j * np.random.random((size, size))
        cpu_data = cpu_data.astype(np.complex64)

        # Transfer to GPU (not timed)
        gpu_data = cp.asarray(cpu_data)

        # Warm-up run
        print("Performing GPU warm-up...")
        _ = cp.fft.fft2(gpu_data)
        cp.cuda.Stream.null.synchronize()

        print("Running large dataset benchmarks...")

        # CPU implementation
        cpu_times = []
        for i in range(iterations):
            cpu_start = time.time()
            cpu_fft = np.fft.fft2(cpu_data)
            cpu_power = np.abs(cpu_fft)**2
            cpu_result = np.fft.ifft2(cpu_power)
            cpu_times.append(time.time() - cpu_start)

        cpu_time = sum(cpu_times) / iterations

        # GPU implementation
        gpu_times = []
        for i in range(iterations):
            gpu_start = time.time()
            gpu_fft = cp.fft.fft2(gpu_data)
            gpu_power = cp.abs(gpu_fft)**2
            gpu_result = cp.fft.ifft2(gpu_power)
            cp.cuda.Stream.null.synchronize()
            gpu_times.append(time.time() - gpu_start)

        gpu_time = sum(gpu_times) / iterations

        # Report results
        print(f"CPU time for large dataset: {cpu_time:.4f} seconds")
        print(f"GPU time for large dataset: {gpu_time:.4f} seconds")
        print(f"Large dataset speedup: {cpu_time/gpu_time:.2f}x")

        return cpu_time/gpu_time > 1.0  # Return True if GPU is faster

    except Exception as e:
        print(f"Error during large dataset test: {e}")
        print("Skipping large dataset test due to memory constraints or other issues.")
        return False

def main():
    """Main function to run all GPU tests."""
    print("=== GPU Functionality Test ===")

    # Check GPU information
    gpu_available = print_gpu_info()
    if not gpu_available:
        print("Failed to access GPU. Please check your CUDA installation.")
        return

    # Test complex64 operations
    complex_ops_success = test_complex64_operations()

    # Test batch processing
    batch_processing_success = test_batch_processing()

    # Test large dataset performance
    large_dataset_success = test_large_dataset()

    # Final summary
    print("\n=== Test Summary ===")
    if gpu_available and complex_ops_success:
        print("SUCCESS: GPU is working correctly with complex64 data type!")

        # Report on optimization strategies
        print("\nOptimization Strategy Results:")

        if batch_processing_success:
            print("✓ Batch processing shows significant GPU performance advantage.")
        else:
            print("✗ Batch processing did not show GPU advantage.")

        if large_dataset_success:
            print("✓ Large dataset processing shows significant GPU performance advantage.")
        else:
            print("✗ Large dataset processing did not show GPU advantage or was skipped.")

        print("\nRecommendations:")
        if batch_processing_success or large_dataset_success:
            print("- Your GPU works best with either batch processing or large datasets.")
            print("- For best performance, structure your code to process multiple arrays at once or use larger arrays.")
            print("- Consider using memory optimization techniques for large datasets.")
        else:
            print("- Your GPU (Quadro P600) is an entry-level workstation GPU.")
            print("- For this GPU, you may see limited performance gains with small workloads.")
            print("- Try increasing batch size or dataset size further for better GPU utilization.")

        print("\nYou can proceed with GPU acceleration for deep tissue imaging.")
    else:
        print("Some tests failed. Please check the output above for details.")

if __name__ == "__main__":
    main()
