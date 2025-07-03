# GPU-Accelerated Z-scan Simulation

This directory contains GPU-accelerated implementations of the Z-scan simulation code, along with examples of how to use them.

## Requirements

To use the GPU-accelerated implementations, you need:

1. **NVIDIA GPU** with CUDA support
2. **PyCUDA** package installed:
   ```
   pip install pycuda
   ```

## Files

- `operadores_solvers_gpu.py`: GPU-accelerated version of the operators and solvers
- `propagations_gpu.py`: GPU-accelerated version of the propagation functions
- `example_gpu_usage.py`: Example script demonstrating how to use the GPU-accelerated implementations

## How to Use

### 1. Import from GPU-accelerated modules

Instead of importing from the original modules, import from the GPU-accelerated ones:

```python
# Original imports
from zscan_custom.operadores_solvers import single_bpm_step_only_linear_medium
from zscan_custom.propagations import full_propagation_without_sample

# Replace with GPU-accelerated imports
from zscan_custom.operadores_solvers_gpu import single_bpm_step_only_linear_medium
from zscan_custom.propagations_gpu import full_propagation_without_sample
```

### 2. Use the functions as you would with the CPU versions

The GPU-accelerated functions have the same signatures as the original functions, making them drop-in replacements:

```python
# This will use GPU acceleration if available
phi, phi_history = full_propagation_without_sample(Phi0, domain)
```

### 3. Automatic fallback to CPU

If a CUDA-capable GPU is not available, the code automatically falls back to the CPU implementation:

```python
# This will use GPU if available, otherwise CPU
phi = single_bpm_step_only_linear_medium(phi, domain.k_medium, domain.dz, domain.dx, domain.dy)
```

## Example

See `example_gpu_usage.py` for a complete example of how to use the GPU-accelerated implementations.

## Performance Comparison

To compare CPU vs GPU performance, you can run the same simulation with both implementations:

```python
import time
from zscan_custom.propagations import full_propagation_without_sample as cpu_full_propagation_without_sample
from zscan_custom.propagations_gpu import full_propagation_without_sample as gpu_full_propagation_without_sample

# CPU implementation
start_time = time.time()
phi_cpu, phi_history_cpu = cpu_full_propagation_without_sample(Phi0, domain)
cpu_time = time.time() - start_time

# GPU implementation
start_time = time.time()
phi_gpu, phi_history_gpu = gpu_full_propagation_without_sample(Phi0, domain)
gpu_time = time.time() - start_time

# Calculate speedup
speedup = cpu_time / gpu_time
print(f"CPU time: {cpu_time:.2f} s")
print(f"GPU time: {gpu_time:.2f} s")
print(f"Speedup: {speedup:.2f}x")
```

## Performance Considerations

1. **Memory Transfers**: The most significant bottleneck in GPU acceleration is often the transfer of data between CPU and GPU memory. The implementation tries to minimize these transfers.

2. **Thomas Algorithm**: The Thomas algorithm is inherently sequential, which limits its parallelization. However, we can still benefit from GPU acceleration by processing multiple independent systems in parallel.

3. **Matrix Operations**: The matrix operations in the ADI method are well-suited for GPU acceleration.

4. **Small Problem Sizes**: For small problem sizes, the overhead of GPU initialization and memory transfers might outweigh the benefits of parallel processing.

## Further Optimizations

The current implementation could be further optimized by:

1. **Batch Processing**: Processing multiple rows/columns in parallel on the GPU
2. **Shared Memory**: Using shared memory for frequently accessed data
3. **Stream Processing**: Using CUDA streams for concurrent execution
4. **Custom Kernels**: Developing more specialized CUDA kernels for specific operations