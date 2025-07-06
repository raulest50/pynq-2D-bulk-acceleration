# GPU Testing for Deep Tissue Imaging

This directory contains scripts to test GPU functionality for deep tissue imaging applications, with a focus on complex64 data type support.

## Requirements

To use the GPU testing scripts, you need:

1. **NVIDIA GPU** with CUDA support
2. **CuPy** package installed with the correct CUDA version:
   ```
   pip install cupy-cuda12x  # Replace with your CUDA version
   ```

## Files

- `cupy_gpu_test.py`: Comprehensive script to test GPU functionality with complex64 data using CuPy
- `simple_gpu_test.py`: Minimal script to verify GPU functionality with complex64 data using CuPy

## Why GPU Initially Appeared Slower Than CPU

When first testing GPU acceleration, you might encounter situations where the GPU appears slower than the CPU. This happened in our initial tests with the NVIDIA Quadro P600 GPU. Here's why this occurred and how it was fixed:

### Initial Problem
```
CPU time: 2.4185 seconds
GPU time: 24.8598 seconds
Speedup: 0.10x (GPU was 10x slower!)
```

### Reasons for Poor Initial Performance

1. **Overhead Dominated Small Workloads**:
   - Random data generation on the GPU added overhead
   - Memory transfers between CPU and GPU were not optimized
   - JIT compilation of CUDA kernels added first-run penalty
   - Small workload didn't fully utilize GPU parallelism

2. **Inefficient Test Design**:
   - Test included time for data generation (not part of actual computation)
   - Only a single FFT operation was performed (GPUs excel at repeated operations)
   - Data size was too small to overcome GPU initialization overhead

3. **No GPU Warm-up**:
   - First CUDA operations are slower due to JIT compilation
   - Device initialization adds overhead

### How It Was Fixed

After implementing optimizations in `cupy_gpu_test.py`:
```
CPU time: 0.5505 seconds
GPU time: 0.0195 seconds
Speedup: 28.26x (GPU now 28x faster!)
```

The key optimizations included:
- Separating data preparation from computation timing
- Adding GPU warm-up runs to eliminate JIT compilation overhead
- Running multiple iterations for more accurate benchmarking
- Implementing batch processing to leverage GPU parallelism
- Testing with larger datasets to better utilize GPU resources

## Performance Considerations for GPU Testing

1. **Memory Transfers**: The most significant bottleneck in GPU acceleration is often the transfer of data between CPU and GPU memory. The cupy_gpu_test.py script minimizes these transfers by keeping data on the GPU during computation.

2. **JIT Compilation Overhead**: The first run of any CUDA kernel includes compilation time. The test script uses warm-up runs to eliminate this overhead from timing measurements.

3. **Synchronization**: Proper GPU synchronization is essential for accurate timing. The test script uses `cp.cuda.Stream.null.synchronize()` to ensure all GPU operations are complete before stopping the timer.

4. **Small Problem Sizes**: For small problem sizes, the overhead of GPU initialization and memory transfers might outweigh the benefits of parallel processing. This is why the test script includes tests with different data sizes.

## GPU Testing

To verify that your GPU is working correctly and can handle complex64 data (important for deep tissue imaging), you can use the provided test scripts:

### Simple GPU Test

The `simple_gpu_test.py` script provides a minimal test to verify GPU functionality:

```bash
python GPU_xploration/simple_gpu_test.py
```

This script:
1. Checks if CuPy can access your GPU
2. Displays basic information about your GPU
3. Performs a simple FFT operation on complex64 data
4. Compares performance between CPU and GPU

### Comprehensive GPU Test

For a more detailed test, use the `cupy_gpu_test.py` script:

```bash
python GPU_xploration/cupy_gpu_test.py
```

This script provides:
1. Detailed GPU information (device name, memory, CUDA cores, etc.)
2. Comprehensive testing of complex64 operations using FFT
3. Multiple optimization strategies testing:
   - Standard single-array processing
   - Batch processing (multiple arrays)
   - Large dataset processing
4. Performance comparison between CPU and GPU for each strategy
5. Verification that results are consistent between CPU and GPU
6. Specific recommendations based on your GPU's performance

The improved test script includes:
- Separation of data preparation from computation timing
- GPU warm-up runs to eliminate JIT compilation overhead
- Multiple iterations for more accurate benchmarking
- Proper CUDA synchronization
- Batch processing tests to demonstrate parallel workload advantages
- Large dataset tests to demonstrate memory-intensive workload advantages

### Expected Output

If your GPU is working correctly, you should see:
- Information about your GPU (NVIDIA Quadro P600)
- Successful execution of complex64 operations
- Performance metrics for different optimization strategies
- Specific recommendations for your GPU

For the Quadro P600 (entry-level workstation GPU), you might see:
- Limited speedup or even slowdown for small, single-array operations
- Better performance with batch processing (multiple arrays)
- Better performance with larger datasets (4K x 4K or larger)

### Troubleshooting

If you encounter errors:
- Make sure CuPy is installed with the correct CUDA version: `pip install cupy-cuda12x` (replace with your CUDA version)
- Verify that your NVIDIA drivers are up to date
- Check that your CUDA installation is correct
- Ensure Visual Studio with C++ support is installed (for Windows)

## Optimization Strategies for Deep Tissue Imaging

Based on the performance tests with the Quadro P600 GPU, here are specific optimization strategies for deep tissue imaging applications:

### 1. Batch Processing

Process multiple images or image slices simultaneously:

```python
# Instead of processing one image at a time:
for image in images:
    result = process_single_image(image)

# Process multiple images in parallel:
batch_results = process_image_batch(images)
```

Implementation tips:
- Group similar-sized images together
- Keep batch size reasonable (8-16 for Quadro P600)
- Consider memory constraints (2GB for Quadro P600)

### 2. Increase Dataset Size

For single image processing, use larger arrays when possible:

```python
# Increase resolution or field of view
# For example, use 4096x4096 instead of 2048x2048
large_image = acquire_high_resolution_image()
result = process_single_image(large_image)
```

Implementation tips:
- Monitor memory usage to avoid out-of-memory errors
- Consider tiling for extremely large images

### 3. Minimize CPU-GPU Transfers

Keep data on the GPU as long as possible:

```python
# Bad: Multiple transfers
gpu_data = cp.asarray(cpu_data)
result1 = gpu_operation1(gpu_data)
cpu_result1 = cp.asnumpy(result1)  # Unnecessary transfer
result2 = gpu_operation2(cp.asarray(cpu_result1))  # Unnecessary transfer

# Good: Keep data on GPU
gpu_data = cp.asarray(cpu_data)
result1 = gpu_operation1(gpu_data)
result2 = gpu_operation2(result1)  # No transfer needed
cpu_final_result = cp.asnumpy(result2)  # Transfer only once
```

### 4. Use GPU Warm-up

Eliminate JIT compilation overhead:

```python
# Warm up GPU with a small example before timing
_ = gpu_function(small_test_data)
cp.cuda.Stream.null.synchronize()

# Now run the actual computation
start_time = time.time()
result = gpu_function(real_data)
cp.cuda.Stream.null.synchronize()
elapsed = time.time() - start_time
```

### 5. Optimize FFT Operations

For deep tissue imaging with FFT operations:

```python
# Use CuPy's FFT plan to optimize repeated FFTs
fft_plan = cp.cuda.cufft.Plan2d(height, width, cp.complex64)
result = cp.empty_like(data)
cp.cuda.cufft.fft2(data, result, fft_plan)
```

### 6. Use Mixed Precision

Consider using lower precision for non-critical calculations:

```python
# Use float32 instead of float64 where possible
data_f32 = data.astype(cp.float32)
result = gpu_function(data_f32)
```

## Further Optimizations

The current implementation could be further optimized by:

1. **Shared Memory**: Using shared memory for frequently accessed data
2. **Stream Processing**: Using CUDA streams for concurrent execution
3. **Custom Kernels**: Developing more specialized CUDA kernels for specific operations
4. **Memory Prefetching**: Overlapping computation with memory transfers
