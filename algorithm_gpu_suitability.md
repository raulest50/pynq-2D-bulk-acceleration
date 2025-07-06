# GPU Suitability Analysis for Deep Tissue Imaging Algorithm

## Why the Current Algorithm is Not Ideal for GPU Acceleration

The deep tissue imaging simulation uses an Alternating Direction Implicit (ADI) method with a Thomas algorithm for solving tridiagonal systems. This approach has several characteristics that make it suboptimal for GPU acceleration:

### 1. Inherently Sequential Thomas Algorithm

The Thomas algorithm (also known as the tridiagonal matrix algorithm) is a specialized form of Gaussian elimination for tridiagonal systems. It consists of two phases:
- Forward elimination (top to bottom)
- Back substitution (bottom to top)

Both phases have strict data dependencies where each step depends on the result of the previous step. This sequential nature fundamentally limits parallelization, which is the primary advantage of GPUs.

### 2. Row/Column-wise Processing in ADI Method

The ADI method processes the domain one row or column at a time:
- In the x-direction, each row is processed independently
- In the y-direction, each column is processed independently

While this allows for some parallelism (processing multiple rows/columns simultaneously), it doesn't fully utilize the massive parallelism available in modern GPUs, which can handle thousands of threads concurrently.

### 3. Low Arithmetic Intensity

The algorithm has a relatively low compute-to-memory ratio. Each operation in the Thomas algorithm performs only a few arithmetic operations per memory access, which doesn't leverage the GPU's computational capabilities effectively.

### 4. Memory Access Patterns

The original implementation had suboptimal memory access patterns, particularly in the column-wise operations where non-contiguous memory access reduces cache efficiency.

## Optimizations Implemented

Despite these limitations, we've implemented several optimizations to improve GPU performance:

### 1. Replaced Custom Thomas Solver with CuPy's Sparse Solver

We replaced the custom Thomas algorithm implementation with CuPy's optimized sparse matrix solver:
- Uses `cupyx.scipy.sparse.diags` to create a sparse tridiagonal matrix
- Uses `cupyx.scipy.sparse.linalg.spsolve` to solve the system
- Leverages highly optimized CUDA libraries for sparse linear algebra

### 2. Added Batch Processing for Rows/Columns

We implemented batch processing to handle multiple rows/columns in parallel:
- Processes rows/columns in batches of configurable size (default: 16)
- Synchronizes after each batch to manage memory usage
- Adjusts batch size based on available resources

### 3. Optimized Memory Access Patterns

We improved memory access patterns for better GPU performance:
- Replaced sequential loops with vectorized operations where possible
- Used `cp.roll` for efficient shifted array operations
- Ensured proper handling of boundary conditions

### 4. Kept Phase Mask Operations on CPU

Phase mask operations are kept on CPU as they:
- Are applied infrequently (only 3 times in 361 steps)
- Use SciPy's `gaussian_filter` which doesn't have a direct CuPy equivalent
- Involve random number generation which may be more efficient on CPU

## Conclusion

While our optimizations improve GPU performance for this algorithm, the fundamental limitations of the ADI method with Thomas algorithm remain. For truly optimal GPU performance, alternative algorithms with higher parallelism should be considered:

1. **Split-Step Fourier Method**: Uses FFT which is highly optimized on GPUs
2. **Explicit Finite Difference Schemes**: More parallelizable than implicit methods
3. **Parallel Cyclic Reduction (PCR)**: A parallel alternative to the Thomas algorithm

The current implementation represents a compromise between maintaining the numerical properties of the original algorithm and improving performance through GPU acceleration.




---

---


# PCR vs. Current Implementation: Analysis for Deep Tissue Imaging

## Would PCR (Parallel Cyclic Reduction) Be Better?

For your specific case with a 256×256 beam profile and the ADI algorithm, implementing PCR would likely **not provide significant benefits** compared to the current optimized implementation for several reasons:

### 1. Small Problem Size

Your 256×256 grid means each tridiagonal system is only 256 elements long. PCR is most beneficial for very large systems (thousands of elements) where its parallel nature can overcome its higher computational complexity. For your relatively small systems:

- PCR requires O(log₂n) steps with higher constant factors
- The current sparse solver is already optimized for small to medium-sized problems
- The overhead of implementing a custom PCR kernel would likely outweigh benefits

### 2. Current Optimizations Are Already Effective

Your current implementation already includes significant optimizations:

- Using CuPy's sparse solver instead of a custom Thomas algorithm
- Batch processing of rows/columns (16 at a time)
- Optimized memory access patterns with vectorized operations
- Proper GPU memory management

These optimizations already address many of the performance bottlenecks without requiring a complex PCR implementation.

### 3. Implementation Complexity

Implementing an efficient PCR algorithm on GPU requires:

- Custom CUDA kernel development
- Complex shared memory management
- Careful handling of boundary conditions
- Extensive testing to ensure numerical stability

This represents a significant development effort for potentially modest gains.

## Should You Revert to CPU for ADI Operations?

**Yes, for your specific case, a hybrid approach is likely optimal:**

### Recommended Hybrid Approach:

1. **Move ADI operations (Thomas solver) to CPU**:
   - The CPU implementation of the Thomas algorithm is simple and efficient
   - For 256×256 arrays, the CPU Thomas algorithm may outperform the GPU sparse solver
   - Eliminates the row-by-row processing bottleneck on GPU

2. **Keep halfsteps on GPU**:
   - The halfstep operations (`half_2photon_absorption`, `half_nonlinear`, `half_linear_absorption`) are perfectly parallelizable
   - They have high arithmetic intensity and no data dependencies
   - They operate on the entire array at once, fully utilizing GPU parallelism

3. **Minimize CPU-GPU transfers**:
   - Transfer the entire field to CPU for ADI operations
   - Transfer back to GPU for halfsteps
   - This adds some overhead but is likely less than the inefficiency of running ADI on GPU

### Implementation Suggestion:

```python
def full_step_within_tissue_hybrid(phi_gpu, tejido, d):
    # Convert to CPU for ADI operations
    phi_cpu = cp.asnumpy(phi_gpu)
    
    # ADI x-direction on CPU
    phi_cpu = so.adi_x(phi_cpu, d.Ny, d.eps, d.k, d.dz, d.dx)
    
    # Back to GPU for halfsteps
    phi_gpu = cp.asarray(phi_cpu)
    phi_gpu = so.half_2photon_absorption_gpu(phi_gpu, tejido.beta, d.dz)
    phi_gpu = so.half_nonlinear_gpu(phi_gpu, d.k, tejido.n2, d.dz)
    phi_gpu = so.half_linear_absorption_gpu(phi_gpu, tejido.alpha, d.dz)
    
    # Back to CPU for ADI y-direction
    phi_cpu = cp.asnumpy(phi_gpu)
    phi_cpu = so.adi_y(phi_cpu, d.Nx, d.eps, d.k, d.dz, d.dy)
    
    # Back to GPU for final halfsteps
    phi_gpu = cp.asarray(phi_cpu)
    phi_gpu = so.half_2photon_absorption_gpu(phi_gpu, tejido.beta, d.dz)
    phi_gpu = so.half_nonlinear_gpu(phi_gpu, d.k, tejido.n2, d.dz)
    phi_gpu = so.half_linear_absorption_gpu(phi_gpu, tejido.alpha, d.dz)
    
    return phi_gpu
```

## Conclusion

For your specific case with a 256×256 beam profile:

1. **PCR implementation is likely not worth the effort** - the potential speedup doesn't justify the development complexity
2. **A hybrid CPU-GPU approach is recommended** - use CPU for ADI operations and GPU for halfsteps
3. **Benchmark both approaches** - test the hybrid approach against your current implementation to confirm performance gains

This hybrid approach leverages the strengths of both CPU (for sequential ADI operations) and GPU (for parallel halfstep operations) while avoiding the development complexity of implementing PCR.



---


---


(tesis-py3.12) PS C:\Users\raule\OneDrive\Desktop\tesis_final\pynq-2D-bulk-acceleration> python .\deep_tissue_imaging_gpu.py
=== Deep Tissue Imaging: GPU Implementation ===
CuPy is available. GPU acceleration is enabled.
Number of GPU devices: 1

Device 0: Quadro P600
  Compute Capability: 6.1
  Total Memory: 2.00 GB
  CUDA Cores: 3

Setting up domain parameters...
Creating initial field...
Creating phase mask manager...

Preparing GPU-optimized parameters...
Warming up GPU...
GPU warm-up complete

Running GPU simulation with memory optimization...
Initializing phase masks...
Initializing phase masks...
  Initializing mask 1...
  Initializing mask 2...
  Initializing mask 3...
aplicada mascara aleatoria 1 en z = 120
aplicada mascara aleatoria 2 en z = 240
aplicada mascara aleatoria 3 en z = 360
GPU execution time: 1170.635373 seconds

Preparing results for visualization...

Measuring PSF parameters (lateral FWHM only)...

=== PSF Parameters ===
FWHM Lateral: nan µm

Radius containing 80% of energy: 45.40 µm

Encircled Energy:
  Radius nan µm: 0.0%
  Radius nan µm: 0.0%
  Radius nan µm: 0.0%

Max Sidelobe Level: 99.3% of peak

