import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""
__global__ void double_it(float *a) {
  int idx = threadIdx.x;
  a[idx] *= 2;
}
""")
fn = mod.get_function("double_it")
a = np.random.rand(256).astype(np.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)
fn(a_gpu, block=(256,1,1), grid=(1,1))
cuda.memcpy_dtoh(a, a_gpu)
print(a)  # Valores duplicados :contentReference[oaicite:5]{index=5}
