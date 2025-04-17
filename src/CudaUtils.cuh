#include <cuda_runtime.h>

extern __device__ double atomicAddDouble(double *address, double val);