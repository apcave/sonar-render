#include <cuda_runtime.h>

extern __device__ double atomicAddDouble(double *address, double val);
extern __device__ float atomicMaxFloat(float *address, float val);
extern __device__ float atomicMinFloat(float *address, float val);