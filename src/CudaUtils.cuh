#include <cuda_runtime.h>

extern __device__ double atomicAddDouble(double *address, double val);
extern __device__ float atomicMaxFloat(float *address, float val);
extern __device__ float atomicMinFloat(float *address, float val);
extern __device__ float3 MakeVector(float3 Origin, float3 Dest);
extern __device__ float3 DivideVector(float3 vect, float variable);
extern __device__ float DotProduct(float3 v1, float3 v2);
extern __device__ float GetVectorLength(float3 v1);
extern __device__ float3 subtract(const float3 &a, const float3 &b);