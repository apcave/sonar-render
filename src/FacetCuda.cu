#include "FacetCuda.hpp"
#include "dcomplex.h"
#include "CudaUtils.cuh"

#include <thrust/complex.h>

#include <cuda_runtime.h>

__global__ void MakeSurface(double *Pr, double *Pi, cudaSurfaceObject_t surface, int maxXpnt, float *stats)
{
    int xPnt = threadIdx.x;
    int yPnt = blockIdx.x;

    int index = yPnt * maxXpnt + xPnt;

    thrust::complex<float> p((float)Pr[index], (float)Pi[index]);

    float mag = abs(p) / stats[0];
    float phase = (atan2(p.imag(), p.real()) + M_PI) / (2.0f * M_PI);
    float2 value = make_float2(mag, phase);

    // value = (float)yPnt / (float)maxXpnt;
    surf2Dwrite(value, surface, xPnt * sizeof(float2), yPnt);

    // printf("Write surface: %f, %f, %f\n", value, stats[0], stats[1]);
}

void FacetCuda::WriteSurface(float *dev_frag_stats)
{
    MapToCuda();
    // Copy the data to the device.
    // Make the texture out of the real values.
    dim3 threadsPerBlock(numXpnts, 1);
    dim3 numBlocks(numYpnts, 1);
    MakeSurface<<<numBlocks, threadsPerBlock>>>(dev_Pr, dev_Pi, surface, numXpnts, dev_frag_stats);

    cudaGraphicsUnmapResources(1, &cudaResource, 0);
    readyToRender = true;
}

__global__ void GetMaxValue(double *Pr, double *Pi, int maxXpnt, float *stats)
{
    int xPnt = threadIdx.x;
    int yPnt = blockIdx.x;

    int index = yPnt * maxXpnt + xPnt;

    thrust::complex<float> p((float)Pr[index], (float)Pi[index]);

    float mag = abs(p);

    atomicMaxFloat(&stats[0], mag);
    // atomicMinFloat(&stats[1], real);
    // atomicMaxFloat(&stats[2], abs);

    // printf("GetMaxValue: %e, %e, %e\n", stats[0], stats[1], stats[2]);
    //  printf("CurVal: %e, %e, %e\n", real, imag, abs);
}

void FacetCuda::GetSurfaceScalers(float *dev_frag_stats)
{
    dim3 threadsPerBlock(numXpnts, 1);
    dim3 numBlocks(numYpnts, 1);

    GetMaxValue<<<numBlocks, threadsPerBlock>>>(dev_Pr,
                                                dev_Pi,
                                                numXpnts,
                                                dev_frag_stats);
}