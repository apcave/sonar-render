#include "FacetCuda.hpp"
#include "dcomplex.h"
#include "CudaUtils.cuh"

#include <thrust/complex.h>

#include <cuda_runtime.h>

__device__ __always_inline float4 hsv2rgb(float h, float s, float v)
{
    float c = v * s;
    float x = c * (1.0 - abs(fmod(h * 6.0, 2.0) - 1.0));
    float m = v - c;
    float alpha = 1.0;

    if (h < 1.0 / 6.0)
    {
        return make_float4(c + m, x + m, 0.0 + m, alpha);
    }
    else if (h < 2.0 / 6.0)
    {
        return make_float4(x + m, c + m, 0.0 + m, alpha);
    }
    else if (h < 3.0 / 6.0)
    {
        return make_float4(0.0 + m, c + m, x + m, alpha);
    }
    else if (h < 4.0 / 6.0)
    {
        return make_float4(0.0 + m, x + m, c + m, alpha);
    }
    else if (h < 5.0 / 6.0)
    {
        return make_float4(x + m, 0.0 + m, c + m, alpha);
    }
    else
    {
        return make_float4(c + m, 0.0 + m, x + m, alpha);
    }
}

__global__ void MakeSurface(dcomplex *P, cudaSurfaceObject_t surface, int maxXpnt, float *stats)
{
    int xPnt = threadIdx.x;
    int yPnt = blockIdx.x;

    int index = yPnt * maxXpnt + xPnt;

    thrust::complex<float> p((float)P[index].r, (float)P[index].i);
    float mag_normal = (abs(p) / stats[0]); // The largest magnitude is 0 zero.
    float mag = 1 - exp(-mag_normal * 3.5); // The brightness is 0 to 1. Zero is black.
    float phase = (atan2(p.imag(), p.real()) + M_PI) / (2.0f * M_PI);

    float4 value = hsv2rgb(phase, 0.8, mag);

    // value = (float)yPnt / (float)maxXpnt;
    surf2Dwrite(value, surface, xPnt * sizeof(float4), yPnt);

    // printf("Write surface: %f, %f, %f\n", value, stats[0], stats[1]);
}

void FacetCuda::WriteSurface(float *dev_frag_stats)
{
    MapToCuda();
    // Copy the data to the device.
    // Make the texture out of the real values.
    dim3 threadsPerBlock(numXpnts, 1);
    dim3 numBlocks(numYpnts, 1);
    MakeSurface<<<numBlocks, threadsPerBlock>>>(dev_P, surface, numXpnts, dev_frag_stats);

    cudaGraphicsUnmapResources(1, &cudaResource, 0);
    readyToRender = true;
}

__global__ void GetMaxValue(dcomplex *P, int maxXpnt, float *stats)
{
    int xPnt = threadIdx.x;
    int yPnt = blockIdx.x;

    int index = yPnt * maxXpnt + xPnt;

    thrust::complex<float> p((float)P[index].r, (float)P[index].i);

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

    GetMaxValue<<<numBlocks, threadsPerBlock>>>(dev_P,
                                                numXpnts,
                                                dev_frag_stats);
}

__global__ void AccumulatePressureKernel(dcomplex *P, dcomplex *dev_P_out, int maxXpnt)
{
    int xPnt = threadIdx.x;
    int yPnt = blockIdx.x;

    int index = yPnt * maxXpnt + xPnt;

    P[index].r = P[index].r + dev_P_out[index].r;
    P[index].i = P[index].i + dev_P_out[index].i;
}

void FacetCuda::AccumulatePressure()
{
    // Copy the data to the device.
    // Make the texture out of the real values.
    dim3 threadsPerBlock(numXpnts, 1);
    dim3 numBlocks(numYpnts, 1);
    AccumulatePressureKernel<<<numBlocks, threadsPerBlock>>>(dev_P, dev_P_out, numXpnts);
}