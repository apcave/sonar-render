#include "FacetCuda.hpp"
#include "dcomplex.h"
#include "CudaUtils.cuh"

#include <thrust/complex.h>

#include <cuda_runtime.h>
#include <stdexcept>

__device__ __always_inline float4 hsv2rgb(float h, float s, float v)
{
    float c = v * s;
    float x = c * (1.0 - abs(fmod(h * 6.0, 2.0) - 1.0));
    float m = v - c;
    float alpha = 1;

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

__global__ void MakeSurface(dcomplex *P, cudaSurfaceObject_t surface, int maxXpnt, float min_dB, float max_dB, bool render_phase)
{
    int xPnt = threadIdx.x;
    int yPnt = blockIdx.x;

    int index = yPnt * maxXpnt + xPnt;

    thrust::complex<float> p((float)P[index].r, (float)P[index].i);
    // float mag_normal = (abs(p) / stats[0]); // The largest magnitude is 0 zero.
    // float mag = 1 - exp(-mag_normal * 3.5); // The brightness is 0 to 1. Zero is black.

    const float saturation = 0.8f; // Saturation for the color.
    float dB = 20.0f * log10f(abs(p) + 1e-8f);
    float mag = (dB - min_dB) / (max_dB - min_dB);
    mag = fminf(fmaxf(mag, 0.0f), 1.0f);

    float4 value = make_float4(saturation * mag, saturation * mag, saturation * mag, 1.0f);
    if (render_phase)
    {
        // If we are rendering the phase, we need to convert the complex number to a color.
        // The phase is in the range [0, 1].
        float phase = (atan2f(p.imag(), p.real()) + M_PI) / (2.0f * M_PI);
        value = hsv2rgb(phase, 0.8f, mag);
    }

    // value = (float)yPnt / (float)maxXpnt;
    surf2Dwrite(value, surface, xPnt * sizeof(float4), yPnt);

    // printf("Write surface: %f, %f, %f\n", value, stats[0], stats[1]);
}

void FacetCuda::WriteSurface(float min_dB, float max_dB, bool render_phase)
{
    MapToCuda();
    // Copy the data to the device.
    // Make the texture out of the real values.
    dim3 threadsPerBlock(numXpnts, 1);
    dim3 numBlocks(numYpnts, 1);
    MakeSurface<<<numBlocks, threadsPerBlock>>>(dev_P, surface, numXpnts, min_dB, max_dB, render_phase);

    cudaGraphicsUnmapResources(1, &cudaResource, 0);
    readyToRender = true;
}

__global__ void AccumulatePressureKernel(dcomplex *P, dcomplex *dev_P_out, int maxXpnt)
{
    int xPnt = threadIdx.x;
    int yPnt = blockIdx.x;

    int index = yPnt * maxXpnt + xPnt;

    // P[index].r = P[index].r + dev_P_out[index].r;
    // P[index].i = P[index].i + dev_P_out[index].i;

    // printf("AccumulatePressureKernel: %d, %d, %f, %f\n", xPnt, yPnt, dev_P_out[index].r, dev_P_out[index].i);

    P[index].r = dev_P_out[index].r;
    P[index].i = dev_P_out[index].i;
}

void FacetCuda::AccumulatePressure()
{
    // Copy the data to the device.
    // Make the texture out of the real values.
    dim3 threadsPerBlock(numXpnts, 1);
    dim3 numBlocks(numYpnts, 1);

    if (dev_P == 0 || dev_P_out == 0)
    {
        throw std::runtime_error("Error: dev_P or dev_P_out is not allocated.\n");
    }

    AccumulatePressureKernel<<<numBlocks, threadsPerBlock>>>(dev_P, dev_P_out, numXpnts);
}