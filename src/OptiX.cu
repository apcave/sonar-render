#include "ModelShared.h"
// #include "CudaUtils.cuh"

#include "OptiX/vec_math.h"
#include <optix.h>
#include <optix_device.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include <thrust/complex.h>

extern "C" __constant__ globalParams params;

/**
 * @Brief Mutex style lock for double precision floating point numbers.
 *        This is a workaround for the lack of atomicAdd for double precision
 *       floating point numbers in CUDA on older devices.
 */
__device__ double atomicAddDouble(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ void projectSourcePointToFragment(int xPnt, int yPnt)
{
    const int b = params.srcFacetNum;
    const dev_facet B = params.srcObject.facets[b];
    const int numXpntsNegative = B.frag_points.z;
    const float delta = params.frag_delta;
    const int ind_j = yPnt * B.frag_points.x + xPnt;

    // This is the x offset from the base point to the approximate centriod of the pixel.
    const float xoffset = delta * (xPnt - numXpntsNegative) + delta / 2; // This value can be negative.
    // This is the y offset from the base point to the approximate centriod of the pixel.
    const float yoffset = delta * yPnt + delta / 2;

    const float3 xAxis = B.xAxis * xoffset;
    const float3 yAxis = B.yAxis * yoffset;
    const float3 p_j = B.base_point + xAxis + yAxis;

    thrust::complex<double> i1 = thrust::complex<double>(0, 1);
    thrust::complex<double> k = thrust::complex<double>(params.k_wave.r, params.k_wave.i);

    int numSrc = params.srcPoints.numPnts;
    for (int i = 0; i < numSrc; i++)
    {
        float3 p_i = params.srcPoints.position[i];

        const float3 v_ij = p_j - p_i;
        const float r_ij = length(v_ij);
        const float3 uv_ij = v_ij / r_ij;
        const float epsilon = 1e-3f;

        unsigned int hit = 0;
        optixTrace(
            params.handle,
            p_i,
            uv_ij,
            0.0f,           // Min t
            r_ij - epsilon, // Max t
            0.0f,           // Ray time
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            0, // SBT offset
            1, // SBT stride
            0, // Miss SBT index
            hit);

        if (hit)
        {
            continue;
        }

        dcomplex cP_i = params.srcPoints.pressure[i];
        thrust::complex<double> P_i = thrust::complex<double>(cP_i.r, cP_i.i);

        thrust::complex<double> P_j = (P_i * thrust::exp(i1 * k * i1 * r_ij)) / r_ij;

        dcomplex *p_out = &(B.P[ind_j]);
        atomicAddDouble(&(p_out->r), P_j.real());
        atomicAddDouble(&(p_out->i), P_j.imag());
    }
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int i_x = idx.x;
    const unsigned int i_y = idx.y;

    switch (params.calcType)
    {
    case SOURCE_POINTS:
        // Call the source points kernel
        projectSourcePointToFragment(i_x, i_y);
        return;
    case FIELD_POINTS:
        // Project is done from the facet fragments to the field point
        return;
    case FACET:
        // Call the facet kernel
        return;
    case FACET_NO_RESP:
        // Call the facet no response kernel
        return;
        ;
    default:
        printf("Invalid calculation type\n");
        return;
    }
}

extern "C" __global__ void __closesthit__ch()
{
    // Mark the ray as hitting geometry
    optixSetPayload_0(1);
}

extern "C" __global__ void __miss__ms()
{
    // Mark the ray as not hitting any geometry
    optixSetPayload_0(0);
}