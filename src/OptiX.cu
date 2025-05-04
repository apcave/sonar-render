#include "ModelShared.h"
// #include "CudaUtils.cuh"

#include "OptiX/vec_math.h"
#include <optix.h>
#include <optix_device.h>

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

__device__ void projectSourcePointsToFacet(int b_ind)
{
    const dev_facet B = params.dstObject.facets[b_ind];
    const int numXpntsNegative = B.frag_points.z;
    const float delta = params.frag_delta;
    const int numSrc = params.srcPoints.numPnts;

    const thrust::complex<double> i1 = thrust::complex<double>(0, 1);
    const thrust::complex<double> k = thrust::complex<double>(params.k_wave.r, params.k_wave.i);

    for (int xPnt = 0; xPnt < B.frag_points.x; xPnt++)
    {
        for (int yPnt = 0; yPnt < B.frag_points.y; yPnt++)
        {
            const int ind_j = yPnt * B.frag_points.x + xPnt;
            // This is the x offset from the base point to the approximate centriod of the pixel.
            const float xoffset = delta * (xPnt - numXpntsNegative) + delta / 2; // This value can be negative.
            // This is the y offset from the base point to the approximate centriod of the pixel.
            const float yoffset = delta * yPnt + delta / 2;

            const float3 xAxis = B.xAxis * xoffset;
            const float3 yAxis = B.yAxis * yoffset;
            const float3 p_j = B.base_point + xAxis + yAxis;

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
                thrust::complex<double> P_j = (P_i * thrust::exp(i1 * k * r_ij)) / r_ij;

                dcomplex *p_out = &(B.P[ind_j]);
                atomicAddDouble(&(p_out->r), P_j.real());
                atomicAddDouble(&(p_out->i), P_j.imag());
            }
        }
    }
}

__device__ void projectFacetToFieldPoints(int a_ind)
{
    const dev_facet A = params.srcObject.facets[a_ind];
    const int numXpntsNegative = A.frag_points.z;
    const float delta = params.frag_delta;
    const int numDst = params.dstPoints.numPnts;
    const float epsilon = 1e-3f;

    const thrust::complex<double> i1 = thrust::complex<double>(0, 1);
    const thrust::complex<double> k = thrust::complex<double>(params.k_wave.r, params.k_wave.i);

    for (int xPnt = 0; xPnt < A.frag_points.x; xPnt++)
    {
        for (int yPnt = 0; yPnt < A.frag_points.y; yPnt++)
        {
            const int ind_i = yPnt * A.frag_points.x + xPnt;
            // This is the x offset from the base point to the approximate centriod of the pixel.
            const float xoffset = delta * (xPnt - numXpntsNegative) + delta / 2; // This value can be negative.
            // This is the y offset from the base point to the approximate centriod of the pixel.
            const float yoffset = delta * yPnt + delta / 2;

            const float3 xAxis = A.xAxis * xoffset;
            const float3 yAxis = A.yAxis * yoffset;
            const float3 p_i = A.base_point + xAxis + yAxis;
            const float A_i = A.frag_area[ind_i];

            if (A_i <= 1e-6f)
            {
                // printf("facets_PixelArea is zero\n");
                continue;
            }

            dcomplex cP_i = A.P[ind_i];
            thrust::complex<double> P_i = thrust::complex<double>(cP_i.r, cP_i.i);

            for (int j = 0; j < numDst; j++)
            {
                float3 p_j = params.dstPoints.position[j];

                const float3 v_ji = p_i - p_j;
                const float r_ji = length(v_ji);
                const float3 uv_ji = v_ji / r_ji;

                const float cos_inc = dot(-uv_ji, A.normal);
                if (cos_inc < 1e-6f)
                {
                    // printf("Normal doesn't align, not adding to field point.\n");
                    continue;
                }

                unsigned int hit = 0;
                optixTrace(
                    params.handle,
                    p_j,
                    uv_ji,
                    0.0f,           // Min t
                    r_ji - epsilon, // Max t
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

                thrust::complex<double> P_j = A_i * cos_inc * P_i * (-thrust::exp(i1 * k * r_ji) / ((4 * M_PI)) * ((i1 * k) / r_ji) + (1 / (r_ji * r_ji)));

                dcomplex *p_out = &(params.dstPoints.pressure[j]);
                atomicAddDouble(&(p_out->r), P_j.real());
                atomicAddDouble(&(p_out->i), P_j.imag());
            }
        }
    }
}

__device__ void projectFacetToFacets(int a_ind, bool useReciprocity, bool isSelfProject)
{
    const dev_facet A = params.srcObject.facets[a_ind];
    const int numXpntsNegative_i = A.frag_points.z;
    const float delta = params.frag_delta;
    const int numDst = params.dstObject.numFacets;
    const float epsilon = 1e-3f;

    const thrust::complex<double> i1 = thrust::complex<double>(0, 1);
    const thrust::complex<double> k = thrust::complex<double>(params.k_wave.r, params.k_wave.i);

    for (int xPnt_i = 0; xPnt_i < A.frag_points.x; xPnt_i++)
    {
        for (int yPnt_i = 0; yPnt_i < A.frag_points.y; yPnt_i++)
        {
            const int ind_i = yPnt_i * A.frag_points.x + xPnt_i;
            // This is the x offset from the base point to the approximate centriod of the pixel.
            const float xoffset_i = delta * (xPnt_i - numXpntsNegative_i) + delta / 2; // This value can be negative.
            // This is the y offset from the base point to the approximate centriod of the pixel.
            const float yoffset_i = delta * yPnt_i + delta / 2;

            const float3 xAxis_i = A.xAxis * xoffset_i;
            const float3 yAxis_i = A.yAxis * yoffset_i;
            const float3 p_i = A.base_point + xAxis_i + yAxis_i;
            const float A_i = A.frag_area[ind_i];

            if (A_i <= 1e-6f)
            {
                // printf("facets_PixelArea is zero\n");
                continue;
            }

            dcomplex cP_i = A.P[ind_i];
            thrust::complex<double> P_i = thrust::complex<double>(cP_i.r, cP_i.i);

            int startB_ind = 0;
            if (isSelfProject)
            {
                // To keep calculation even do not do a double projection when the two facets are the same.
                startB_ind = a_ind + 1;
            }

            // To keep calculation even do not do a double projection when the two facets are the same.
            for (int b_ind = startB_ind; b_ind < numDst; b_ind++)
            {

                dev_facet B = params.dstObject.facets[b_ind];
                const int numXpntsNegative_j = B.frag_points.z;

                for (int xPnt_j = 0; xPnt_j < B.frag_points.x; xPnt_j++)
                {
                    for (int yPnt_j = 0; yPnt_j < B.frag_points.y; yPnt_j++)
                    {
                        const int ind_j = yPnt_j * B.frag_points.x + xPnt_j;
                        // This is the x offset from the base point to the approximate centriod of the pixel.
                        const float xoffset_j = delta * (xPnt_j - numXpntsNegative_j) + delta / 2; // This value can be negative.
                        // This is the y offset from the base point to the approximate centriod of the pixel.
                        const float yoffset_j = delta * yPnt_j + delta / 2;

                        const float3 xAxis_j = B.xAxis * xoffset_j;
                        const float3 yAxis_j = B.yAxis * yoffset_j;
                        const float3 p_j = B.base_point + xAxis_j + yAxis_j;

                        const float3 v_ij = p_j - p_i;
                        const float r_ij = length(v_ij);
                        const float3 uv_ij = v_ij / r_ij;

                        const float B_j = B.frag_area[ind_j];
                        if (B_j <= 1e-6f)
                        {
                            // printf("facets_PixelArea is zero\n");
                            continue;
                        }

                        const float cos_inc = dot(uv_ij, A.normal);
                        if (cos_inc < 1e-6f)
                        {
                            // printf("Normal doesn't align, not adding to field point.\n");
                            continue;
                        }

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

                        thrust::complex<double> G = (-thrust::exp(i1 * k * r_ij) / ((4 * M_PI)) * ((i1 * k) / r_ij) + (1 / (r_ij * r_ij)));
                        thrust::complex<double> P_j = A_i * cos_inc * P_i * G;

                        dcomplex *p_out = &(B.P_out[ind_j]);
                        atomicAddDouble(&(p_out->r), P_j.real());
                        atomicAddDouble(&(p_out->i), P_j.imag());

                        if (useReciprocity)
                        {
                            // Reciprocity: project the pressure from facet B to facet A

                            const float rA_i = B.frag_area[ind_j];
                            dcomplex rcP_i = B.P[ind_j];
                            thrust::complex<double> rP_i = thrust::complex<double>(rcP_i.r, rcP_i.i);
                            float3 ruv_ij = -uv_ij; // Reverse the direction of the ray
                            const float rcos_inc = dot(ruv_ij, B.normal);
                            if (rcos_inc < 1e-6f)
                            {
                                // printf("Normal doesn't align, not adding to field point.\n");
                                continue;
                            }

                            thrust::complex<double> rP_j = rA_i * rcos_inc * rP_i * G;

                            dcomplex *rp_out = &(A.P_out[ind_i]);
                            atomicAddDouble(&(rp_out->r), rP_j.real());
                            atomicAddDouble(&(rp_out->i), rP_j.imag());
                        }
                    }
                }
            }
        }
    }
}

extern "C" __global__ void __raygen__rg()
{
    // printf("Raygen kernel called\n");

    const uint3 idx = optixGetLaunchIndex();
    const unsigned int facet_num = idx.x;

    switch (params.calcType)
    {
    case SOURCE_POINTS:
        // printf("Source points to facet projection\n");
        //  Call the source points kernel
        projectSourcePointsToFacet(facet_num);
        return;
    case FIELD_POINTS:
        // Project is done from the facet fragments to the field point
        // printf("Project facet to field points\n");
        projectFacetToFieldPoints(facet_num);
        return;
    case FACET_RESP:
        // Used for target object to other target object.
        projectFacetToFacets(facet_num, true, false);
        return;
    case FACET_NO_RESP:
        // No reciprocity is done. Used for target objects to field objects.
        projectFacetToFacets(facet_num, false, false);
        return;
    case FACET_SELF:
        // Accelerated self projection for target to target.
        // Uses reciprocity to half the number of calculations.
        projectFacetToFacets(facet_num, true, true);
        return;

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