#include "ModelCuda.hpp"
#include "dcomplex.h"
#include "CudaUtils.cuh"

#include <stdio.h>

__global__ void ProjectFacetToFieldPointKernel(
    dcomplex *k_wave,
    float *frag_delta,
    int field_point_num,
    float3 *field_points_position,
    dcomplex *field_points_pressure,
    dev_facet *facet_data,
    float *frag_area,
    dcomplex *P_facet)
{
    dcomplex k = *k_wave;
    float delta = *frag_delta;

    // printf("k_wave: %f, %f\n", k.r, k.i);
    // printf("pixel_delta: %f\n", delta);

    // Kernel code to project point to point
    // printf("ThreadIdx.x: %d, ThreadIdx.y: %d, blockIdx.x: %d, blockDim.x: %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockDim.x);
    int xPnt = threadIdx.x;
    int yPnt = blockIdx.x;

    int NumXpnts = facet_data->frag_points.x;
    // int NumYpnts = facet_Points[facet_num].y;
    int NumXpntsNegative = facet_data->frag_points.z;

    int index = yPnt * NumXpnts + xPnt;

    float A_i = frag_area[index];
    if (A_i == 0)
    {
        // printf("facets_PixelArea is zero\n");
        return;
    }

    // printf("xPnt: %d, yPnt: %d\n", xPnt, yPnt);
    // printf("NumXpnts: %d, NumYpnts: %d, NumXpntsNegative: %d\n", NumXpnts, NumYpnts, NumXpntsNegative);

    float3 r = field_points_position[field_point_num];
    // printf("Field Point: %f, %f, %f\n", P2g.x, P2g.y, P2g.z);

    // This is the x offset from the base point to the approximate centriod of the pixel.
    float xoffset = delta * (xPnt - NumXpntsNegative) + delta / 2; // This value can be negative.
    // This is the y offset from the base point to the approximate centriod of the pixel.
    float yoffset = delta * yPnt + delta / 2;

    float3 xAxis = facet_data->xAxis;
    float3 yAxis = facet_data->yAxis;

    xAxis.x = xoffset * xAxis.x;
    xAxis.y = xoffset * xAxis.y;
    xAxis.z = xoffset * xAxis.z;

    yAxis.x = yoffset * yAxis.x;
    yAxis.y = yoffset * yAxis.y;
    yAxis.z = yoffset * yAxis.z;

    float3 facet_base = facet_data->base_point;
    float3 r_i;
    r_i.x = xAxis.x + yAxis.x + facet_base.x;
    r_i.y = xAxis.y + yAxis.y + facet_base.y;
    r_i.z = xAxis.z + yAxis.z + facet_base.z;
    // printf("Facet Point Global Ref: %f, %f, %f\n", P1g.x, P1g.y, P1g.z);

    dcomplex p_inc = P_facet[index];

    float3 vr_ri = MakeVector(r_i, r);
    float r_if = GetVectorLength(vr_ri);
    float3 ur_ri = DivideVector(vr_ri, r_if);

    float3 n = facet_data->normal;
    double sc = (double)DotProduct(ur_ri, n);

    if (sc < 0)
    {
        // printf("Normal doesn't align, not adding to field point.\n");
        return;
    }

    // printf("Distance from pixel to field point: %f\n", r_sf);

    // P2 = P2*exp(-i*k*r_sf)
    dcomplex i = devComplex(0, 1);
    dcomplex ik = devCmul(i, k);
    dcomplex exp_ikr = devRCmul(r_if, ik);
    exp_ikr = devCexp(exp_ikr); // This has phase and attenuation.

    double p = -1 / (4 * PI);
    dcomplex Gt = devRCmul(p, exp_ikr);

    dcomplex dG_dr_a = devRCmul(1 / r_if, ik); // First term.
    double dG_dr_b = 1 / (r_if * r_if);        // Second term often ignored useful for near field.
    dcomplex dG_dr = dG_dr_a;
    dG_dr.r += dG_dr_b;
    dG_dr = devCmul(dG_dr, Gt); // This is the derivative of the Greens function.

    // printf("var: %e, %e\n", var.r, var.i);
    dcomplex result = devRCmul(A_i * sc, dG_dr); // Area term for the integral.
    result = devCmul(result, p_inc);             // This includes the original pressure.

    if (devCabs(result) > 1.0)
    {
        printf("Pressure is too large to add to field point.\n");
        printf("lr_i: %f\n", r_if);
        printf("source_pressure: %e, %e\n", p_inc.r, p_inc.i);
        // printf("Spherical spread: %e\n", realTerms);
        printf("Pressure to field point prior to spreading: %e, %e\n", result.r, result.i);
        return;
    }

    // Save the pressure to the facet pressure array.
    // Note var may be small and accumulate over may projects that why the complex numbers are doubles.
    atomicAddDouble(&(field_points_pressure[field_point_num].r), result.r);
    atomicAddDouble(&(field_points_pressure[field_point_num].i), result.i);
}

int ModelCuda::ProjectFromFacetsToFieldPoints()
{
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0); // Query device 0
    // printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

    for (auto object : targetObjects)
    {
        int numSrc = object->facets.size();
        for (int srcCnt = 0; numSrc > srcCnt; srcCnt++)
        {

            auto facet = object->facets[srcCnt];

            for (int dstCnt = 0; dstCnt < host_num_field_points; dstCnt++)
            {
                dim3 threadsPerBlock(facet->frag_points.x, 1);
                dim3 numBlocks(facet->frag_points.y, 1);

                // printf("ThreadsPerBlock.x: %d, threadsPerBlock.y: %d\n", threadsPerBlock.x, threadsPerBlock.y);
                // printf("numBlocks.x: %d, numBlocks.y: %d\n", numBlocks.x, numBlocks.y);

                // printf("Projecting facet %d to field point %d.\n", srcCnt, dstCnt);

                ProjectFacetToFieldPointKernel<<<numBlocks, threadsPerBlock>>>(
                    dev_k_wave,
                    dev_frag_delta,
                    dstCnt,
                    dev_field_points_position,
                    dev_field_points_pressure,
                    facet->dev_data,
                    facet->dev_frag_area,
                    facet->dev_P);

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("ProjectFromFacetsToFieldPoints, Kernel launch failed: %s\n", cudaGetErrorString(err));
                    return 1;
                }
            }
        }
        // object->PrintSurfacePressure();
    }
    // More testing is required on large models to see how CUDA manages the cores.
    cudaDeviceSynchronize();
    return 0;
}
