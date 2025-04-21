#include "CudaModelTes.cuh"
#include "dcomplex.h"
#include "CudaUtils.cuh"

#include <stdio.h>

__global__ void ProjectFacetToFieldPointKernel(
    dcomplex *k_wave,
    float *pixel_delta,
    int field_point_num,
    int facet_num,
    float3 *field_points_position,
    dcomplex *field_points_pressure,
    float3 *facet_normals,
    int3 *facet_Points,
    float3 *base_points,
    float3 *facets_xaxis,
    float3 *facets_yaxis,
    float **facets_PixelArea,
    double *Pr_facet,
    double *Pi_facet)
{
    dcomplex k = *k_wave;
    float delta = *pixel_delta;

    // printf("k_wave: %f, %f\n", k.r, k.i);
    // printf("pixel_delta: %f\n", delta);

    // Kernel code to project point to point
    // printf("ThreadIdx.x: %d, ThreadIdx.y: %d, blockIdx.x: %d, blockDim.x: %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockDim.x);
    int xPnt = threadIdx.x;
    int yPnt = blockIdx.x;

    int NumXpnts = facet_Points[facet_num].x;
    // int NumYpnts = facet_Points[facet_num].y;
    int NumXpntsNegative = facet_Points[facet_num].z;

    int index_Ai = yPnt * NumXpnts + xPnt;

    float A_i = facets_PixelArea[facet_num][index_Ai];
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

    float3 xAxis = facets_xaxis[facet_num];
    float3 yAxis = facets_yaxis[facet_num];

    xAxis.x = xoffset * xAxis.x;
    xAxis.y = xoffset * xAxis.y;
    xAxis.z = xoffset * xAxis.z;

    yAxis.x = yoffset * yAxis.x;
    yAxis.y = yoffset * yAxis.y;
    yAxis.z = yoffset * yAxis.z;

    float3 facet_base = base_points[facet_num];
    float3 r_i;
    r_i.x = xAxis.x + yAxis.x + facet_base.x;
    r_i.y = xAxis.y + yAxis.y + facet_base.y;
    r_i.z = xAxis.z + yAxis.z + facet_base.z;
    // printf("Facet Point Global Ref: %f, %f, %f\n", P1g.x, P1g.y, P1g.z);

    dcomplex p_inc;
    p_inc.r = Pr_facet[index_Ai];
    p_inc.i = Pi_facet[index_Ai];

    float3 vr_ri = MakeVector(r_i, r);
    float lr_i = GetVectorLength(vr_ri);
    float3 ur_ri = DivideVector(vr_ri, lr_i);

    float3 n = facet_normals[facet_num];
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
    dcomplex var = devRCmul(lr_i, ik);
    var = devCexp(var); // This has phase and attenuation.
    var = devCmul(var, p_inc);
    // var = devCmul(var, ik);

    // printf("var: %e, %e\n", var.r, var.i);
    double spread = sqrt(A_i / (2 * PI));

    double realTerms = (A_i * sc) / lr_i;

    var = devRCmul(realTerms, var);

    if (devCabs(var) > 1.0)
    {
        printf("Pressure is too large to add to field point.\n");
        printf("lr_i: %f\n", lr_i);
        printf("source_pressure: %e, %e\n", p_inc.r, p_inc.i);
        printf("Spherical spread: %e\n", realTerms);
        printf("Pressure to field point prior to spreading: %e, %e\n", var.r, var.i);
        return;
    }

    // Save the pressure to the facet pressure array.
    // Note var may be small and accumulate over may projects that why the complex numbers are doubles.
    atomicAddDouble(&(field_points_pressure[field_point_num].r), var.r);
    atomicAddDouble(&(field_points_pressure[field_point_num].i), var.i);
}

int CudaModelTes::ProjectFromFacetsToFieldPoints()
{
    printf("ProjectFromFacetsToFieldPoints .......\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Query device 0
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

    for (int object_num = 0; object_num < host_object_num_facets.size(); object_num++)
    {

        for (int facet_num = 0; facet_num < host_object_num_facets[object_num]; facet_num++)
        {

            for (int field_point_num = 0; field_point_num < host_num_field_points; field_point_num++)
            {

                int3 h_Facets_points = host_Object_Facets_points[object_num][facet_num];

                dim3 threadsPerBlock(h_Facets_points.x, 1);
                dim3 numBlocks(h_Facets_points.y, 1);

                // printf("ThreadsPerBlock.x: %d, threadsPerBlock.y: %d\n", threadsPerBlock.x, threadsPerBlock.y);
                // printf("numBlocks.x: %d, numBlocks.y: %d\n", numBlocks.x, numBlocks.y);

                ProjectFacetToFieldPointKernel<<<numBlocks, threadsPerBlock>>>(
                    dev_k_wave,
                    dev_pixel_delta,
                    field_point_num,
                    facet_num,
                    dev_field_points_position,
                    dev_field_points_pressure,
                    dev_Object_normals[object_num],
                    dev_Object_Facets_points[object_num],
                    dev_Object_base_points[object_num],
                    dev_Object_Facets_xAxis[object_num],
                    dev_Object_Facets_yAxis[object_num],
                    dev_Object_Facets_PixelArea[object_num],
                    dev_object_facet_Pr[object_num][facet_num],
                    dev_object_facet_Pi[object_num][facet_num]);

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
                    return 1;
                }
            }
        }
    }
    // More testing is required on large models to see how CUDA manages the cores.
    cudaDeviceSynchronize();
    printf("ProjectFromFacetsToFieldPoints done.\n");
    return 0;
}
