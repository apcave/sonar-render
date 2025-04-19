#include "CudaModelTes.cuh"
#include "GeoMath.h"
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
    int yPnt = threadIdx.y;

    int NumXpnts = facet_Points[facet_num].x;
    // int NumYpnts = facet_Points[facet_num].y;
    int NumXpntsNegative = facet_Points[facet_num].z;

    int index_Ai = yPnt * NumXpnts + xPnt;

    float pixel_area = facets_PixelArea[facet_num][index_Ai];
    if (pixel_area == 0)
    {
        // printf("facets_PixelArea is zero\n");
        return;
    }

    // printf("xPnt: %d, yPnt: %d\n", xPnt, yPnt);
    // printf("NumXpnts: %d, NumYpnts: %d, NumXpntsNegative: %d\n", NumXpnts, NumYpnts, NumXpntsNegative);

    float3 P2g = field_points_position[field_point_num];
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
    float3 P1g;
    P1g.x = xAxis.x + yAxis.x + facet_base.x;
    P1g.y = xAxis.y + yAxis.y + facet_base.y;
    P1g.z = xAxis.z + yAxis.z + facet_base.z;
    // printf("Facet Point Global Ref: %f, %f, %f\n", P1g.x, P1g.y, P1g.z);

    dcomplex source_pressure;
    source_pressure.r = Pr_facet[index_Ai];
    source_pressure.i = Pi_facet[index_Ai];

    // The distance from the source point to the facet point.
    float r_sf = sqrtf((P1g.x - P2g.x) * (P1g.x - P2g.x) + (P1g.y - P2g.y) * (P1g.y - P2g.y) + (P1g.z - P2g.z) * (P1g.z - P2g.z));

    // printf("Distance from pixel to field point: %f\n", r_sf);

    // P2 = P2*exp(-i*k*r_sf)
    dcomplex i = devComplex(0, 1);
    dcomplex var = devCmul(i, k);
    var = devRCmul(r_sf, var);
    var = devCexp(var);                  // This has phase and attenuation.
    var = devCmul(var, source_pressure); // This includes the orginal pressure.
    // printf("Pressure add to field point prior to spreading: %f, %f\n", var.r, var.i);

    // Area1 = Pressure the 1Pa over 1m^2
    // Area2 = 4 * PI * r_sf * r_sf
    // atten_spread = Area1 / Area2 <--- important for other projections.
    float att_spread = pow(pixel_area / (4 * M_PI * r_sf * r_sf), 0.5);
    var = devRCmul(att_spread, var);
    // printf("Spherical spread: %f\n", att_spread);

    if (devCabs(var) > 1.0)
    {
        // printf("Pressure is too large to add to field point.\n");
        // printf("r_sf: %f\n", r_sf);
        // printf("source_pressure: %e, %e\n", source_pressure.r, source_pressure.i);
        // printf("Spherical spread: %e\n", att_spread);
        // printf("Pressure add to field point prior to spreading: %e, %e\n", var.r, var.i);
        return;
    }

    // printf("Pressure added to field point: %f, %f\n", var.r, var.i);

    // Save the pressure to the facet pressure array.
    // Note var may be small and accumulate over may projects that why the complex numbers are doubles.
    atomicAddDouble(&(field_points_pressure[field_point_num].r), var.r);
    atomicAddDouble(&(field_points_pressure[field_point_num].i), var.i);
}

int CudaModelTes::ProjectFromFacetsToFieldPoints()
{
    printf("ProjectFromFacetsToFieldPoints .......\n");

    for (int object_num = 0; object_num < host_object_num_facets.size(); object_num++)
    {

        for (int facet_num = 0; facet_num < host_object_num_facets[object_num]; facet_num++)
        {

            for (int field_point_num = 0; field_point_num < host_num_field_points; field_point_num++)
            {

                int3 h_Facets_points = host_Object_Facets_points[object_num][facet_num];

                dim3 threadsPerBlock(h_Facets_points.x, h_Facets_points.y);
                dim3 numBlocks(1, 1);

                // printf("ThreadsPerBlock.x: %d, threadsPerBlock.y: %d\n", threadsPerBlock.x, threadsPerBlock.y);
                // printf("numBlocks.x: %d, numBlocks.y: %d\n", numBlocks.x, numBlocks.y);

                ProjectFacetToFieldPointKernel<<<numBlocks, threadsPerBlock>>>(
                    dev_k_wave,
                    dev_pixel_delta,
                    field_point_num,
                    facet_num,
                    dev_field_points_position,
                    dev_field_points_pressure,
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
    return 0;
}
