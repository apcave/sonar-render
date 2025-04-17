#include "CudaModelTes.cuh"
#include "GeoMath.h"
#include "dcomplex.h"
#include "CudaUtils.cuh"

#include <stdio.h>

/**
 * @brief Kernel to waves from a point source to a facet.
 *
 * Note all pointers are to memory addressses on the device.
 * This function is not to be used with matrix compression.
 * The calculations are done in the global coordinate system.
 */
__global__ void ProjectSourcePointToFacetKernel(
    dcomplex *k_wave,
    float *pixel_delta,
    int source_point_num,
    int facet_num,
    float3 *source_points_position,
    dcomplex *source_points_pressure,
    int3 *facet_Points,
    float3 *base_points,
    float3 *facets_xaxis,
    float3 *facets_yaxis,
    float **facets_PixelArea,
    dcomplex **facets_Pressure)
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

    if (facets_PixelArea[facet_num][yPnt * NumXpnts + xPnt] == 0)
    {
        // printf("facets_PixelArea is zero\n");
        return;
    }

    // printf("xPnt: %d, yPnt: %d\n", xPnt, yPnt);
    // printf("NumXpnts: %d, NumYpnts: %d, NumXpntsNegative: %d\n", NumXpnts, NumYpnts, NumXpntsNegative);

    float3 P1g = source_points_position[source_point_num];
    dcomplex source_pressure = source_points_pressure[source_point_num];

    // printf("Source Point: %f, %f, %f\n", P1g.x, P1g.y, P1g.z);
    // printf("Source Pressure: %f, %f\n", source_pressure.r, source_pressure.i);

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
    float3 P2g;
    P2g.x = xAxis.x + yAxis.x + facet_base.x;
    P2g.y = xAxis.y + yAxis.y + facet_base.y;
    P2g.z = xAxis.z + yAxis.z + facet_base.z;

    // The distance from the source point to the facet point.
    float r_ij = sqrtf((P1g.x - P2g.x) * (P1g.x - P2g.x) + (P1g.y - P2g.y) * (P1g.y - P2g.y) + (P1g.z - P2g.z) * (P1g.z - P2g.z));

    // P2 = P2*exp(-i*k*r_sf)
    dcomplex i = devComplex(0, 1);
    dcomplex var = devCmul(i, k);
    var = devRCmul(r_ij, var);
    var = devCexp(var);                  // This has phase and attenuation.
    var = devCmul(var, source_pressure); // This includes the orginal pressure.
    // printf("Pressure prior to spreading at facet point: %f, %f\n", var.r, var.i);

    // Area1 = Pressure the 1Pa over 1m^2
    // Area2 = 4 * PI * r_sf * r_sf
    // atten_spread = Area1 / Area2 <--- important for other projections.

    // Point sources have pressure values @ RE 1 m
    // A_i = 4 * PI * 1^2
    // A_j = 4 * PI * r_sf * r_sf
    float A_r = 1 / (r_ij * r_ij);

    var = devRCmul(A_r, var);
    // printf("Spherical spread: %f\n", att_spread);

    if (devCabs(var) > 1.0)
    {
        printf("Source Point to Facet Error.\n");
        printf("Radius: %e\n", r_ij);
        printf("Spherical spread: %e\n", A_r);
        printf("Pressure add to field point prior to spreading: %e, %e\n", var.r, var.i);
        return;
    }

    // printf("Pressure at facet point: %f, %f\n", var.r, var.i);

    // Save the pressure to the facet pressure array.
    // Note var may be small and accumulate over may projects that why the complex numbers are doubles.
    atomicAddDouble(&(facets_Pressure[facet_num][yPnt * NumXpnts + xPnt].r), var.r);
    atomicAddDouble(&(facets_Pressure[facet_num][yPnt * NumXpnts + xPnt].i), var.i);
}

int CudaModelTes::ProjectSourcePointsToFacet()
{
    // Every facet can have a different number of pixels, where n = 1096^0.5 is the maximum number of pixels per facet.
    // printf("Host ProjectPointToFacet....\n");

    for (int source_point_num = 0; source_point_num < host_num_source_points; source_point_num++)
    {
        for (int object_num = 0; object_num < host_object_num_facets.size(); object_num++)
        {
            for (int facet_num = 0; facet_num < host_object_num_facets[object_num]; facet_num++)
            {
                int3 h_Facets_points = host_Object_Facets_points[object_num][facet_num];

                dim3 threadsPerBlock(h_Facets_points.x, h_Facets_points.y);
                dim3 numBlocks(1, 1);

                // printf("ThreadsPerBlock.x: %d, threadsPerBlock.y: %d\n", threadsPerBlock.x, threadsPerBlock.y);
                // printf("numBlocks.x: %d, numBlocks.y: %d\n", numBlocks.x, numBlocks.y);

                ProjectSourcePointToFacetKernel<<<numBlocks, threadsPerBlock>>>(
                    dev_k_wave,
                    dev_pixel_delta,
                    source_point_num,
                    facet_num,
                    dev_source_points_position,
                    dev_source_points_pressure,
                    dev_Object_Facets_points[object_num],
                    dev_Object_base_points[object_num],
                    dev_Object_Facets_xAxis[object_num],
                    dev_Object_Facets_yAxis[object_num],
                    dev_Object_Facets_PixelArea[object_num],
                    dev_Object_Facets_Pressure[object_num]);

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
