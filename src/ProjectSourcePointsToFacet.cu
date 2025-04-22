#include "CudaModelTes.cuh"
#include "GeoMath.h"
#include "dcomplex.h"
#include "CudaUtils.cuh"

#include <stdio.h>

/**
 * @brief Kernel to waves from a point source to a facet.
 *
 * Note all pointers are to memory addresses on the device.
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

    int index_Bi = yPnt * NumXpnts + xPnt;

    float A_i = facets_PixelArea[facet_num][index_Bi];

    if (A_i == 0)
    {
        // printf("facets_PixelArea is zero\n");
        return;
    }

    float3 pg_i = source_points_position[source_point_num];
    dcomplex source_pressure = source_points_pressure[source_point_num];

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
    float3 pg_j;
    pg_j.x = xAxis.x + yAxis.x + facet_base.x;
    pg_j.y = xAxis.y + yAxis.y + facet_base.y;
    pg_j.z = xAxis.z + yAxis.z + facet_base.z;

    // The distance from the source point to the facet point.
    float r_si = sqrtf((pg_i.x - pg_j.x) * (pg_i.x - pg_j.x) + (pg_i.y - pg_j.y) * (pg_i.y - pg_j.y) + (pg_i.z - pg_j.z) * (pg_i.z - pg_j.z));

    dcomplex i = devComplex(0, 1);
    dcomplex ik = devCmul(i, k);
    dcomplex exp_ikr = devRCmul(r_si, ik);
    exp_ikr = devCexp(exp_ikr); // This has phase and attenuation.

    double p = 1 / r_si;
    dcomplex G = devRCmul(p, exp_ikr);

    dcomplex R = devCmul(G, source_pressure); // Greens function times the source pressure.

    if (devCabs(G) > 1.0)
    {
        printf("Source Point to Facet Error.\n");
        printf("Radius: %e\n", r_si);
        printf("Spherical spread: %e\n", A_i);
        printf("Pressure add to field point prior to spreading: %e, %e\n", R.r, R.i);
        return;
    }

    atomicAddDouble(&Pr_facet[index_Bi], R.r);
    atomicAddDouble(&Pi_facet[index_Bi], R.i);
}

int CudaModelTes::ProjectSourcePointsToFacet()
{
    printf("Host ProjectPointToFacet....\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Query device 0
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

    printf("ProjectFromFacetsToFieldPoints .......\n");

    // Every facet can have a different number of pixels, where n = 1096^0.5 is the maximum number of pixels per facet.

    for (int source_point_num = 0; source_point_num < host_num_source_points; source_point_num++)
    {
        for (int object_num = 0; object_num < host_object_num_facets.size(); object_num++)
        {
            for (int facet_num = 0; facet_num < host_object_num_facets[object_num]; facet_num++)
            {
                int3 h_Facets_points = host_Object_Facets_points[object_num][facet_num];

                dim3 threadsPerBlock(h_Facets_points.x, 1);
                dim3 numBlocks(h_Facets_points.y, 1);

                // printf("Mutex Address : "
                //        "%p\n",
                //        mutex_in_cuda[object_num][facet_num]);

                // printf("Surface Real Address : "
                //        "%p\n",
                //        dev_Object_Facets_Surface_Pr[object_num][facet_num]);

                // printf("Surface Imaginary Address : "
                //        "%p\n",
                //        dev_Object_Facets_Surface_Pi[object_num][facet_num]);

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

    cudaDeviceSynchronize();

    printf("ProjectSourcePointsToFacet done.\n");
    return 0;
}
