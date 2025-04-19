#include "CudaModelTes.cuh"
#include "GeoMath.h"
#include "dcomplex.h"
#include "CudaUtils.cuh"

#include <stdio.h>

/**
 * @brief Kernel to project waves from facet pixels to facet pixels.
 *
 * Resprository holds so each projection is a forward and reverse projection.
 * It is not possible to do too many projections as pressure decays.
 * This is the most expensive kernel.
 */
__global__ void ProjectFromFacetsToFacetsKernel(
    dcomplex *k_wave,
    float *pixel_delta,
    int facet_num_A,
    int facet_num_B,
    int3 *facet_Points,
    float3 *base_points,
    float3 *facets_xaxis,
    float3 *facets_yaxis,
    float **facets_PixelArea,
    cudaSurfaceObject_t Pr_facet_A,
    cudaSurfaceObject_t Pi_facet_A,
    int *mutex_facet_A,
    cudaSurfaceObject_t Pr_facet_B,
    cudaSurfaceObject_t Pi_facet_B,
    int *mutex_facet_B,
    float *pixel_Pressure_stats,
    int *pixel_Pressure_stats_mutex)
{

    dcomplex k = *k_wave;
    float delta = *pixel_delta;

    // Kernel code to project point to point
    // printf("ThreadIdx.x: %d, ThreadIdx.y: %d, blockIdx.x: %d, blockDim.x: %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockDim.x);
    int xPnt_A = threadIdx.x;
    int yPnt_A = threadIdx.y;

    int xPnt_B = blockIdx.x;
    int yPnt_B = blockIdx.y;

    // To keep calculation even do not do a double projection when the two facets are the same.
    // bool skipReciprosity = facet_num_A == facet_num_B;

    // Do not do self projection the radius is zero.
    if (facet_num_A == facet_num_B && xPnt_A == xPnt_B && yPnt_A == yPnt_B)
    {
        return;
    }

    int NumXpnts_A = facet_Points[facet_num_A].x;
    int NumXpntsNegative_A = facet_Points[facet_num_A].z;

    int NumXpnts_B = facet_Points[facet_num_B].x;
    int NumXpntsNegative_B = facet_Points[facet_num_B].z;

    float pixel_area_A = facets_PixelArea[facet_num_A][yPnt_A * NumXpnts_A + xPnt_A];
    float pixel_area_B = facets_PixelArea[facet_num_B][yPnt_B * NumXpnts_B + xPnt_B];
    if (pixel_area_A == 0 || pixel_area_B == 0)
    {
        return;
    }

    // This is the x offset from the base point to the approximate centriod of the pixel.
    float xoffset_A = delta * (xPnt_A - NumXpntsNegative_A) + delta / 2; // This value can be negative.
    float xoffset_B = delta * (xPnt_B - NumXpntsNegative_B) + delta / 2; // This value can be negative.
    // This is the y offset from the base point to the approximate centriod of the pixel.
    float yoffset_A = delta * yPnt_A + delta / 2;
    float yoffset_B = delta * yPnt_B + delta / 2;

    float3 xAxis_A = facets_xaxis[facet_num_A];
    float3 xAxis_B = facets_xaxis[facet_num_B];

    float3 yAxis_A = facets_yaxis[facet_num_A];
    float3 yAxis_B = facets_yaxis[facet_num_B];

    xAxis_A.x = xoffset_A * xAxis_A.x;
    xAxis_A.y = xoffset_A * xAxis_A.y;
    xAxis_A.z = xoffset_A * xAxis_A.z;

    yAxis_A.x = yoffset_A * yAxis_A.x;
    yAxis_A.y = yoffset_A * yAxis_A.y;
    yAxis_A.z = yoffset_A * yAxis_A.z;

    xAxis_B.x = xoffset_B * xAxis_B.x;
    xAxis_B.y = xoffset_B * xAxis_B.y;
    xAxis_B.z = xoffset_B * xAxis_B.z;

    yAxis_B.x = yoffset_B * yAxis_B.x;
    yAxis_B.y = yoffset_B * yAxis_B.y;
    yAxis_B.z = yoffset_B * yAxis_B.z;

    float3 facet_base_A = base_points[facet_num_A];
    float3 PAg; // Point A global reference.
    PAg.x = xAxis_A.x + yAxis_A.x + facet_base_A.x;
    PAg.y = xAxis_A.y + yAxis_A.y + facet_base_A.y;
    PAg.z = xAxis_A.z + yAxis_A.z + facet_base_A.z;

    float3 facet_base_B = base_points[facet_num_B];
    float3 PBg;
    PBg.x = xAxis_B.x + yAxis_B.x + facet_base_B.x;
    PBg.y = xAxis_B.y + yAxis_B.y + facet_base_B.y;
    PBg.z = xAxis_B.z + yAxis_B.z + facet_base_B.z;

    float tmp_r, tmp_i;
    surf2Dread<float>(&tmp_r, Pr_facet_A, xPnt_A * sizeof(float), yPnt_A, cudaBoundaryModeTrap);
    surf2Dread<float>(&tmp_i, Pi_facet_A, xPnt_A * sizeof(float), yPnt_A, cudaBoundaryModeTrap);
    dcomplex PA_press;
    PA_press.r = tmp_r;
    PA_press.i = tmp_i;

    surf2Dread<float>(&tmp_r, Pr_facet_B, xPnt_B * sizeof(float), yPnt_B, cudaBoundaryModeTrap);
    surf2Dread<float>(&tmp_i, Pi_facet_B, xPnt_B * sizeof(float), yPnt_B, cudaBoundaryModeTrap);
    dcomplex PB_press;
    PB_press.r = tmp_r;
    PB_press.i = tmp_i;

    // The distance from the source point to the facet point.
    float r_AB = sqrtf((PAg.x - PBg.x) * (PAg.x - PBg.x) + (PAg.y - PBg.y) * (PAg.y - PBg.y) + (PAg.z - PBg.z) * (PAg.z - PBg.z));

    // printf("Distance from pixel to field point: %f\n", r_sf);

    // P2 = P2*exp(-i*k*r_sf)
    dcomplex i = devComplex(0, 1);
    dcomplex var = devCmul(i, k);
    var = devRCmul(r_AB, var);
    var = devCexp(var);                                // This has phase and attenuation.
    dcomplex delta_Press_atB = devCmul(var, PA_press); // This includes the orginal pressure.
    dcomplex delta_Press_atA = devCmul(var, PB_press);

    // Area1 = Pressure the 1Pa over 1m^2
    // Area2 = 4 * PI * r_sf * r_sf
    // atten_spread = Area1 / Area2 <--- important for other projections.
    float att_spread_AB = pow(pixel_area_A / (4 * M_PI * r_AB * r_AB), 0.5);
    float att_spread_BA = pow(pixel_area_B / (4 * M_PI * r_AB * r_AB), 0.5);

    delta_Press_atA = devRCmul(att_spread_BA, delta_Press_atA);
    delta_Press_atB = devRCmul(att_spread_AB, delta_Press_atB);
    // printf("Spherical spread: %f\n", att_spread);

    if (devCabs(delta_Press_atB) > 1.0)
    {
        printf("Pressure is too large to add to field point.\n");
        printf("r_AB: %f\n", r_AB);
        printf("source_pressure: %e, %e\n", PA_press.r, PA_press.i);
        printf("Spherical spread: %e\n", att_spread_AB);
        printf("Pressure add to field point prior to spreading: %e, %e\n", delta_Press_atB.r, delta_Press_atB.i);
        return;
    }

    if (devCabs(delta_Press_atA) > 1.0)
    {
        printf("Pressure is too large to add to field point.\n");
        printf("r_AB: %f\n", r_AB);
        printf("source_pressure: %e, %e\n", PB_press.r, PB_press.i);
        printf("Spherical spread: %e\n", att_spread_AB);
        printf("Pressure add to field point prior to spreading: %e, %e\n", delta_Press_atA.r, delta_Press_atA.i);
        return;
    }

    // printf("Pressure added to field point: %f, %f\n", var.r, var.i);

    int index_A = yPnt_A * NumXpnts_A + xPnt_A;
    int index_B = yPnt_B * NumXpnts_B + xPnt_B;

    tmp_r = (float)PA_press.r + delta_Press_atA.r;
    tmp_i = (float)PA_press.i + delta_Press_atA.i;

    if (mutex_facet_A[index_A] != 0)
    {
        printf("Test -- First A %d, %d, B %d, %d\n", index_A, mutex_facet_A[index_A], index_B, mutex_facet_B[index_B]);
    }
    while (atomicCAS(&mutex_facet_A[index_A], 0, 1) != 0)
    {
    }
    surf2Dwrite<float>(tmp_r, Pr_facet_A, xPnt_A * sizeof(float), yPnt_A, cudaBoundaryModeTrap);
    surf2Dwrite<float>(tmp_i, Pi_facet_A, xPnt_A * sizeof(float), yPnt_A, cudaBoundaryModeTrap);
    atomicExch(&mutex_facet_A[index_A], 0);

    tmp_r = (float)PB_press.r + delta_Press_atB.r;
    tmp_i = (float)PB_press.i + delta_Press_atB.i;

    while (atomicCAS(&mutex_facet_B[index_B], 0, 1) != 0)
    {
    }
    surf2Dwrite<float>(tmp_r, Pr_facet_B, xPnt_B * sizeof(float), yPnt_B, cudaBoundaryModeTrap);
    surf2Dwrite<float>(tmp_i, Pi_facet_B, xPnt_B * sizeof(float), yPnt_B, cudaBoundaryModeTrap);
    atomicExch(&mutex_facet_B[index_B], 0);

    printf("Test 3\n");
    return;
    float max_real = max(Pr_facet_A, Pr_facet_B);
    atomicMax((int *)&pixel_Pressure_stats[0], __float_as_int(max_real));

    float max_imag = max(Pi_facet_A, Pi_facet_B);
    atomicMax((int *)&pixel_Pressure_stats[1], __float_as_int(max_imag));

    float abs_Pi_facet_A = (float)devCabs(PA_press);
    float abs_Pi_facet_B = (float)devCabs(PB_press);
    float max_abs = max(abs_Pi_facet_A, abs_Pi_facet_B);
    atomicMax((int *)&pixel_Pressure_stats[2], __float_as_int(max_abs));
}

__global__ void CopySurfacePressureToMatrix(double *P_facet, cudaSurfaceObject_t P_facet, int numXpnt)
{
    int xPnt = threadIdx.x;
    int yPnt = threadIdx.y;

    float tmp;
    surf2Dread<float>(&tmp, P_facet, xPnt * sizeof(float), yPnt, cudaBoundaryModeTrap);

    P_facet[xPnt * numXpnt + yPnt] = (double)tmp;
}

/**
 * @brief Project the pressure from one facet to another facet.
 *
 * Note to save on memory there is one dcomplex per pixel.
 * Pressure is projected from facet to facet O(n^2) but intermediate results
 * are saved and projected to other facets and the field points.
 * The most accurate calulations are to project over and over to get the steady state solution.
 * As p_0, p_1=(p_0*T), p_2=(p_1*T), p_3(p_2*T)....p_n(p_(n-1)*T) where T < 1 <<-- Needs work!
 *
 * Note if the problem has not converged the results may not be symmetric.
 */

int CudaModelTes::ProjectFromFacetsToFacets()
{
    // TODO: Restrict the pixel maxtrix to 1024 pixels.
    // printf("ProjectFromFacetsToFacets .......\n");
    cudaMemset(dev_pixel_Pressure_stats, 0, 3 * sizeof(float));

    std::vector<std::vector<double *>> host_object_facet_Pi;
    std::vector<std::vector<double *>> host_object_facet_Pr;
    for (int object_num_a = 0; object_num_a < host_object_num_facets.size(); object_num_a++)
    {
        for (int facet_num_a = 0; facet_num_a < host_object_num_facets[object_num_a]; facet_num_a++)
        {
            int3 h_Facets_a_points = host_Object_Facets_points[object_num_a][facet_num_a];
            double *Pr;
            double *Pi;
            cudaMalloc((void **)&Pr, h_Facets_a_points.x * h_Facets_a_points.y * sizeof(double));
            cudaMalloc((void **)&Pi, h_Facets_a_points.x * h_Facets_a_points.y * sizeof(double));
            host_object_facet_Pi[object_num_a].push_back(Pi);
            host_object_facet_Pr[object_num_a].push_back(Pr);

            dim3 threadsPerBlock(h_Facets_a_points.x, h_Facets_a_points.y);
            dim3 numBlocks(1, 1);

            CopySurfacePressureToMatrix<<<numBlocks, threadsPerBlock>>>(Pr, dev_Object_Facets_Surface_Pr[object_num_a][facet_num_a], h_Facets_a_points.x);
            CopySurfacePressureToMatrix<<<numBlocks, threadsPerBlock>>>(Pi, dev_Object_Facets_Surface_Pi[object_num_a][facet_num_a], h_Facets_a_points.x);
        }
    }

    // One mutex per facet, locks it so only one thread can write to the surface at a time.

    for (int object_num_a = 0; object_num_a < host_object_num_facets.size(); object_num_a++)
    {
        for (int facet_num_a = 0; facet_num_a < host_object_num_facets[object_num_a]; facet_num_a++)
        {
            for (int object_num_b = 0; object_num_b < host_object_num_facets.size(); object_num_b++)
            {
                for (int facet_num_b = 0; facet_num_b < host_object_num_facets[object_num_b]; facet_num_b++)
                {
                    // printf("Facet A: %d, %d <<--->> Facet B: %d, %d\n", object_num_a, facet_num_a, object_num_b, facet_num_b);

                    int3 h_Facets_a_points = host_Object_Facets_points[object_num_a][facet_num_a];
                    int3 h_Facets_b_points = host_Object_Facets_points[object_num_b][facet_num_b];

                    dim3 threadsPerBlock(h_Facets_a_points.x, h_Facets_a_points.y);
                    dim3 numBlocks(h_Facets_b_points.x, h_Facets_b_points.y);

                    // printf("ThreadsPerBlock.x: %d, threadsPerBlock.y: %d\n", threadsPerBlock.x, threadsPerBlock.y);
                    //  printf("numBlocks.x: %d, numBlocks.y: %d\n", numBlocks.x, numBlocks.y);

                    ProjectFromFacetsToFacetsKernel<<<numBlocks, threadsPerBlock>>>(
                        dev_k_wave,
                        dev_pixel_delta,
                        facet_num_a,
                        facet_num_b,
                        dev_Object_Facets_points[0],
                        dev_Object_base_points[0],
                        dev_Object_Facets_xAxis[0],
                        dev_Object_Facets_yAxis[0],
                        dev_Object_Facets_PixelArea[0],
                        dev_Object_Facets_Surface_Pr[0][facet_num_a],
                        dev_Object_Facets_Surface_Pi[0][facet_num_a],
                        dev_Object_Facets_Surface_Pr[0][facet_num_b],
                        dev_Object_Facets_Surface_Pi[0][facet_num_b],
                        host_object_facet_Pr[0][facet_num_a],
                        host_object_facet_Pi[0][facet_num_a],
                        host_object_facet_Pr[0][facet_num_b],
                        host_object_facet_Pi[0][facet_num_b],
                        dev_pixel_Pressure_stats,
                        pixel_Pressure_stats_mutex);

                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
                        return 1;
                    }
                }
            }
        }
    }
    cudaMemcpy(host_pixel_Pressure_stats, dev_pixel_Pressure_stats, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Max Pressure: %e, %e\n", host_pixel_Pressure_stats[0], host_pixel_Pressure_stats[1]);
    printf("Max Abs Pressure: %e\n", host_pixel_Pressure_stats[2]);

    // More testing is required on large models to see how CUDA manages the cores.
    cudaDeviceSynchronize();

    return 0;
}