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
    double *Pr_acc_A,
    double *Pi_acc_A,
    double *Pr_acc_B,
    double *Pi_acc_B,

    double *Pr_ini_A,
    double *Pi_ini_A,
    double *Pr_ini_B,
    double *Pi_ini_B,

    double *Pr_res_A,
    double *Pi_res_A,
    double *Pr_res_B,
    double *Pi_res_B)
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

    int index_A = yPnt_A * NumXpnts_A + xPnt_A;
    int index_B = yPnt_B * NumXpnts_B + xPnt_B;

    float pixel_area_A = facets_PixelArea[facet_num_A][index_A];
    float pixel_area_B = facets_PixelArea[facet_num_B][index_B];
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

    dcomplex PA_press;
    PA_press.r = Pr_ini_A[index_A];
    PA_press.i = Pi_ini_A[index_A];
    ;

    dcomplex PB_press;
    PB_press.r = Pr_ini_B[index_B];
    PB_press.i = Pi_ini_B[index_B];
    ;

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
    // float att_spread_AB = pow(pixel_area_A / (4 * M_PI * r_AB * r_AB), 0.5);
    // float att_spread_BA = pow(pixel_area_B / (4 * M_PI * r_AB * r_AB), 0.5);

    float att_spread_AB = pixel_area_A / (4 * M_PI * r_AB * r_AB);
    float att_spread_BA = pixel_area_B / (4 * M_PI * r_AB * r_AB);
    if (att_spread_AB > 1.0)
    {
        printf("att_spread_AB: %f\n", att_spread_AB);
        att_spread_AB = 1.0;
    }
    if (att_spread_BA > 1.0)
    {
        printf("att_spread_BA: %f\n", att_spread_BA);
        att_spread_BA = 1.0;
    }

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

    atomicAddDouble(&Pr_acc_A[index_A], delta_Press_atA.r);
    atomicAddDouble(&Pi_acc_A[index_A], delta_Press_atA.i);
    atomicAddDouble(&Pr_acc_B[index_B], delta_Press_atB.r);
    atomicAddDouble(&Pi_acc_B[index_B], delta_Press_atB.i);

    atomicAddDouble(&Pr_res_A[index_A], delta_Press_atA.r);
    atomicAddDouble(&Pi_res_A[index_A], delta_Press_atA.i);
    atomicAddDouble(&Pr_res_B[index_B], delta_Press_atB.r);
    atomicAddDouble(&Pi_res_B[index_B], delta_Press_atB.i);
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
 * The pressure waves can only go forward so there is an initial pressure matrix.
 * Ip -> Tp  Ap = Ap + Tp, Tp -> Ip
 * The pressure transmitted get smaller and smaller and is acculutated every step.
 * So there is an upper bound on the accumulated pressure.
 */
int CudaModelTes::ProjectFromFacetsToFacets()
{
    // TODO: Restrict the pixel maxtrix to 1024 pixels.
    printf("ProjectFromFacetsToFacets .......\n");

    cudaMemset(dev_pixel_Pressure_stats, 0, 3 * sizeof(float));

    // One mutex per facet, locks it so only one thread can write to the surface at a time.
    for (int object_num = 0; object_num < host_object_num_facets.size(); object_num++)
    {
        for (int facet_num = 0; facet_num < host_object_num_facets[object_num]; facet_num++)
        {
            int3 h_Facets_points = host_Object_Facets_points[object_num][facet_num];
            int buff_sz = h_Facets_points.x * h_Facets_points.y * sizeof(double);
            cudaMemcpy((*dev_object_facet_InitialPr)[object_num][facet_num], dev_object_facet_Pr[object_num][facet_num], buff_sz, cudaMemcpyHostToHost);
            cudaMemcpy((*dev_object_facet_InitialPi)[object_num][facet_num], dev_object_facet_Pi[object_num][facet_num], buff_sz, cudaMemcpyHostToHost);
        }
    }

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
                    //   printf("numBlocks.x: %d, numBlocks.y: %d\n", numBlocks.x, numBlocks.y);

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
                        dev_object_facet_Pr[0][facet_num_a],
                        dev_object_facet_Pi[0][facet_num_a],
                        dev_object_facet_Pr[0][facet_num_b],
                        dev_object_facet_Pi[0][facet_num_b],
                        (*dev_object_facet_InitialPr)[object_num_a][facet_num_a],
                        (*dev_object_facet_InitialPi)[object_num_a][facet_num_a],
                        (*dev_object_facet_InitialPr)[object_num_b][facet_num_b],
                        (*dev_object_facet_InitialPi)[object_num_b][facet_num_b],
                        (*dev_object_facet_ResultPr)[object_num_a][facet_num_a],
                        (*dev_object_facet_ResultPi)[object_num_a][facet_num_a],
                        (*dev_object_facet_ResultPr)[object_num_b][facet_num_b],
                        (*dev_object_facet_ResultPi)[object_num_b][facet_num_b]);

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

    cudaDeviceSynchronize();

    // Swap the results and the initial pressure.
    std::vector<std::vector<double *>> *tmp = dev_object_facet_InitialPr;
    dev_object_facet_InitialPr = dev_object_facet_ResultPr;
    dev_object_facet_ResultPr = tmp;

    tmp = dev_object_facet_InitialPi;
    dev_object_facet_InitialPi = dev_object_facet_ResultPi;
    dev_object_facet_ResultPi = tmp;

    // One mutex per facet, locks it so only one thread can write to the surface at a time.
    for (int object_num = 0; object_num < host_object_num_facets.size(); object_num++)
    {
        for (int facet_num = 0; facet_num < host_object_num_facets[object_num]; facet_num++)
        {
            int3 h_Facets_points = host_Object_Facets_points[object_num][facet_num];
            int buff_sz = h_Facets_points.x * h_Facets_points.y * sizeof(double);
            cudaMemset((*dev_object_facet_ResultPr)[object_num][facet_num], 0, buff_sz);
            cudaMemset((*dev_object_facet_ResultPi)[object_num][facet_num], 0, buff_sz);
        }
    }

    return 0;
}