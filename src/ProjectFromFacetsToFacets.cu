#include "ModelCuda.hpp"
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
    float *frag_delta,
    dev_facet *src_facet_data,
    float *src_frag_area,
    double *src_Pr_facet,
    double *src_Pi_facet,
    dev_facet *dst_facet_data,
    float *dst_frag_area,
    double *dst_Pr_facet,
    double *dst_Pi_facet)
{
    dcomplex k = *k_wave;
    float delta = *frag_delta;

    // Kernel code to project point to point
    // printf("ThreadIdx.x: %d, ThreadIdx.y: %d, blockIdx.x: %d, blockIdx.y: %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

    int xPnt_A = threadIdx.x;
    int yPnt_A = threadIdx.y;

    int xPnt_B = blockIdx.x;
    int yPnt_B = blockIdx.y;

    // To keep calculation even do not do a double projection when the two facets are the same.
    // bool skipReciprosity = facet_num_A == facet_num_B;

    int NumXpnts_A = src_facet_data->frag_points.x;
    int NumXpntsNegative_A = src_facet_data->frag_points.z;

    int NumXpnts_B = dst_facet_data->frag_points.x;
    int NumXpntsNegative_B = dst_facet_data->frag_points.z;

    int index_A = yPnt_A * NumXpnts_A + xPnt_A;
    int index_B = yPnt_B * NumXpnts_B + xPnt_B;

    // Always calculate the pressure even if the destination area is zero.
    // It doesn't take extra time and can be used for the texture.
    float A_i = src_frag_area[index_A];
    // float pixel_area_B = dst_frag_area[index_B];

    // This is the x offset from the base point to the approximate centriod of the pixel.
    float xoffset_A = delta * (xPnt_A - NumXpntsNegative_A) + delta / 2; // This value can be negative.
    float xoffset_B = delta * (xPnt_B - NumXpntsNegative_B) + delta / 2; // This value can be negative.
    // This is the y offset from the base point to the approximate centriod of the pixel.
    float yoffset_A = delta * yPnt_A + delta / 2;
    float yoffset_B = delta * yPnt_B + delta / 2;

    float3 xAxis_A = src_facet_data->xAxis;
    float3 xAxis_B = dst_facet_data->xAxis;

    float3 yAxis_A = src_facet_data->yAxis;
    float3 yAxis_B = dst_facet_data->yAxis;

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

    float3 facet_base_A = src_facet_data->base_point;
    float3 r_i; // Point A global reference.
    r_i.x = xAxis_A.x + yAxis_A.x + facet_base_A.x;
    r_i.y = xAxis_A.y + yAxis_A.y + facet_base_A.y;
    r_i.z = xAxis_A.z + yAxis_A.z + facet_base_A.z;

    float3 facet_base_B = dst_facet_data->base_point;
    float3 r_j;
    r_j.x = xAxis_B.x + yAxis_B.x + facet_base_B.x;
    r_j.y = xAxis_B.y + yAxis_B.y + facet_base_B.y;
    r_j.z = xAxis_B.z + yAxis_B.z + facet_base_B.z;

    dcomplex p_i;
    p_i.r = src_Pr_facet[index_A];
    p_i.i = src_Pi_facet[index_A];

    float3 vr_ij = MakeVector(r_i, r_j);
    float r_ij = GetVectorLength(vr_ij);
    float3 ur_ij = DivideVector(vr_ij, r_ij);

    float3 n = src_facet_data->normal;
    double sc = (double)DotProduct(ur_ij, n);

    if (sc < 0)
    {
        // printf("Normal doesn't align, not adding to field point.\n");
        return;
    }

    dcomplex i = devComplex(0, 1);
    dcomplex ik = devCmul(i, k);
    dcomplex exp_ikr = devRCmul(r_ij, ik);
    exp_ikr = devCexp(exp_ikr); // This has phase and attenuation.

    double p = -1 / (4 * PI);
    dcomplex Gt = devRCmul(p, exp_ikr);

    dcomplex dG_dr_a = devRCmul(1 / r_ij, ik); // First term.
    double dG_dr_b = 1 / (r_ij * r_ij);        // Second term often ignored useful for near field.
    dcomplex dG_dr = dG_dr_a;
    dG_dr.r += dG_dr_b;
    dG_dr = devCmul(dG_dr, Gt); // This is the derivative of the Greens function.

    dcomplex result = devRCmul(A_i * sc, dG_dr); // Area term for the integral.
    result = devCmul(result, p_i);               // This includes the original pressure.

    if (devCabs(result) > 1.0)
    {
        printf("Pressure is too large to add to surface point.\n");
        printf("r_ij: %f\n", r_ij);
        printf("source_pressure: %e, %e\n", p_i.r, p_i.i);
        // printf("Spherical spread: %e\n", realTerms);
        printf("Pressure to field point prior to spreading: %e, %e\n", result.r, result.i);
        return;
    }
    atomicAddDouble(&(dst_Pr_facet[index_B]), result.r);
    atomicAddDouble(&(dst_Pi_facet[index_B]), result.i);
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
int ModelCuda::ProjectFromFacetsToFacets(std::vector<Object *> &scrObjects, std::vector<Object *> &dstObjects, bool reciprocity)
{
    for (auto srcOb : scrObjects)
    {
        auto srcPnts = srcOb->GetCentroids();
        int numScr = srcPnts.size();

        for (auto dstOb : dstObjects)
        {
            auto dstPnts = dstOb->GetCentroids();
            printf("Doing collision detection.\n");
            int *hasCollision = OptiXCol.DoCollisions(srcPnts, dstPnts);
            int numDst = dstPnts.size();

            for (int srcCnt = 0; numScr > srcCnt; srcCnt++)
            {
                for (int dstCnt = 0; numDst > dstCnt; dstCnt++)
                {
                    printf("NumSrcFacets: %d, scrNum %d, NumDstFacets: %d, dstNum %d\n", numScr, srcCnt, numDst, dstCnt);

                    if (hasCollision[srcCnt * numDst + dstCnt] == 1)
                    {
                        printf("Collision detected.\n");
                        continue;
                    }
                    printf("Running Projection.\n");
                    auto srcFacet = srcOb->facets[srcCnt];
                    auto dstFacet = dstOb->facets[dstCnt];

                    dim3 threadsPerBlock(srcFacet->frag_points.x, srcFacet->frag_points.y);
                    dim3 numBlocks(dstFacet->frag_points.x, dstFacet->frag_points.y, 1);

                    ProjectFromFacetsToFacetsKernel<<<numBlocks, threadsPerBlock>>>(
                        dev_k_wave,
                        dev_frag_delta,
                        srcFacet->dev_data,
                        srcFacet->dev_frag_area,
                        srcFacet->dev_Pr,
                        srcFacet->dev_Pi,
                        dstFacet->dev_data,
                        dstFacet->dev_frag_area,
                        dstFacet->dev_Pr,
                        dstFacet->dev_Pi);
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
                        return 1;
                    }
                }
                cudaDeviceSynchronize();
            }

            delete[] hasCollision;
        }
    }

        return 0;
}
