#include "CudaModelTes.cuh"
#include "GeoMath.h"
#include "dcomplex.h"
#include "CudaUtils.cuh"

#include <stdio.h>

using namespace std;

int CudaModelTes::SetGlobalParameters(dcomplex k_wave, float pixel_delta)
{
    printf("SetGlobalParameters .......\n");
    // Set the global parameters for the GPU
    cudaMalloc(&dev_k_wave, 1 * sizeof(dcomplex));
    cudaError_t cudaStatus = cudaMemcpy(dev_k_wave, &k_wave, sizeof(dcomplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "k_wave failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    printf("pixel_delta: %f\n", pixel_delta);
    cudaMalloc(&dev_pixel_delta, 1 * sizeof(float));
    cudaStatus = cudaMemcpy(dev_pixel_delta, &pixel_delta, sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "pixel_delta failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    return 0;
}

int CudaModelTes::MakeSourcePointsOnGPU(vector<PressurePoint *> source_points)
{
    host_num_source_points = source_points.size();
    printf("MakeSourcePointsOnGPU (num pnts %d) .......\n", host_num_source_points);

    auto position = new float3[host_num_source_points];
    auto pressure = new dcomplex[host_num_source_points];
    for (int i = 0; i < host_num_source_points; i++)
    {
        position[i] = source_points[i]->position;
        pressure[i] = source_points[i]->pressure;
    }
    // Allocate memory for the source points on the device
    cudaMalloc(&dev_source_points_position, host_num_source_points * sizeof(float3));
    cudaMemcpy(dev_source_points_position, position, host_num_source_points * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_source_points_pressure, host_num_source_points * sizeof(dcomplex));
    cudaMemcpy(dev_source_points_pressure, pressure, host_num_source_points * sizeof(dcomplex), cudaMemcpyHostToDevice);
    // Free the host memory
    delete[] position;
    delete[] pressure;
    return 0;
}

int CudaModelTes::MakeFieldPointsOnGPU(vector<PressurePoint *> field_points)
{

    host_num_field_points = field_points.size();

    auto position = new float3[host_num_field_points];
    auto pressure = new dcomplex[host_num_field_points];
    for (int i = 0; i < host_num_field_points; i++)
    {
        position[i] = field_points[i]->position;
        pressure[i] = field_points[i]->pressure;
    }
    // Allocate memory for the source points on the device
    cudaMalloc(&dev_field_points_position, host_num_field_points * sizeof(float3));
    cudaMemcpy(dev_field_points_position, position, host_num_field_points * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_field_points_pressure, host_num_field_points * sizeof(dcomplex));
    cudaMemcpy(dev_field_points_pressure, pressure, host_num_field_points * sizeof(dcomplex), cudaMemcpyHostToDevice);
    // Free the host memory
    delete[] position;
    delete[] pressure;

    return 0;
}

int CudaModelTes::MakeObjectOnGPU(vector<Facet *> facets)
{
    // Copy the facet data to the GPU
    printf("MakeObjectOnGPU .......\n");

    int number_of_facets = facets.size();
    host_object_num_facets.push_back(number_of_facets);

    float **dev_Facets_PixelArea;
    cudaMalloc(&dev_Facets_PixelArea, number_of_facets * sizeof(float *));
    dev_Object_Facets_PixelArea.push_back(dev_Facets_PixelArea);

    float **host_PixelArea = new float *[number_of_facets];
    for (int i = 0; i < number_of_facets; i++)
    {
        int ArrLen = facets[i]->NumXpnts * facets[i]->NumYpnts;
        // Allocate the Area for the pixel array on the host.
        cudaMalloc(&host_PixelArea[i], ArrLen * sizeof(float));
        // Copy the pixel area data to host CUDA to the device.
        cudaMemcpy(host_PixelArea[i], facets[i]->PixelArea, ArrLen * sizeof(float), cudaMemcpyHostToDevice);
    }
    // Copy the pixel area data to the device.
    cudaMemcpy(dev_Facets_PixelArea, host_PixelArea, number_of_facets * sizeof(float *), cudaMemcpyHostToDevice);

    dcomplex **dev_Facets_Pressure;
    cudaMalloc(&dev_Facets_Pressure, number_of_facets * sizeof(dcomplex *));
    dev_Object_Facets_Pressure.push_back(dev_Facets_Pressure);
    dcomplex **host_Pressure = new dcomplex *[number_of_facets];
    for (int i = 0; i < number_of_facets; i++)
    {
        int ArrLen = facets[i]->NumXpnts * facets[i]->NumYpnts;
        // Allocate the Area for the pixel pressure array on the host.
        cudaMalloc(&host_Pressure[i], ArrLen * sizeof(dcomplex));
        // Copy the pixel pressure data to host CUDA to the device.
        cudaMemcpy(host_Pressure[i], facets[i]->PressureValues, ArrLen * sizeof(dcomplex), cudaMemcpyHostToDevice);
    }
    // Copy the pixel pressure data to the device.
    cudaMemcpy(dev_Facets_Pressure, host_Pressure, number_of_facets * sizeof(dcomplex *), cudaMemcpyHostToDevice);

    int facet_cnt = 0;
    auto host_Facets_points = new int3[number_of_facets];
    auto host_Facets_Normals = new float3[number_of_facets];
    auto host_base_points = new float3[number_of_facets];
    auto host_Facets_xAxis = new float3[number_of_facets];
    auto host_Facets_yAxis = new float3[number_of_facets];
    for (int i = 0; i < number_of_facets; i++)
    {
        // Memory for dev_Facets_PixelArea[facet_cnt] is already allocated in the initialization loop above.
        host_Facets_points[i].x = facets[i]->NumXpnts;
        host_Facets_points[i].y = facets[i]->NumYpnts;
        host_Facets_points[i].z = facets[i]->NumXpntsNegative;
        host_Facets_Normals[i] = facets[i]->Normal;
        host_base_points[i] = facets[i]->pointOnBase;
        host_Facets_xAxis[i] = facets[i]->xAxis;
        host_Facets_yAxis[i] = facets[i]->yAxis;
        facet_cnt++;
    }

    int3 *dev_Facets_points;
    cudaMalloc(&dev_Facets_points, number_of_facets * sizeof(int3));
    cudaMemcpy(dev_Facets_points, host_Facets_points, number_of_facets * sizeof(int3), cudaMemcpyHostToDevice);
    dev_Object_Facets_points.push_back(dev_Facets_points);
    host_Object_Facets_points.push_back(host_Facets_points);

    float3 *dev_Facets_Normals;
    cudaMalloc(&dev_Facets_Normals, number_of_facets * sizeof(float3));
    cudaMemcpy(dev_Facets_Normals, host_Facets_Normals, number_of_facets * sizeof(float3), cudaMemcpyHostToDevice);
    dev_Object_Facets_Normals.push_back(dev_Facets_Normals);

    float3 *dev_base_points;
    cudaMalloc(&dev_base_points, number_of_facets * sizeof(float3));
    cudaMemcpy(dev_base_points, host_base_points, number_of_facets * sizeof(float3), cudaMemcpyHostToDevice);
    dev_Object_base_points.push_back(dev_base_points);

    float3 *dev_Facets_xAxis;
    cudaMalloc(&dev_Facets_xAxis, number_of_facets * sizeof(float3));
    cudaMemcpy(dev_Facets_xAxis, host_Facets_xAxis, number_of_facets * sizeof(float3), cudaMemcpyHostToDevice);
    dev_Object_Facets_xAxis.push_back(dev_Facets_xAxis);

    float3 *dev_Facets_yAxis;
    cudaMalloc(&dev_Facets_yAxis, number_of_facets * sizeof(float3));
    cudaMemcpy(dev_Facets_yAxis, host_Facets_yAxis, number_of_facets * sizeof(float3), cudaMemcpyHostToDevice);
    dev_Object_Facets_yAxis.push_back(dev_Facets_yAxis);

    delete[] host_PixelArea;
    delete[] host_Pressure;
    delete[] host_Facets_Normals;
    delete[] host_base_points;

    printf("Allocated object memory on GPU.\n");
    return 0;
}

int CudaModelTes::StartCuda()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        return 1;
    }
    printf("CUDA started successfully.\n");
    return 0;
}

int CudaModelTes::StopCuda()
{
    // cudaError_t cudaStatus = cudaUnbindTexture(dev_Positions);
    // if (cudaStatus != cudaSuccess)
    // {
    //     printf("Unbinding of Positions Texture Failed!\n");
    //     return 1;
    // }

    // cudaStatus = cudaDeviceReset();
    // if (cudaStatus != cudaSuccess)
    // {
    //     fprintf(stderr, "cudaDeviceReset failed!\n");
    //     return 1;
    // }
    return 0;
}

int CudaModelTes::DoCalculations()
{
    printf("DoCalculations .......\n");
    // TestGPU();
    ProjectSourcePointsToFacet();
    for (int i = 0; i < 1000; i++)
    {
        ProjectSourcePointsToFacet();
    }
    ProjectFromFacetsToFieldPoints();
    return 0;
}

__global__ void ProjectSourceFacetToFeildPointKernel(
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

    float pixel_area = facets_PixelArea[facet_num][yPnt * NumXpnts + xPnt];
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

    dcomplex source_pressure = facets_Pressure[facet_num][yPnt * NumXpnts + xPnt];

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
    float att_spread = pixel_area / (4 * M_PI * r_sf * r_sf);
    var = devRCmul(att_spread, var);
    // printf("Spherical spread: %f\n", att_spread);

    if (devCabs(var) > 1.0)
    {
        printf("Pressure is too large to add to field point.\n");
        printf("r_sf: %f\n", r_sf);
        printf("source_pressure: %e, %e\n", source_pressure.r, source_pressure.i);
        printf("Spherical spread: %e\n", att_spread);
        printf("Pressure add to field point prior to spreading: %e, %e\n", var.r, var.i);
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

                ProjectSourceFacetToFeildPointKernel<<<numBlocks, threadsPerBlock>>>(
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

int CudaModelTes::GetFieldPointValGPU(dcomplex *field_points_pressure)
{
    // Copy the field point pressures from the device to the host.
    cudaMemcpy(field_points_pressure, dev_field_points_pressure, host_num_field_points * sizeof(dcomplex), cudaMemcpyDeviceToHost);
    return 0;
}

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
    dcomplex **facets_Pressure)
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

    dcomplex PA_press = facets_Pressure[facet_num_A][yPnt_A * NumXpnts_A + xPnt_A];
    dcomplex PB_press = facets_Pressure[facet_num_B][yPnt_B * NumXpnts_B + xPnt_B];

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
    float att_spread_AB = pixel_area_A / (4 * M_PI * r_AB * r_AB);
    float att_spread_BA = pixel_area_B / (4 * M_PI * r_AB * r_AB);

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

    // Save the pressure to the facet pressure array.
    // Note var may be small and accumulate over may projects that why the complex numbers are doubles.
    atomicAddDouble(&(facets_Pressure[facet_num_A][yPnt_A * NumXpnts_A + xPnt_A]).r, delta_Press_atA.r);
    atomicAddDouble(&(facets_Pressure[facet_num_A][yPnt_A * NumXpnts_A + xPnt_A]).i, delta_Press_atA.i);

    atomicAddDouble(&(facets_Pressure[facet_num_B][yPnt_B * NumXpnts_B + xPnt_B]).r, delta_Press_atB.r);
    atomicAddDouble(&(facets_Pressure[facet_num_B][yPnt_B * NumXpnts_B + xPnt_B]).i, delta_Press_atB.i);
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
                    // printf("numBlocks.x: %d, numBlocks.y: %d\n", numBlocks.x, numBlocks.y);

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
                        dev_Object_Facets_Pressure[0]);

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
    // More testing is required on large models to see how CUDA manages the cores.
    cudaDeviceSynchronize();
    return 0;
}