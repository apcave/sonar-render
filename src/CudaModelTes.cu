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

int CudaModelTes::GetFieldPointValGPU(dcomplex *field_points_pressure)
{
    // Copy the field point pressures from the device to the host.
    cudaMemcpy(field_points_pressure, dev_field_points_pressure, host_num_field_points * sizeof(dcomplex), cudaMemcpyDeviceToHost);
    return 0;
}
