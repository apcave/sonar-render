#include "CudaModelTes.cuh"
#include <stdio.h>

using namespace std;

int CudaModelTes::SetGlobalParameters(dcomplex k_wave, float pixel_delta)
{
    // Set the global parameters for the GPU
    cudaMalloc(&dev_k_wave, 1 * sizeof(dcomplex));
    cudaError_t cudaStatus = cudaMemcpyToSymbol(dev_k_wave, &k_wave, sizeof(dcomplex));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "k_wave failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    cudaMemcpyToSymbol(dev_pixel_delta, &pixel_delta, sizeof(float));
    cudaStatus = cudaMemcpyToSymbol(dev_pixel_delta, &pixel_delta, sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "pixel_delta failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    return 0;
}

int CudaModelTes::MakeSourcePointsOnGPU(vector<PressurePoint *> source_points)
{
    int numPoints = source_points.size();

    auto position = new float3[numPoints];
    auto pressure = new dcomplex[numPoints];
    for (int i = 0; i < numPoints; i++)
    {
        position[i] = source_points[i]->position;
        pressure[i] = source_points[i]->pressure;
    }
    // Allocate memory for the source points on the device
    cudaMalloc(&dev_source_points_position, numPoints * sizeof(float3));
    cudaMemcpy(dev_source_points_position, position, numPoints * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_source_points_pressure, numPoints * sizeof(dcomplex));
    cudaMemcpy(dev_source_points_pressure, pressure, numPoints * sizeof(dcomplex), cudaMemcpyHostToDevice);
    // Free the host memory
    delete[] position;
    delete[] pressure;
    return 0;
}

int CudaModelTes::MakeFieldPointsOnGPU(vector<PressurePoint *> field_points)
{

    return 0;
}

int CudaModelTes::MakeObjectOnGPU(vector<Facet *> facets, dcomplex k_wave, float pixel_delta)
{
    // Copy the facet data to the GPU
    printf("MakeObjectOnGPU .......\n");

    int number_of_facets = facets.size();

    float ***dev_Facets_PixelArea;
    cudaMalloc(&dev_Facets_PixelArea, number_of_facets * sizeof(float *));
    dev_Object_Facets_PixelArea.push_back(dev_Facets_PixelArea);

    float **host_PixelArea = new float *[number_of_facets];
    for (int i = 0; i < number_of_facets; i++)
    {
        int ArrLen = facets[i]->NumXpnts * facets[i]->NumYpnts;
        // Allocate the Area for the pixel array on the host.
        cudaMalloc(&host_PixelArea[i], ArrLen * sizeof(float));
        // Copy the pixel area data to host CUDA to the device.
        cudaMemcpy(&host_PixelArea[i], facets[i]->PixelArea, ArrLen * sizeof(float), cudaMemcpyHostToDevice);
    }
    // Copy the pixel area data to the device.
    cudaMemcpy(dev_Facets_PixelArea, host_PixelArea, number_of_facets * sizeof(float *), cudaMemcpyHostToDevice);

    dcomplex ***dev_Facets_Pressure;
    cudaMalloc(&dev_Facets_Pressure, number_of_facets * sizeof(dcomplex *));
    dev_Object_Facets_Pressure.push_back(dev_Facets_Pressure);
    dcomplex **host_Pressure = new dcomplex *[number_of_facets];
    for (int i = 0; i < number_of_facets; i++)
    {
        printf("test 3.1 >>>>>>>>>>>>>>>>>>>>>>>>\n");
        int ArrLen = facets[i]->NumXpnts * facets[i]->NumYpnts;
        // Allocate the Area for the pixel pressure array on the host.
        cudaMalloc(&host_Pressure[i], ArrLen * sizeof(dcomplex));
        // Copy the pixel pressure data to host CUDA to the device.
        cudaMemcpy(&host_Pressure[i], facets[i]->PressureValues, ArrLen * sizeof(dcomplex), cudaMemcpyHostToDevice);
    }
    // Copy the pixel pressure data to the device.
    cudaMemcpy(dev_Facets_Pressure, host_Pressure, number_of_facets * sizeof(dcomplex *), cudaMemcpyHostToDevice);

    int facet_cnt = 0;
    auto host_Facets_NumXpnts = new int[number_of_facets];
    auto host_Facets_NumYpnts = new int[number_of_facets];
    auto host_Facets_Normals = new float3[number_of_facets];
    auto host_base_points = new float3[number_of_facets];
    for (int i = 0; i < number_of_facets; i++)
    {
        // Memory for dev_Facets_PixelArea[facet_cnt] is already allocated in the initialization loop above.
        host_Facets_NumXpnts[i] = facets[i]->NumXpnts;
        host_Facets_NumYpnts[i] = facets[i]->NumYpnts;
        host_Facets_Normals[i] = facets[i]->Normal;
        host_base_points[i] = facets[i]->pointOnBase;
        facet_cnt++;
    }

    int **dev_Facets_NumXpnts;
    cudaMalloc(&dev_Facets_NumXpnts, number_of_facets * sizeof(int));
    cudaMemcpy(dev_Facets_NumXpnts, host_Facets_NumXpnts, number_of_facets * sizeof(int), cudaMemcpyHostToDevice);
    dev_Object_Facet_NumXpnts.push_back(dev_Facets_NumXpnts);

    int **dev_Facets_NumYpnts;
    cudaMalloc(&dev_Facets_NumYpnts, number_of_facets * sizeof(int));
    cudaMemcpy(dev_Facets_NumYpnts, host_Facets_NumYpnts, number_of_facets * sizeof(int), cudaMemcpyHostToDevice);
    dev_Object_Facet_NumYpnts.push_back(dev_Facets_NumYpnts);

    float3 **dev_Facets_Normals;
    cudaMalloc(&dev_Facets_Normals, number_of_facets * sizeof(float3));
    cudaMemcpy(dev_Facets_Normals, host_Facets_Normals, number_of_facets * sizeof(float3), cudaMemcpyHostToDevice);
    dev_Object_Facets_Normals.push_back(dev_Facets_Normals);

    float3 **dev_base_points;
    cudaMalloc(&dev_base_points, number_of_facets * sizeof(float3));
    cudaMemcpy(dev_base_points, host_base_points, number_of_facets * sizeof(float3), cudaMemcpyHostToDevice);
    dev_Object_base_points.push_back(dev_base_points);

    delete[] host_PixelArea;
    delete[] host_Pressure;
    delete[] host_Facets_NumXpnts;
    delete[] host_Facets_NumYpnts;
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