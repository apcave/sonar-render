#include "ModelCuda.hpp"
#include "GeoMath.h"
#include "dcomplex.h"

#include <stdio.h>

using namespace std;

int ModelCuda::SetGlobalParameters(dcomplex k_wave, float pixel_delta)
{
    // Set the global parameters for the GPU
    cudaMalloc(&dev_k_wave, 1 * sizeof(dcomplex));
    cudaError_t cudaStatus = cudaMemcpy(dev_k_wave, &k_wave, sizeof(dcomplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "k_wave failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    cudaMalloc(&dev_frag_delta, 1 * sizeof(float));
    cudaStatus = cudaMemcpy(dev_frag_delta, &pixel_delta, sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "pixel_delta failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    cudaMalloc(&dev_frag_stats, 3 * sizeof(float));
    return 0;
}

int ModelCuda::MakeSourcePointsOnGPU(vector<PressurePoint *> source_points)
{
    host_num_source_points = source_points.size();

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

int ModelCuda::MakeFieldPointsOnGPU(vector<PressurePoint *> field_points)
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

int ModelCuda::MakeObjectOnGPU(vector<Facet *> facets)
{
    for (auto facet : facets)
    {
        facet->MakeCuda();
    }
    return 0;
}

int ModelCuda::StartCuda()
{
    // cudaDeviceSynchronize();
    // cudaFree(0);
    // cudaDeviceReset();
    // cudaError_t cudaStatus = cudaSetDevice(0);
    // if (cudaStatus != cudaSuccess)
    // {
    //     printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
    //     return 1;
    // }
    return 0;
}

int ModelCuda::StopCuda()
{
    if (dev_k_wave)
    {
        cudaFree(dev_k_wave);
        dev_k_wave = nullptr;
    }
    if (dev_frag_delta)
    {
        cudaFree(dev_frag_delta);
        dev_frag_delta = nullptr;
    }
    if (dev_frag_stats)
    {
        cudaFree(dev_frag_stats);
        dev_frag_stats = nullptr;
    }
    if (dev_source_points_position)
    {
        cudaFree(dev_source_points_position);
        dev_source_points_position = nullptr;
    }
    if (dev_source_points_pressure)
    {
        cudaFree(dev_source_points_pressure);
        dev_source_points_pressure = nullptr;
    }
    if (dev_field_points_position)
    {
        cudaFree(dev_field_points_position);
        dev_field_points_position = nullptr;
    }
    if (dev_field_points_pressure)
    {
        cudaFree(dev_field_points_pressure);
        dev_field_points_pressure = nullptr;
    }

    for (auto object : targetObjects)
    {
        delete object;
    }
    targetObjects.clear();

    OptiXCol.StopCollision();

    return 0;
}

int ModelCuda::DoCalculations()
{

    if (ProjectSourcePointsToFacet() != 0)
    {
        printf("ProjectSourcePointsToFacet failed.\n");
        return 1;
    }

    for (int i = 0; i < 0; i++)
    {
        if (ProjectFromFacetsToFacets() != 0)
        {
            printf("ProjectFromFacetsToFacets failed.\n");
            return 1;
        }
    }

    if (ProjectFromFacetsToFieldPoints() != 0)
    {
        printf("ProjectFromFacetsToFieldPoints failed.\n");
        return 1;
    }

    return 0;
}

int ModelCuda::GetFieldPointValGPU(dcomplex *field_points_pressure)
{
    // Copy the field point pressures from the device to the host.
    cudaMemcpy(field_points_pressure, dev_field_points_pressure, host_num_field_points * sizeof(dcomplex), cudaMemcpyDeviceToHost);
    return 0;
}

void ModelCuda::WriteCudaToGlTexture()
{
    for (auto object : targetObjects)
    {
        object->WriteSurfaceToGlTexture(dev_frag_stats);
    }
}

int ModelCuda::GetSurfaceScalers()
{
    // Clear the pressure stats.
    cudaMemset(dev_frag_stats, 0, 3 * sizeof(float));

    // Copy the pressure from the matrix to the surface.
    for (auto object : targetObjects)
    {
        object->GetSurfaceScalers(dev_frag_stats);
    }
    return 1;
}

ModelCuda::ModelCuda()
{
    // Initialize the CUDA device
    cudaFree(0);
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
    }
    OptiXCol.StartOptix();
}

ModelCuda::~ModelCuda()
{
    StopCuda();
}
