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

    cudaMalloc(&dev_scratch, sizeof(dev_scratch_pad));
    cudaStatus = cudaMemset(dev_scratch, 0, sizeof(dev_scratch_pad));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "scratch data failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

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

    if (dev_source_points_position || dev_source_points_pressure)
    {
        throw std::runtime_error("Source points already allocated on GPU.");
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

    if (dev_field_points_position || dev_field_points_pressure)
    {
        throw std::runtime_error("Field points already allocated on GPU.");
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
    if (!cudaStarted)
    {
        std::cout << "Starting CUDA... <---------------------------------------------" << std::endl;
        cudaDeviceSynchronize();
        cudaFree(0);
        cudaDeviceReset();
        cudaError_t cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
            return 1;
        }
        cudaStarted = true;
    }
    return 0;
}

int ModelCuda::StopCuda()
{
    std::cout << "Stopping CUDA..." << std::endl;
    if (dev_k_wave)
    {
        cudaFree(dev_k_wave);
        dev_k_wave = 0;
    }
    if (dev_frag_delta)
    {
        cudaFree(dev_frag_delta);
        dev_frag_delta = 0;
    }

    if (dev_source_points_position)
    {
        cudaFree(dev_source_points_position);
        dev_source_points_position = 0;
    }
    if (dev_source_points_pressure)
    {
        cudaFree(dev_source_points_pressure);
        dev_source_points_pressure = 0;
    }
    if (dev_field_points_position)
    {
        cudaFree(dev_field_points_position);
        dev_field_points_position = 0;
    }
    if (dev_field_points_pressure)
    {
        cudaFree(dev_field_points_pressure);
        dev_field_points_pressure = 0;
    }

    if (dev_scratch)
    {
        cudaFree(dev_scratch);
        dev_scratch = 0;
    }

    for (auto object : targetObjects)
    {
        delete object;
    }
    targetObjects.clear();

    for (auto object : fieldObjects)
    {
        delete object;
    }
    fieldObjects.clear();

    std::cout << "Stopping OptiX..." << std::endl;
    optiX.TearDown();

    return 0;
}

int ModelCuda::DoCalculations()
{
    std::cout << "Source Points to target." << std::endl;
    ProjectSrcPointsToObjects(false);

    // for (int i = 0; i < 4; i++)
    // {
    // //     std::cout << "Target to target objects." << std::endl;
    //     ProjectTargetToTargetObjects();
    // }

    // std::cout << "Target to field objects." << std::endl;
    // ProjectTargetToFieldObjects();

    // std::cout << "Target to field points." << std::endl;
    // ProjectTargetToFieldPoints();

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
        object->WriteSurfaceToGlTexture(min_dB, max_dB, render_phase_target);
    }

    for (auto object : fieldObjects)
    {
        std::cout << "Writing field object to OpenGL texture." << std::endl;
        object->WriteSurfaceToGlTexture(min_dB, max_dB, render_phase_field);
    }
}

ModelCuda::ModelCuda()
{
    // std::cout << "Made ModelCuda::ModelCuda() object." << std::endl;
    // // Initialize the CUDA device
    // cudaFree(0);
    // cudaError_t cudaStatus = cudaSetDevice(0);
    // if (cudaStatus != cudaSuccess)
    // {
    //     printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
    // }
}

ModelCuda::~ModelCuda()
{
    std::cout << "Cleaning up ModelCuda..." << std::endl;
    StopCuda();
}

void ModelCuda::ProjectSrcPointsToObjects(bool projectFieldObjects)
{
    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << "Projecting source points to target objects." << std::endl;
    globalParams gp = {};
    gp.calcType = CalcType::SOURCE_POINTS;
    gp.k_wave = k_wave;
    gp.scratch = dev_scratch;
    gp.srcPoints.numPnts = host_num_source_points;
    gp.srcPoints.position = dev_source_points_position;
    gp.srcPoints.pressure = dev_source_points_pressure;

    for (auto object : targetObjects)
    {
        std::cout << "Doing source point to target object projection." << std::endl;
        gp.dstObject = object->MakeOptixStructArray();
        std::cout << "Doing facet to facet projection."
                  << " Num Dest: " << gp.dstObject.numFacets << std::endl;

        optiX.DoProjection(gp);

        // object->AccumulatePressure();
    }

    if (projectFieldObjects)
    {
        for (auto object : fieldObjects)
        {
            std::cout << "Doing source point to field object projection." << std::endl;
            gp.dstObject = object->MakeOptixStructArray();

            std::cout << "Doing facet to facet projection. Num Source: " << gp.srcObject.numFacets
                      << ", Num Dest: " << gp.dstObject.numFacets << std::endl;
            optiX.DoProjection(gp);
        }
    }
}

void ModelCuda::ProjectTargetToFieldObjects()
{
    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << "Projecting target objects to field objects." << std::endl;
    globalParams gp = {};
    gp.calcType = CalcType::FACET_NO_RESP;
    gp.k_wave = k_wave;
    gp.scratch = dev_scratch;

    for (auto targetOb : targetObjects)
    {
        gp.srcObject = targetOb->MakeOptixStructArray();

        for (auto fieldOb : fieldObjects)
        {
            gp.dstObject = fieldOb->MakeOptixStructArray();
            std::cout << "Doing facet to facet projection. Num Source: " << gp.srcObject.numFacets
                      << ", Num Dest: " << gp.dstObject.numFacets << std::endl;
            optiX.DoProjection(gp);
        }
    }
}
void ModelCuda::ProjectTargetToFieldPoints()
{
    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << "Projecting target objects to field points." << std::endl;
    globalParams gp = {};
    gp.calcType = CalcType::FIELD_POINTS;
    gp.k_wave = k_wave;
    gp.scratch = dev_scratch;

    gp.dstPoints.numPnts = host_num_field_points;
    gp.dstPoints.position = dev_field_points_position;
    gp.dstPoints.pressure = dev_field_points_pressure;

    for (auto targetOb : targetObjects)
    {
        gp.srcObject = targetOb->MakeOptixStructArray();

        optiX.DoProjection(gp);
    }
}

void ModelCuda::ProjectTargetToTargetObjects(int number_reflections)
{
    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << "Projecting target objects to target objects." << std::endl;
    globalParams gp = {};
    gp.calcType = CalcType::FACET_SELF;
    gp.k_wave = k_wave;
    gp.scratch = dev_scratch;

    for (auto targetOb : targetObjects)
    {
        targetOb->PrimeReflections();
    }

    for (int i = 0; number_reflections > i; i++)
    {
        for (auto targetOb : targetObjects)
        {
            gp.srcObject = targetOb->MakeOptixStructArray();
            targetOb->SwapOutputToInputPressure();
            gp.dstObject = gp.srcObject;
            std::cout << "__________________________________________________________________________" << std::endl;
            std::cout << "Doing facet to facet projection. Num Source: " << gp.srcObject.numFacets
                      << ", Num Dest: " << gp.dstObject.numFacets << std::endl;
            optiX.DoProjection(gp);
            targetOb->AccumulatePressure();
        }
    }

    // for (auto targetOb : targetObjects)
    // {
    //     targetOb->AccumulatePressure();
    // }
}

void ModelCuda::SoundVisualisationInit(float maxDbValue, float minDbValue, bool renderPhaseTarget, bool renderPhaseField)
{
    max_dB = maxDbValue;
    min_dB = minDbValue;
    render_phase_target = renderPhaseTarget;
    render_phase_field = renderPhaseField;

    std::cout << "FacetCuda: SoundVisualisationInit(): max_dB: " << max_dB << ", min_dB: " << min_dB
              << ", render_phase_target: " << render_phase_target
              << ", render_phase_field: " << render_phase_field << std::endl;
}