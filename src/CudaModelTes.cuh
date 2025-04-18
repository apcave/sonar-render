#ifndef _MODEL_TES_HPP
#define _MODEL_TES_HPP
#include "Facet.hpp"
#include "dcomplex.h"
#include "PressurePoint.hpp"
#include <cuda_runtime.h>
#include <vector>
using namespace std;

/**
 * @brief CudaModelTes is a class that manages the GPU resources and
 *        calculations for the pressure field in a 3D model.
 *
 * Memory on the device is allocated and stored in points dev_.
 * Data to be used on the host processor is marked with host_.
 *
 * The function definitions are spread across multiple CUDA files.
 * The CUDA kernels map some of the class functions as C style functions.
 * Each CUDA kernel is defined in a separate file with the C++ class function
 * that calls it.
 */

class CudaModelTes
{
protected:
    // Global parameters.
    // Complex wave number.
    dcomplex *dev_k_wave;
    // Pixel length and width.
    float *dev_pixel_delta;

    // Object based data.
    vector<int> host_object_num_facets;
    vector<float **> dev_Object_Facets_PixelArea; // constant
    vector<dcomplex **> dev_Object_Facets_Pressure;
    vector<int3 *> dev_Object_Facets_points;    // constant
    vector<float3 *> dev_Object_Facets_Normals; // constant
    vector<float3 *> dev_Object_base_points;    // constant
    vector<float3 *> dev_Object_Facets_xAxis;   // constant
    vector<float3 *> dev_Object_Facets_yAxis;

    int host_num_source_points;
    float3 *dev_source_points_position;   // constant
    dcomplex *dev_source_points_pressure; // constant

    int host_num_field_points;
    float3 *dev_field_points_position;   // constant
    dcomplex *dev_field_points_pressure; // constant

    // These are used to manage CUDA threads.
    vector<int3 *> host_Object_Facets_points;

public:
    CudaModelTes() {};
    ~CudaModelTes() {};

protected:
    int SetGlobalParameters(dcomplex k_wave, float pixel_delta);
    int MakeObjectOnGPU(vector<Facet *> facets);
    int MakeSourcePointsOnGPU(vector<PressurePoint *> source_points);
    int MakeFieldPointsOnGPU(vector<PressurePoint *> field_points);
    int StartCuda();
    int StopCuda();
    int DoCalculations();

    int ProjectSourcePointsToFacet();
    int ProjectFromFacetsToFieldPoints();
    int ProjectFromFacetsToFacets();

    int GetFieldPointValGPU(dcomplex *field_points_pressure);
};

#endif