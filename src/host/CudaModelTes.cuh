#ifndef _MODEL_TES_HPP
#define _MODEL_TES_HPP
#include "Facet.hpp"
#include "dcomplex.h"
#include "PressurePoint.hpp"
#include <cuda_runtime.h>
#include <vector>
using namespace std;

class CudaModelTes
{
protected:
    dcomplex *dev_k_wave;
    float *dev_pixel_delta;

    // Object based data.
    vector<float **> dev_Object_Facets_PixelArea; // constant
    vector<dcomplex **> dev_Object_Facets_Pressure;
    vector<int3 *> dev_Object_Facets_points;    // constant
    vector<float3 *> dev_Object_Facets_Normals; // constant
    vector<float3 *> dev_Object_base_points;    // constant
    vector<float3 *> dev_Object_Facets_xAxis;   // constant
    vector<float3 *> dev_Object_Facets_yAxis;

    float3 *dev_source_points_position;   // constant
    dcomplex *dev_source_points_pressure; // constant

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

    int TestGPU();

    int ProjectSourcePointToFacet(int source_point_num, int object_num, int facet_num);
};

#endif