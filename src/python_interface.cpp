#include "Model.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <thrust/complex.h>
#include <complex>
#include <thrust/device_vector.h>

using namespace std;

Model modelTes;

extern "C" void load_geometry(float *v1, int num_vertices);
extern "C" void load_field_points(float *v1, int num_feild_points);
extern "C" void load_source_points(float *v1, int num_source_points);
extern "C" void set_initial_conditions(float cp, float frequency, float attenuation, float density);
extern "C" void render_cuda();
extern "C" void GetFieldPointPressures(dcomplex *field_points_pressure, int NumPoints);
extern "C" void RenderOpenGL();
extern "C" void TearDownCuda();

extern "C" void set_initial_conditions(float cp, float frequency, float attenuation, float density)
{
    modelTes.set_inital_conditions(cp, frequency, attenuation, density);
};

extern "C" void load_geometry(float *v1, int num_vertices)
{
    auto object = new Object();

    for (int i = 0; i < num_vertices; ++i)
    {
        float3 p_v1 = {v1[i * 9 + 0], v1[i * 9 + 1], v1[i * 9 + 2]};
        float3 p_v2 = {v1[i * 9 + 3], v1[i * 9 + 4], v1[i * 9 + 5]};
        float3 p_v3 = {v1[i * 9 + 6], v1[i * 9 + 7], v1[i * 9 + 8]};

        object->AddFacet(p_v1, p_v2, p_v3);
    }
    modelTes.addTargetObject(object);
};

extern "C" void load_field_points(float *v1, int num_feild_points)
{
    for (int i = 0; i < num_feild_points; ++i)
    {
        float3 p1 = {v1[i * 3 + 0], v1[i * 3 + 1], v1[i * 3 + 2]};
        modelTes.addFeildPoint(p1);
    }
};

extern "C" void load_source_points(float *v1, int num_source_points)
{
    for (int i = 0; i < num_source_points; ++i)
    {
        float3 p1 = {v1[i * 3 + 0], v1[i * 3 + 1], v1[i * 3 + 2]};
        modelTes.addSourcePoint(p1);
    }
};

extern "C" void render_cuda()
{
    modelTes.MakeFragments();
    modelTes.RenderCuda();
};

extern "C" void GetFieldPointPressures(dcomplex *field_points_pressure, int NumPoints)
{
    modelTes.GetFieldPointPressures(field_points_pressure, NumPoints);
}

extern "C" void RenderOpenGL()
{
    modelTes.RenderOpenGL();
}

extern "C" void TearDownCuda()
{
    modelTes.TearDownModel();
}