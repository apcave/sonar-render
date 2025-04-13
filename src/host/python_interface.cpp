#include "ModelTes.hpp"

#include <cuda_runtime.h>
#include <iostream>

using namespace std;

ModelTes modelTes;

extern "C" void load_geometry(float *v1, float *v2, float *v3, int num_vertices);
extern "C" void load_field_points(float *v1, int num_feild_points);
extern "C" void load_source_points(float *v1, int num_source_points);
extern "C" void set_initial_conditions(float cp, float frequency, float attenuation);
extern "C" void pixelate_facets();

extern "C" void set_initial_conditions(float cp, float frequency, float attenuation)
{
    cout << "Setting initial conditions..." << endl;
    modelTes.set_inital_conditions(cp, frequency, attenuation);
    cout << "Initial conditions set." << endl;
};

extern "C" void load_geometry(float *v1, float *v2, float *v3, int num_vertices)
{
    float *d_a, *d_b, *d_c;

    cout << "Allocating device memory..." << endl;

    auto object = new TargetObject();

    for (int i = 0; i < num_vertices; ++i)
    {
        float3 p_v1 = {v1[i * 3 + 0], v1[i * 3 + 1], v1[i * 3 + 2]};
        float3 p_v2 = {v2[i * 3 + 0], v2[i * 3 + 1], v2[i * 3 + 2]};
        float3 p_v3 = {v3[i * 3 + 0], v3[i * 3 + 1], v3[i * 3 + 2]};
        object->addFacet(p_v1, p_v2, p_v3);
    }
    modelTes.addTargetObject(object);
    cout << "Loaded geometry with " << num_vertices << " vertices." << endl;
};

extern "C" void load_field_points(float *v1, int num_feild_points)
{

    cout << "Allocating device memory..." << endl;

    for (int i = 0; i < num_feild_points; ++i)
    {
        float3 p1 = {v1[i * 3 + 0], v1[i * 3 + 1], v1[i * 3 + 2]};
        modelTes.addFeildPoint(p1);
    }

    cout << "Loaded field points with " << num_feild_points << " points." << endl;
};

extern "C" void load_source_points(float *v1, int num_source_points)
{

    cout << "Allocating device memory..." << endl;

    for (int i = 0; i < num_source_points; ++i)
    {
        float3 p1 = {v1[i * 3 + 0], v1[i * 3 + 1], v1[i * 3 + 2]};
        printf("Source Point: %f %f %f\n", p1.x, p1.y, p1.z);
        modelTes.addSourcePoint(p1);
    }

    cout << "Loaded source points with " << num_source_points << " points." << endl;
};

extern "C" void pixelate_facets()
{
    cout << "Pixelating facets..." << endl;
    modelTes.pixelate_facets();
    cout << "Pixelated facets." << endl;
    modelTes.copyToDevice();
    cout << "Copied data to device." << endl;
};