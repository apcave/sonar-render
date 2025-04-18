#include "ModelTes.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <thrust/complex.h>
#include <complex>
#include <thrust/device_vector.h>

using namespace std;

ModelTes modelTes;

extern "C" void load_geometry(float *v1, int num_vertices);
extern "C" void load_field_points(float *v1, int num_feild_points);
extern "C" void load_source_points(float *v1, int num_source_points);
extern "C" void set_initial_conditions(float cp, float frequency, float attenuation, float density);
extern "C" void pixelate_facets();
extern "C" void GetFieldPointPressures(dcomplex *field_points_pressure, int NumPoints);
extern "C" void RenderOpenGL();

extern "C" void set_initial_conditions(float cp, float frequency, float attenuation, float density)
{
    cout << "Setting initial conditions..." << endl;
    modelTes.set_inital_conditions(cp, frequency, attenuation, density);
    cout << "Initial conditions set." << endl;
};

extern "C" void load_geometry(float *v1, int num_vertices)
{

    cout << "Allocating device memory..." << endl;

    auto object = new TargetObject();

    for (int i = 0; i < num_vertices; ++i)
    {
        float3 p_v1 = {v1[i * 9 + 0], v1[i * 9 + 1], v1[i * 9 + 2]};
        float3 p_v2 = {v1[i * 9 + 3], v1[i * 9 + 4], v1[i * 9 + 5]};
        float3 p_v3 = {v1[i * 9 + 6], v1[i * 9 + 7], v1[i * 9 + 8]};

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

extern "C" void GetFieldPointPressures(dcomplex *field_points_pressure, int NumPoints)
{
    cout << "Getting field point pressures..." << endl;
    modelTes.GetFieldPointPressures(field_points_pressure, NumPoints);
    cout << "Got field point pressures." << endl;
}

extern "C" void RenderOpenGL()
{
    cout << "Rendering OpenGL..." << endl;
    modelTes.RenderOpenGL();
    cout << "Rendered OpenGL." << endl;
}