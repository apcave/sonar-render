#ifndef _MODEL
#define _MODEL
#include "Facet.hpp"
#include "PressurePoint.hpp"
#include "ModelCuda.hpp"

#include <vector>
#include <iostream>

using namespace std;

class Model : public ModelCuda
{

private:
    /**
     * The source usually comes from a single point or piston surface.
     * Using a surface is important for directionality of the beam.
     * When using a surface the pressure radiates from top of the surface only.
     * Sound doesn't reflect off the source surface.
     */
    vector<PressurePoint *> sourcePoints;

    /**
     * Feild points are the output points at where the pressure is measured.
     */
    vector<PressurePoint *> feildPoints;

    float medium_waveSpeed;
    float frequency;
    float medium_attenuation;
    float omega;
    float k;
    dcomplex k_wave;
    float resolution_factor = 7.5;
    float frag_length;
    float density;

public:
    void addFeildPoint(float3 p1);

    void addSourcePoint(float3 p1);
    void addTargetObject(Object *object);

    void set_inital_conditions(float cp, float t_frequency, float attenuation, float t_density);

    void MakeFragments();

    void RenderCuda();

    void TearDownModel();

    void GetFieldPointPressures(dcomplex *field_points_pressure, int NumPoints);
    void RenderOpenGL();
};

#endif