#ifndef _ModelTes
#define _ModelTes
#include "Facet.hpp"
#include "PressurePoint.hpp"
#include "CudaModelTes.cuh"
#include "OpenGL_TES.hpp"
#include <vector>
#include <iostream>

using namespace std;

class TargetObject
{
public:
    vector<Facet *> facets;
    // TODO: Add material and medium properties.

public:
    void addFacet(float3 v1, float3 v2, float3 v3)
    {
        // Add the facet to the target object
        // This is a placeholder for the actual implementation
        auto facet = new Facet(v1, v2, v3);
        facets.push_back(facet);
    }

    void MakePixelData(float pixel_length)
    {
        // Iterate over each facet and call the pixelation function
        for (auto &facet : facets)
        {
            // Call the pixelation function on each facet
            facet->MakePixelData(pixel_length);
            facet->PrintMatrix();
        }
    }
};

class ModelTes : public CudaModelTes, public OpenGL_TES
{

private:
    /**
     * The source usually comes from a single point or piston surface.
     * Using a surface is important for directionality of the beam.
     * When using a surface the pressure radiates from top of the surface only.
     * Sound doesn't reflect off the source surface.
     */
    vector<PressurePoint *> sourcePoints;
    TargetObject *sourceObject = NULL;

    /**
     * Objects the targets that cause reflections.
     * Targets consist of facets.
     * Targets can have material coatings with different acoustics properties.
     */
    vector<TargetObject *> targetObjects;

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
    float pixel_length;
    float density;

public:
    void addFeildPoint(float3 p1);

    void addSourcePoint(float3 p1);
    void addTargetObject(TargetObject *object);

    void set_inital_conditions(float cp, float t_frequency, float attenuation, float t_density);

    void pixelate_facets();

    void copyToDevice();

    void GetFieldPointPressures(dcomplex *field_points_pressure, int NumPoints);
    void RenderOpenGL();
};

#endif