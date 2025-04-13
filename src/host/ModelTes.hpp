#ifndef _ModelTes
#define _ModelTes
#include "Facet.hpp"
#include "PressurePoint.hpp"
#include "CudaModelTes.cuh"
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

class ModelTes : public CudaModelTes
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

public:
    void addFeildPoint(float3 p1)
    {
        // Add the field point to the model
        // This is a placeholder for the actual implementation
        auto fieldPoint = new PressurePoint(p1);
        feildPoints.push_back(fieldPoint);

        printf("Field Point: %f %f %f\n", fieldPoint->position.x, fieldPoint->position.y, fieldPoint->position.z);
    }

    void addSourcePoint(float3 p1)
    {
        // Add the field point to the model
        // This is a placeholder for the actual implementation
        auto sourcePoint = new PressurePoint(p1);
        sourcePoint->pressure.r = 1.0; // Set the pressure value for the source point
        sourcePoints.push_back(sourcePoint);

        printf("Source Point: %f %f %f\n", sourcePoint->position.x, sourcePoint->position.y, sourcePoint->position.z);
    }

    void addTargetObject(TargetObject *object)
    {
        // Add the target object to the model
        // This is a placeholder for the actual implementation
        targetObjects.push_back(object);
    }
    void set_inital_conditions(float cp, float t_frequency, float attenuation)
    {
        medium_waveSpeed = cp;
        frequency = t_frequency;
        medium_attenuation = attenuation;
        omega = 2 * M_PI * frequency;
        k = omega / cp;
        k_wave.r = (double)k;
        k_wave.i = (double)attenuation;
        pixel_length = medium_waveSpeed / (frequency * resolution_factor);

        cout << "-------------------------------------------\n";
        cout << "Medium Wave Speed: " << medium_waveSpeed << endl;
        cout << "Frequency: " << frequency << endl;
        cout << "Medium Attenuation: " << medium_attenuation << endl;
        cout << "Omega: " << omega << endl;
        cout << "Wave Number: " << k << endl;
        cout << "Pixel Length: " << pixel_length << endl;
        cout << "Resolution Factor: " << resolution_factor << endl;
    }

    void pixelate_facets()
    {
        // Iterate over each target object and its facets
        for (auto &targetObject : targetObjects)
        {
            targetObject->MakePixelData(pixel_length);
        }
    }

    void copyToDevice()
    {
        StartCuda();
        SetGlobalParameters(k_wave, pixel_length);

        MakeSourcePointsOnGPU(sourcePoints);
        MakeFieldPointsOnGPU(feildPoints);

        int objectCnt = 0;
        for (auto &targetObject : targetObjects)
        {
            auto &object = targetObjects[0];
            auto &facets = targetObject->facets;
            MakeObjectOnGPU(facets);
            objectCnt++;
        };

        DoCalculations();
    }
};

#endif