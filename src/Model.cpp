#include "Model.hpp"

void Model::addFeildPoint(float3 p1)
{
    // Add the field point to the model
    // This is a placeholder for the actual implementation
    auto fieldPoint = new PressurePoint(p1);
    fieldPoints.push_back(fieldPoint);

    // printf("Field Point: %f %f %f\n", fieldPoint->position.x, fieldPoint->position.y, fieldPoint->position.z);
}

void Model::addSourcePoint(float3 p1)
{
    // Add the field point to the model
    // This is a placeholder for the actual implementation
    auto sourcePoint = new PressurePoint(p1);
    sourcePoint->pressure.r = 1.0; // Set the pressure value for the source point
    sourcePoints.push_back(sourcePoint);

    // printf("Source Point: %f %f %f\n", sourcePoint->position.x, sourcePoint->position.y, sourcePoint->position.z);
}

void Model::addTargetObject(Object *object)
{
    // Add the target object to the model
    // This is a placeholder for the actual implementation
    switch (object->objectType)
    {
    case OBJECT_TYPE_TARGET:
        targetObjects.push_back(object);
        break;
    case OBJECT_TYPE_SOURCE:
        sourceObjects.push_back(object);
        break;
    case OBJECT_TYPE_FIELD:
        fieldObjects.push_back(object);
        break;
    default:
        break;
    }
}

void Model::set_inital_conditions(float cp, float t_frequency, float attenuation, float t_density)
{
    medium_waveSpeed = cp;
    frequency = t_frequency;
    medium_attenuation = attenuation;
    density = t_density;
    omega = 2 * M_PI * frequency;
    k = omega / cp;
    k_wave.r = (double)k;
    k_wave.i = (double)attenuation;
    frag_length = medium_waveSpeed / (frequency * resolution_factor);
}

void Model::MakeFragments()
{
    // Iterate over each target object and its facets
    for (auto object : targetObjects)
    {
        object->MakeFragmentData(frag_length);
    }

    for (auto object : fieldObjects)
    {
        object->MakeFragmentData(frag_length);
    }
}

void Model::RenderCuda()
{
    std::cout << "Rendering CUDA..." << std::endl;
    StartCuda();
    optiXCol.StartCollision(targetObjects);
    optiX.StartCollision(fieldObjects);

    SetGlobalParameters(k_wave, frag_length);

    for (auto object : targetObjects)
    {
        object->MakeCudaObjects();
    }

    for (auto object : fieldObjects)
    {
        object->MakeCudaObjects();
    }

    MakeSourcePointsOnGPU(sourcePoints);
    MakeFieldPointsOnGPU(fieldPoints);

    DoCalculations();
}

void Model::GetFieldPointPressures(dcomplex *field_points_pressure, int NumPoints)
{
    // Copy the pressure values from the device to the host
    GetFieldPointValGPU(field_points_pressure);
}

/**
 * @brief Renders surface pressure to OpenGL viewport.
 *
 * Note it expects the CUDA calculations to be done before calling this function.
 */
void Model::RenderOpenGL()
{
    InitOpenGL();

    MakeObjectsOnGl();

    GetSurfaceScalers();
    WriteCudaToGlTexture();

    ProcessFrame();
}

void Model::TearDownModel()
{
    for (auto pnt : sourcePoints)
    {
        delete pnt;
    }
    sourcePoints.clear();

    for (auto pnt : fieldPoints)
    {
        delete pnt;
    }
    fieldPoints.clear();
    StopCuda();
}