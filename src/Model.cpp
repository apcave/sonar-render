#include "Model.hpp"

void Model::addFeildPoint(float3 p1)
{
    // Add the field point to the model
    // This is a placeholder for the actual implementation
    auto fieldPoint = new PressurePoint(p1);
    feildPoints.push_back(fieldPoint);

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
    targetObjects.push_back(object);
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

    cout << "-------------------------------------------\n";
    cout << "Medium Wave Speed: " << medium_waveSpeed << endl;
    cout << "Frequency: " << frequency << endl;
    cout << "Medium Attenuation: " << medium_attenuation << endl;
    cout << "Medium Density: " << density << endl;
    cout << "Omega: " << omega << endl;
    cout << "Wave Number: " << k << endl;
    cout << "Pixel Length: " << frag_length << endl;
    cout << "Resolution Factor: " << resolution_factor << endl;
}

void Model::pixelate_facets()
{
    // Iterate over each target object and its facets
    for (auto object : targetObjects)
    {
        object->MakeFragmentData(frag_length);
    }
}

void Model::copyToDevice()
{
    StartCuda();

    SetGlobalParameters(k_wave, frag_length);

    for (auto object : targetObjects)
    {
        object->MakeCudaObjects();
    }

    MakeSourcePointsOnGPU(sourcePoints);
    MakeFieldPointsOnGPU(feildPoints);

    DoCalculations();

    CleanupCuda();

    std::cout << "CUDA calculations completed successfully." << std::endl;
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
    std::cout << "OpenGL initialized successfully." << std::endl;

    MakeObjectsOnGl();

    GetSurfaceScalers();
    WriteCudaToGlTexture();

    ProcessFrame();
}