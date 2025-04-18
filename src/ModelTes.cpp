#include "ModelTes.hpp"

void ModelTes::addFeildPoint(float3 p1)
{
    // Add the field point to the model
    // This is a placeholder for the actual implementation
    auto fieldPoint = new PressurePoint(p1);
    feildPoints.push_back(fieldPoint);

    // printf("Field Point: %f %f %f\n", fieldPoint->position.x, fieldPoint->position.y, fieldPoint->position.z);
}

void ModelTes::addSourcePoint(float3 p1)
{
    // Add the field point to the model
    // This is a placeholder for the actual implementation
    auto sourcePoint = new PressurePoint(p1);
    sourcePoint->pressure.r = 1.0; // Set the pressure value for the source point
    sourcePoints.push_back(sourcePoint);

    // printf("Source Point: %f %f %f\n", sourcePoint->position.x, sourcePoint->position.y, sourcePoint->position.z);
}

void ModelTes::addTargetObject(TargetObject *object)
{
    // Add the target object to the model
    // This is a placeholder for the actual implementation
    targetObjects.push_back(object);
}

void ModelTes::set_inital_conditions(float cp, float t_frequency, float attenuation, float t_density)
{
    medium_waveSpeed = cp;
    frequency = t_frequency;
    medium_attenuation = attenuation;
    density = t_density;
    omega = 2 * M_PI * frequency;
    k = omega / cp;
    k_wave.r = (double)k;
    k_wave.i = (double)attenuation;
    pixel_length = medium_waveSpeed / (frequency * resolution_factor);

    cout << "-------------------------------------------\n";
    cout << "Medium Wave Speed: " << medium_waveSpeed << endl;
    cout << "Frequency: " << frequency << endl;
    cout << "Medium Attenuation: " << medium_attenuation << endl;
    cout << "Medium Density: " << density << endl;
    cout << "Omega: " << omega << endl;
    cout << "Wave Number: " << k << endl;
    cout << "Pixel Length: " << pixel_length << endl;
    cout << "Resolution Factor: " << resolution_factor << endl;
}

void ModelTes::pixelate_facets()
{
    // Iterate over each target object and its facets
    for (auto &targetObject : targetObjects)
    {
        targetObject->MakePixelData(pixel_length);
    }
}

void ModelTes::copyToDevice()
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

void ModelTes::GetFieldPointPressures(dcomplex *field_points_pressure, int NumPoints)
{
    // Copy the pressure values from the device to the host
    GetFieldPointValGPU(field_points_pressure);
}

void ModelTes::RenderOpenGL()
{
    // Call the OpenGL rendering function
    // This is a placeholder for the actual implementation

    InitOpenGL();
    std::cout << "OpenGL initialized successfully." << std::endl;
    // MakeObjectOnGL(targetObjects[0]->facets);
    RenderGL();
}