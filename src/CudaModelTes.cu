#include "CudaModelTes.cuh"
#include "GeoMath.h"
#include "dcomplex.h"
#include "CudaUtils.cuh"

#include <stdio.h>

using namespace std;

int CudaModelTes::SetGlobalParameters(dcomplex k_wave, float pixel_delta)
{
    printf("SetGlobalParameters .......\n");
    // Set the global parameters for the GPU
    cudaMalloc(&dev_k_wave, 1 * sizeof(dcomplex));
    cudaError_t cudaStatus = cudaMemcpy(dev_k_wave, &k_wave, sizeof(dcomplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "k_wave failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    printf("pixel_delta: %f\n", pixel_delta);
    cudaMalloc(&dev_pixel_delta, 1 * sizeof(float));
    cudaStatus = cudaMemcpy(dev_pixel_delta, &pixel_delta, sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "pixel_delta failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    cudaMalloc(&dev_pixel_Pressure_stats, 3 * sizeof(float));
    return 0;
}

int CudaModelTes::MakeSourcePointsOnGPU(vector<PressurePoint *> source_points)
{
    host_num_source_points = source_points.size();
    printf("MakeSourcePointsOnGPU (num pnts %d) .......\n", host_num_source_points);

    auto position = new float3[host_num_source_points];
    auto pressure = new dcomplex[host_num_source_points];
    for (int i = 0; i < host_num_source_points; i++)
    {
        position[i] = source_points[i]->position;
        pressure[i] = source_points[i]->pressure;
    }
    // Allocate memory for the source points on the device
    cudaMalloc(&dev_source_points_position, host_num_source_points * sizeof(float3));
    cudaMemcpy(dev_source_points_position, position, host_num_source_points * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_source_points_pressure, host_num_source_points * sizeof(dcomplex));
    cudaMemcpy(dev_source_points_pressure, pressure, host_num_source_points * sizeof(dcomplex), cudaMemcpyHostToDevice);
    // Free the host memory
    delete[] position;
    delete[] pressure;
    return 0;
}

int CudaModelTes::MakeFieldPointsOnGPU(vector<PressurePoint *> field_points)
{

    host_num_field_points = field_points.size();

    auto position = new float3[host_num_field_points];
    auto pressure = new dcomplex[host_num_field_points];
    for (int i = 0; i < host_num_field_points; i++)
    {
        position[i] = field_points[i]->position;
        pressure[i] = field_points[i]->pressure;
    }
    // Allocate memory for the source points on the device
    cudaMalloc(&dev_field_points_position, host_num_field_points * sizeof(float3));
    cudaMemcpy(dev_field_points_position, position, host_num_field_points * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_field_points_pressure, host_num_field_points * sizeof(dcomplex));
    cudaMemcpy(dev_field_points_pressure, pressure, host_num_field_points * sizeof(dcomplex), cudaMemcpyHostToDevice);
    // Free the host memory
    delete[] position;
    delete[] pressure;

    return 0;
}
/**
 * @brief Allocate the texture memory for the facet pressure.
 *
 * The surfaces are used to write/read the pressure values to the device memory.
 * The textures are used to render the pressure in OpenGL.
 */
void CudaModelTes::AllocateTexture(int num_xpnts,
                                   int num_ypnts,
                                   vector<cudaSurfaceObject_t> *dest_surface,
                                   vector<cudaArray_t> *dest_array)
{
    cudaError_t err;
    FacetGL *renderFacet;
    cudaArray_t d_array;

    // Allocate the texture memory for the facet pressure
    if (usingOpenGL)
    {
        renderFacet = new FacetGL();
        gl_object_facets.push_back(renderFacet);
        CreateTexture(num_xpnts, num_ypnts, &(renderFacet->textureID));
        err = cudaGraphicsGLRegisterImage(&renderFacet->cudaResource, renderFacet->textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
        if (err != cudaSuccess)
        {
            printf("cudaGraphicsGLRegisterImage failed: %s\n", cudaGetErrorString(err));
            return;
        }

        printf("Test 0.9\n");
        err = cudaGraphicsMapResources(1, &renderFacet->cudaResource, 0);
        if (err != cudaSuccess)
        {
            printf("cudaGraphicsMapResources failed: %s\n", cudaGetErrorString(err));
            return;
        }
        cudaGraphicsSubResourceGetMappedArray(&d_array, renderFacet->cudaResource, 0, 0);

        err = cudaGraphicsSubResourceGetMappedArray(&d_array, renderFacet->cudaResource, 0, 0);
        if (err != cudaSuccess)
        {
            printf("cudaGraphicsSubResourceGetMappedArray failed: %s\n", cudaGetErrorString(err));
            return;
        }
    }
    else
    {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        err = cudaMallocArray(&d_array, &channelDesc, num_xpnts, num_ypnts);
        if (err != cudaSuccess)
        {
            printf("cudaMallocArray failed: %s\n", cudaGetErrorString(err));
            return;
        }
    }

    cudaResourceDesc resDesc = {};
    memset(&resDesc, 0, sizeof(resDesc)); // Ensure all fields are zero-initialized
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_array;

    cudaTextureDesc texDesc = {};
    memset(&texDesc, 0, sizeof(texDesc)); // Ensure all fields are zero-initialized
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaSurfaceObject_t surface;
    err = cudaCreateSurfaceObject(&surface, &resDesc);
    if (err != cudaSuccess)
    {
        printf("cudaCreateSurfaceObject failed: %s\n", cudaGetErrorString(err));
        return;
    }
    dest_surface->push_back(surface);

    printf("Finished Allocating Texture.\n");
}

int CudaModelTes::MakeObjectOnGPU(vector<Facet *> facets)
{
    // Copy the facet data to the GPU
    printf("MakeObjectOnGPU .......\n");

    int number_of_facets = facets.size();
    host_object_num_facets.push_back(number_of_facets);

    float **dev_Facets_PixelArea;
    cudaMalloc(&dev_Facets_PixelArea, number_of_facets * sizeof(float *));
    dev_Object_Facets_PixelArea.push_back(dev_Facets_PixelArea);

    std::vector<double *> facet_Pi;
    std::vector<double *> facet_Pr;

    for (int i = 0; i < number_of_facets; i++)
    {
        int num_pixels = facets[i]->NumXpnts * facets[i]->NumYpnts;
        double *Pr;
        double *Pi;
        cudaMalloc((void **)&Pr, num_pixels * sizeof(double));
        cudaMemset(Pr, 0, num_pixels * sizeof(double));
        cudaMalloc((void **)&Pi, num_pixels * sizeof(double));
        cudaMemset(Pi, 0, num_pixels * sizeof(double));
        facet_Pr.push_back(Pr);
        facet_Pi.push_back(Pi);
    }

    dev_object_facet_Pr.push_back(facet_Pr);
    dev_object_facet_Pi.push_back(facet_Pi);

    vector<cudaSurfaceObject_t> surface_Pr;
    vector<cudaSurfaceObject_t> surface_Pi;

    vector<cudaArray_t> dev_array_Pr;
    vector<cudaArray_t> dev_array_Pi;

    vector<int *> pixel_mutex;

    float **host_PixelArea = new float *[number_of_facets];
    for (int i = 0; i < number_of_facets; i++)
    {

        AllocateTexture(facets[i]->NumXpnts,
                        facets[i]->NumYpnts,
                        &surface_Pr,
                        &dev_array_Pr);

        AllocateTexture(facets[i]->NumXpnts,
                        facets[i]->NumYpnts,
                        &surface_Pi,
                        &dev_array_Pi);

        int num_pixel = facets[i]->NumXpnts * facets[i]->NumYpnts;

        int *d_mutex_array;
        // Allocate memory for the mutex array on the device
        cudaError_t err = cudaMalloc(&d_mutex_array, num_pixel * sizeof(int));
        if (err != cudaSuccess)
        {
            printf("cudaMalloc failed for mutex array: %s\n", cudaGetErrorString(err));
            return 1;
        }

        // Initialize all mutexes to 0 (unlocked state)
        err = cudaMemset(d_mutex_array, 0, num_pixel * sizeof(int));
        if (err != cudaSuccess)
        {
            printf("cudaMemset failed for mutex array: %s\n", cudaGetErrorString(err));
            return 1;
        }
        pixel_mutex.push_back(d_mutex_array);

        int ArrLen = facets[i]->NumXpnts * facets[i]->NumYpnts;
        // Allocate the Area for the pixel array on the host.
        cudaMalloc(&host_PixelArea[i], ArrLen * sizeof(float));
        // Copy the pixel area data to host CUDA to the device.
        cudaMemcpy(host_PixelArea[i], facets[i]->PixelArea, ArrLen * sizeof(float), cudaMemcpyHostToDevice);
    }

    dev_Object_Facets_Surface_Pr.push_back(surface_Pr);
    dev_Object_Facets_Surface_Pi.push_back(surface_Pi);

    dev_Object_Facets_array_Pr.push_back(dev_array_Pr);
    dev_Object_Facets_array_Pi.push_back(dev_array_Pi);

    dev_Object_Facets_pixel_mutex.push_back(pixel_mutex);

    // Copy the pixel area data to the device.
    cudaMemcpy(dev_Facets_PixelArea, host_PixelArea, number_of_facets * sizeof(float *), cudaMemcpyHostToDevice);

    int facet_cnt = 0;
    auto host_Facets_points = new int3[number_of_facets];
    auto host_Facets_Normals = new float3[number_of_facets];
    auto host_base_points = new float3[number_of_facets];
    auto host_Facets_xAxis = new float3[number_of_facets];
    auto host_Facets_yAxis = new float3[number_of_facets];
    for (int i = 0; i < number_of_facets; i++)
    {
        // Memory for dev_Facets_PixelArea[facet_cnt] is already allocated in the initialization loop above.
        host_Facets_points[i].x = facets[i]->NumXpnts;
        host_Facets_points[i].y = facets[i]->NumYpnts;
        host_Facets_points[i].z = facets[i]->NumXpntsNegative;
        host_Facets_Normals[i] = facets[i]->Normal;
        host_base_points[i] = facets[i]->pointOnBase;
        host_Facets_xAxis[i] = facets[i]->xAxis;
        host_Facets_yAxis[i] = facets[i]->yAxis;
        facet_cnt++;
    }

    int3 *dev_Facets_points;
    cudaMalloc(&dev_Facets_points, number_of_facets * sizeof(int3));
    cudaMemcpy(dev_Facets_points, host_Facets_points, number_of_facets * sizeof(int3), cudaMemcpyHostToDevice);
    dev_Object_Facets_points.push_back(dev_Facets_points);
    host_Object_Facets_points.push_back(host_Facets_points);

    float3 *dev_Facets_Normals;
    cudaMalloc(&dev_Facets_Normals, number_of_facets * sizeof(float3));
    cudaMemcpy(dev_Facets_Normals, host_Facets_Normals, number_of_facets * sizeof(float3), cudaMemcpyHostToDevice);
    dev_Object_Facets_Normals.push_back(dev_Facets_Normals);

    float3 *dev_base_points;
    cudaMalloc(&dev_base_points, number_of_facets * sizeof(float3));
    cudaMemcpy(dev_base_points, host_base_points, number_of_facets * sizeof(float3), cudaMemcpyHostToDevice);
    dev_Object_base_points.push_back(dev_base_points);

    float3 *dev_Facets_xAxis;
    cudaMalloc(&dev_Facets_xAxis, number_of_facets * sizeof(float3));
    cudaMemcpy(dev_Facets_xAxis, host_Facets_xAxis, number_of_facets * sizeof(float3), cudaMemcpyHostToDevice);
    dev_Object_Facets_xAxis.push_back(dev_Facets_xAxis);

    float3 *dev_Facets_yAxis;
    cudaMalloc(&dev_Facets_yAxis, number_of_facets * sizeof(float3));
    cudaMemcpy(dev_Facets_yAxis, host_Facets_yAxis, number_of_facets * sizeof(float3), cudaMemcpyHostToDevice);
    dev_Object_Facets_yAxis.push_back(dev_Facets_yAxis);

    delete[] host_PixelArea;
    delete[] host_Facets_Normals;
    delete[] host_base_points;

    printf("Allocated object memory on GPU.\n");
    return 0;
}

int CudaModelTes::StartCuda()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        return 1;
    }
    printf("CUDA started successfully.\n");
    return 0;
}

int CudaModelTes::StopCuda()
{
    // cudaError_t cudaStatus = cudaUnbindTexture(dev_Positions);
    // if (cudaStatus != cudaSuccess)
    // {
    //     printf("Unbinding of Positions Texture Failed!\n");
    //     return 1;
    // }

    // cudaStatus = cudaDeviceReset();
    // if (cudaStatus != cudaSuccess)
    // {
    //     fprintf(stderr, "cudaDeviceReset failed!\n");
    //     return 1;
    // }
    return 0;
}

int CudaModelTes::DoCalculations()
{
    printf("DoCalculations .......\n");

    for (int object_num = 0; object_num < host_object_num_facets.size(); object_num++)
    {
        mutex_in_cuda.push_back(vector<int *>());
        for (int facet_num = 0; facet_num < host_object_num_facets[object_num]; facet_num++)
        {
            int *mutex;
            int value = 0;
            cudaError_t cudaStatus = cudaMalloc(&mutex, sizeof(int));
            if (cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "mutex failed: %s\n", cudaGetErrorString(cudaStatus));
                return 1;
            }
            cudaStatus = cudaMemcpy(mutex, &value, sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "mutex failed: %s\n", cudaGetErrorString(cudaStatus));
                return 1;
            }
            mutex_in_cuda[object_num].push_back(mutex);
        }
    }

    // TestGPU();
    if (ProjectSourcePointsToFacet() != 0)
    {
        printf("ProjectSourcePointsToFacet failed.\n");
        return 1;
    }

    for (int i = 0; i < 100; i++)
    {
        if (ProjectFromFacetsToFacets() != 0)
        {
            printf("ProjectFromFacetsToFacets failed.\n");
            return 1;
        }
    }

    if (ProjectFromFacetsToFieldPoints() != 0)
    {
        printf("ProjectFromFacetsToFieldPoints failed.\n");
        return 1;
    }
    return 0;
}

int CudaModelTes::GetFieldPointValGPU(dcomplex *field_points_pressure)
{
    // Copy the field point pressures from the device to the host.
    cudaMemcpy(field_points_pressure, dev_field_points_pressure, host_num_field_points * sizeof(dcomplex), cudaMemcpyDeviceToHost);
    return 0;
}

void CudaModelTes::CleanupCuda()
{
    for (auto &object : dev_Object_Facets_Surface_Pr)
    {
        for (auto &surface : object)
        {
            cudaDestroySurfaceObject(surface);
        }
    }
    for (auto &object : dev_Object_Facets_Surface_Pi)
    {
        for (auto &surface : object)
        {
            cudaDestroySurfaceObject(surface);
        }
    }
    if (!usingOpenGL)
    {
        for (auto &object : dev_Object_Facets_array_Pr)
        {
            for (auto &array : object)
            {
                cudaFreeArray(array);
            }
        }
        for (auto &object : dev_Object_Facets_array_Pi)
        {
            for (auto &array : object)
            {
                cudaFreeArray(array);
            }
        }
    }
}