#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>

// Structure to hold a triangle (facet)
struct Triangle
{
    float3 v0, v1, v2;
};

// Function to load facets into OptiX
OptixTraversableHandle createGeometry(const std::vector<Triangle> &facets, CUdeviceptr &d_vertices)
{
    // Allocate device memory for vertices
    size_t verticesSize = facets.size() * sizeof(Triangle);
    cudaMalloc(&d_vertices, verticesSize);
    cudaMemcpy(d_vertices, facets.data(), verticesSize, cudaMemcpyHostToDevice);

    // Build GAS (Geometry Acceleration Structure)
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // Set up triangle data
    CUdeviceptr d_indices = 0; // No indices for raw triangle data
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.numVertices = facets.size() * 3;
    buildInput.triangleArray.vertexBuffers = &d_vertices;
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
    buildInput.triangleArray.numIndexTriplets = 0;
    buildInput.triangleArray.indexBuffer = d_indices;

    // Acceleration structure options
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Allocate memory for GAS output
    OptixAccelBufferSizes gasBufferSizes;
    optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &gasBufferSizes);

    CUdeviceptr d_tempBuffer, d_outputBuffer;
    cudaMalloc(&d_tempBuffer, gasBufferSizes.tempSizeInBytes);
    cudaMalloc(&d_outputBuffer, gasBufferSizes.outputSizeInBytes);

    OptixTraversableHandle gasHandle;
    optixAccelBuild(context, 0, &accelOptions, &buildInput, 1, d_tempBuffer,
                    gasBufferSizes.tempSizeInBytes, d_outputBuffer, gasBufferSizes.outputSizeInBytes,
                    &gasHandle, nullptr, 0);

    cudaFree(d_tempBuffer);
    return gasHandle;
}

extern "C" __global__ void __raygen__rg()
{
    const float3 origin = make_float3(optixGetPayload_0(), optixGetPayload_1(), optixGetPayload_2());
    const float3 direction = make_float3(optixGetPayload_3(), optixGetPayload_4(), optixGetPayload_5());

    unsigned int hit = 0;
    optixTrace(
        params.handle,
        origin,
        direction,
        0.0f,  // Min t
        1e16f, // Max t
        0.0f,  // Ray time
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0, // SBT offset
        1, // SBT stride
        0, // Miss SBT index
        hit);

    // Write the result to the output buffer
    params.output[optixGetLaunchIndex().x] = hit;
}

extern "C" __global__ void __closesthit__ch()
{
    // Mark the ray as hitting geometry
    optixSetPayload_0(1);
}

extern "C" __global__ void __miss__ms()
{
    // Mark the ray as not hitting any geometry
    optixSetPayload_0(0);
}

void checkCollision(OptixTraversableHandle gasHandle, const float3 &pointA, const float3 &pointB)
{
    // Calculate ray direction
    float3 direction = normalize(pointB - pointA);

    // Set up ray payload
    unsigned int hit = 0;

    // Define a small epsilon to exclude the destination point
    const float epsilon = 1e-5f;

    // Trace the ray
    optixTrace(
        gasHandle,
        pointA,
        direction,
        0.0f,                              // Min t
        length(pointB - pointA) - epsilon, // Max t
        0.0f,                              // Ray time
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0, // SBT offset
        1, // SBT stride
        0, // Miss SBT index
        hit);

    if (hit)
    {
        printf("Collision detected between points!\n");
    }
    else
    {
        printf("No collision, clear path.\n");
    }
}

int main()
{
    // Initialize OptiX
    OptixDeviceContext context;
    optixInit();
    optixDeviceContextCreate(0, 0, &context);

    // Define facets (triangles)
    std::vector<Triangle> facets = {
        {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}},
        {{1, 0, 0}, {1, 1, 0}, {0, 1, 0}}};

    // Create geometry
    CUdeviceptr d_vertices;
    OptixTraversableHandle gasHandle = createGeometry(facets, d_vertices);

    // Check collision between two points
    float3 pointA = {0.5f, 0.5f, -1.0f};
    float3 pointB = {0.5f, 0.5f, 1.0f};
    checkCollision(gasHandle, pointA, pointB);

    // Clean up
    cudaFree(d_vertices);
    optixDeviceContextDestroy(context);
    return 0;
}