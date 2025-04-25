#include "Collision.hpp"
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <iostream>

// Structure to hold a triangle (facet)
#define OPTIX_CHECK(call)                                                          \
    do                                                                             \
    {                                                                              \
        OptixResult res = call;                                                    \
        if (res != OPTIX_SUCCESS)                                                  \
        {                                                                          \
            fprintf(stderr, "OptiX call (%s) failed with code %d (%s) at %s:%d\n", \
                    #call, res, optixGetErrorString(res), __FILE__, __LINE__);     \
            exit(1);                                                               \
        }                                                                          \
    } while (0)

Collision::~Collision()
{
    StopCollision();
}

int Collision::StopCollision()
{
    if (d_optxVertices)
    {
        cuMemFree(d_optxVertices);
        d_optxVertices = 0;
    }
    if (optxContext)
    {
        optixDeviceContextDestroy(optxContext);
        optxContext = 0;
    }
    gasHandle = 0;

    return 0;
}

void Collision::FreePrams()
{
    if (d_optix_params)
    {
        cudaFree(h_optix_params.vp1);
        cudaFree(h_optix_params.vp2);
        cudaFree(h_optix_params.output);
        cudaFree(d_optix_params);
        d_optix_params = 0;
    }
}

void Collision::DoCollisions(std::vector<float3> vp1, std::vector<float3> vp2)
{
    FreePrams();
    int numSrc = vp1.size();
    int numDst = vp2.size();

    h_optix_params.handle = gasHandle;
    h_optix_params.numSrcPoints = numSrc;
    cudaMalloc((void **)&(h_optix_params.vp1), numSrc * sizeof(float3));
    cudaMemcpy(h_optix_params.vp1, vp1.data(), numSrc * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&(h_optix_params.vp2), numDst * sizeof(float3));
    cudaMemcpy(h_optix_params.vp2, vp2.data(), numDst * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&(h_optix_params.output), numDst * sizeof(int));
    cudaMemset(h_optix_params.output, 0, numDst * sizeof(int));

    cudaMalloc(&d_optix_params, sizeof(Params));
    cudaMemcpy(d_optix_params, &h_optix_params, sizeof(Params), cudaMemcpyHostToDevice);

    optixLaunch(
        pipeline,
        stream,
        reinterpret_cast<CUdeviceptr>(&d_optix_params),
        sizeof(Params),
        &sbt,
        numSrc,
        numDst,
        1);
}

bool Collision::HasCollision(float3 p1, float3 p2)
{
    std::vector<float3> vp1(1, p1);
    std::vector<float3> vp2(1, p2);
    DoCollisions(vp1, vp2);
    return true;
}

Collision::Collision()
{
    StopCollision();

    OPTIX_CHECK(optixInit());
    optixDeviceContextCreate(0, 0, &optxContext);

    cuStreamCreate(&stream, CU_STREAM_DEFAULT);
}

// Function to load facets into OptiX
int Collision::CreateGeometry(const std::vector<Triangle> &facets)
{
    // Allocate device memory for vertices
    size_t verticesSize = facets.size() * sizeof(Triangle);
    cuMemAlloc(&d_optxVertices, verticesSize);
    cuMemcpyHtoD(d_optxVertices, facets.data(), verticesSize);

    // Build GAS (Geometry Acceleration Structure)
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // Set up triangle data
    CUdeviceptr d_indices = 0; // No indices for raw triangle data
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.numVertices = facets.size() * 3;
    buildInput.triangleArray.vertexBuffers = &d_optxVertices;
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
    buildInput.triangleArray.numIndexTriplets = 0;
    buildInput.triangleArray.indexBuffer = d_indices;

    // Acceleration structure options
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Allocate memory for GAS output
    OptixAccelBufferSizes gasBufferSizes;
    optixAccelComputeMemoryUsage(optxContext, &accelOptions, &buildInput, 1, &gasBufferSizes);

    CUdeviceptr d_tempBuffer, d_outputBuffer;
    cuMemAlloc(&d_tempBuffer, gasBufferSizes.tempSizeInBytes);
    cuMemAlloc(&d_outputBuffer, gasBufferSizes.outputSizeInBytes);

    optixAccelBuild(optxContext, 0, &accelOptions, &buildInput, 1, d_tempBuffer,
                    gasBufferSizes.tempSizeInBytes, d_outputBuffer, gasBufferSizes.outputSizeInBytes,
                    &gasHandle, nullptr, 0);

    cuMemFree(d_tempBuffer);
    return 0;
}

int Collision::MakeShaderBufferTable()
{
    // The SBT sets up program to run to define what rays are render, and what to do if a hit or miss occurs.
    // This in not being used in this application.

    // Declare and allocate memory for SBT records
    //     CUdeviceptr raygenRecord;
    //     CUdeviceptr missRecord;
    //     CUdeviceptr hitgroupRecord;

    //     // Ensure RaygenRecord, HitgroupRecord, raygenProgramGroup, and hitgroupProgramGroup are defined
    //     struct RaygenRecord {
    //         char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    //     };

    //     struct HitgroupRecord {
    //         char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    //     };

    //     extern OptixProgramGroup raygenProgramGroup;
    //     extern OptixProgramGroup hitgroupProgramGroup;

    // // Allocate memory for the raygen record
    // size_t raygenRecordSize = sizeof(RaygenRecord);
    // cudaMalloc(reinterpret_cast<void **>(&raygenRecord), raygenRecordSize);
    // RaygenRecord rgRecord = {};
    // rgRecord.header = optixSbtRecordPackHeader(raygenProgramGroup);
    // cudaMemcpy(reinterpret_cast<void *>(raygenRecord), &rgRecord, raygenRecordSize, cudaMemcpyHostToDevice);

    // // Allocate memory for the miss record
    // size_t missRecordSize = sizeof(MissRecord);
    // cudaMalloc(reinterpret_cast<void **>(&missRecord), missRecordSize);
    // MissRecord msRecord = {};
    // msRecord.header = optixSbtRecordPackHeader(missProgramGroup);
    // cudaMemcpy(reinterpret_cast<void *>(missRecord), &msRecord, missRecordSize, cudaMemcpyHostToDevice);

    // // Allocate memory for the hitgroup record
    // size_t hitgroupRecordSize = sizeof(HitgroupRecord);
    // cudaMalloc(reinterpret_cast<void **>(&hitgroupRecord), hitgroupRecordSize);
    // HitgroupRecord hgRecord = {};
    // hgRecord.header = optixSbtRecordPackHeader(hitgroupProgramGroup);
    // cudaMemcpy(reinterpret_cast<void *>(hitgroupRecord), &hgRecord, hitgroupRecordSize, cudaMemcpyHostToDevice);

    // // Set up the SBT
    // sbt.raygenRecord = raygenRecord;
    // sbt.missRecordBase = missRecord;
    // sbt.missRecordStrideInBytes = sizeof(MissRecord);
    // sbt.missRecordCount = 1;
    // sbt.hitgroupRecordBase = hitgroupRecord;
    // sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    // sbt.hitgroupRecordCount = 1;

    return 0;
}

int Collision::MakePipeline()
{
    OptixModule module;
    OptixModuleCompileOptions moduleCompileOptions = {};
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    char log[2048];
    size_t logSize = sizeof(log);

    // Create the OptiX module from PTX
    OPTIX_CHECK(optixModuleCreateFromPTX(
        optxContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        "my_program.optixir", // Path to the IR/PTX file
        log,
        &logSize,
        &module));

    // Create program groups
    // OptixProgramGroup raygenProgramGroup;
    // OptixProgramGroup missProgramGroup;
    // OptixProgramGroup hitgroupProgramGroup;

    // OptixProgramGroupOptions programGroupOptions = {}; // Default options
    // OptixProgramGroupDesc raygenDesc = {};
    // raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    // raygenDesc.raygen.module = module;
    // raygenDesc.raygen.entryFunctionName = "__raygen__rg";

    // OPTIX_CHECK(optixProgramGroupCreate(
    //     optxContext,
    //     &raygenDesc,
    //     1, // Number of program groups
    //     &programGroupOptions,
    //     log,
    //     &logSize,
    //     &raygenProgramGroup));

    // OptixProgramGroupDesc missDesc = {};
    // missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    // missDesc.miss.module = module;
    // missDesc.miss.entryFunctionName = "__miss__ms";

    // OPTIX_CHECK(optixProgramGroupCreate(
    //     optxContext,
    //     &missDesc,
    //     1,
    //     &programGroupOptions,
    //     log,
    //     &logSize,
    //     &missProgramGroup));

    // OptixProgramGroupDesc hitgroupDesc = {};
    // hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    // hitgroupDesc.hitgroup.moduleCH = module;
    // hitgroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

    // OPTIX_CHECK(optixProgramGroupCreate(
    //     optxContext,
    //     &hitgroupDesc,
    //     1,
    //     &programGroupOptions,
    //     log,
    //     &logSize,
    //     &hitgroupProgramGroup));

    // // Array of program groups
    // OptixProgramGroup programGroups[] = {raygenProgramGroup, missProgramGroup, hitgroupProgramGroup};

    // // Configure pipeline compile options
    // pipelineCompileOptions.usesMotionBlur = false;
    // pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    // pipelineCompileOptions.numPayloadValues = 1;   // Number of payload registers
    // pipelineCompileOptions.numAttributeValues = 0; // Number of attribute registers
    // pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    // pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    // // Configure pipeline link options
    // pipelineLinkOptions.maxTraceDepth = 1;

    // // Create the pipeline
    // OPTIX_CHECK(optixPipelineCreate(
    //     optxContext,
    //     &pipelineCompileOptions,
    //     &pipelineLinkOptions,
    //     programGroups,                                    // Array of program groups
    //     sizeof(programGroups) / sizeof(programGroups[0]), // Number of program groups
    //     log,
    //     &logSize,
    //     &pipeline)); // Output pipeline handle

    return 0;
}

/**
 * @brief Pack all the facets into OptiX.
 */
int Collision::StartCollision(std::vector<Object> &targetObjects)
{
    int numFacets = 0;
    for (auto object : targetObjects)
    {
        // Create geometry for each object
        numFacets += object.facets.size();
    }
    std::vector<Triangle> facets(numFacets);
    int i = 0;
    for (auto object : targetObjects)
    {
        for (auto facet : object.facets)
        {
            facets[i].v0 = facet->v1;
            facets[i].v1 = facet->v2;
            facets[i].v2 = facet->v3;
            i++;
        }
    }
    CreateGeometry(facets);
    return 0;
}
