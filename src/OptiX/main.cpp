#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iomanip>
#include <vector>
#include <iostream>
#include <cstring>

#define OPTIX_CHECK(call)                                                     \
    do                                                                        \
    {                                                                         \
        OptixResult res = call;                                               \
        if (res != OPTIX_SUCCESS)                                             \
        {                                                                     \
            fprintf(stderr, "OptiX call (%s) failed with code %d at %s:%d\n", \
                    #call, res, __FILE__, __LINE__);                          \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

typedef struct
{
    OptixTraversableHandle handle;
    float3 ray_origin;
    float3 ray_direction;
    int *output;
} Params;

std::string readFile(const std::string &filename)
{
    std::string contents;
    std::ifstream file(filename, std::ios::binary);
    if (file.good())
    {
        std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
        contents.assign(buffer.begin(), buffer.end());
    }
    else
    {
        std::cerr << "Error opening " << filename << ": " << strerror(errno) << "\n";
    }
    return contents;
}

int main()
{
    // Initialize CUDA
    cudaFree(0);

    // Initialize OptiX
    OPTIX_CHECK(optixInit());

    // Create OptiX device context
    OptixDeviceContext context = 0;
    OPTIX_CHECK(optixDeviceContextCreate(0, 0, &context));

    // Check maximum trace depth
    unsigned int maxTraceDepth = 0;
    OPTIX_CHECK(optixDeviceContextGetProperty(context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH, &maxTraceDepth, sizeof(maxTraceDepth)));
    std::cout << "Maximum trace depth: " << maxTraceDepth << std::endl;

    // Create geometry (a single triangle)
    float3 vertices[] = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}};
    CUdeviceptr d_vertices;
    cudaMalloc((void **)&d_vertices, 3 * sizeof(float3));
    cudaError_t err = cudaMemcpy((void *)d_vertices, vertices, 3 * sizeof(float3), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy failed for d_vertices: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    std::cout << "Vertices copied to d_vertices successfully." << std::endl;

    CUdeviceptr d_indices = 0;
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    buildInput.triangleArray.numVertices = 3;
    buildInput.triangleArray.vertexBuffers = &d_vertices;
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
    buildInput.triangleArray.numIndexTriplets = 0;
    buildInput.triangleArray.indexBuffer = 0; // No index buffer
    buildInput.triangleArray.numSbtRecords = 1;
    const unsigned int triangleFlags = OPTIX_GEOMETRY_FLAG_NONE;
    buildInput.triangleArray.flags = &triangleFlags;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    std::cout << "Number of vertices: " << buildInput.triangleArray.numVertices << std::endl;
    std::cout << "Vertex buffer address: " << d_vertices << std::endl;
    std::cout << "Vertex stride in bytes: " << buildInput.triangleArray.vertexStrideInBytes << std::endl;
    std::cout << "Index format: " << buildInput.triangleArray.indexFormat << std::endl;
    std::cout << "Number of index triplets: " << buildInput.triangleArray.numIndexTriplets << std::endl;

    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &gasBufferSizes));

    std::cout << "Temporary buffer size: " << gasBufferSizes.tempSizeInBytes << std::endl;
    std::cout << "Output buffer size: " << gasBufferSizes.outputSizeInBytes << std::endl;

    CUdeviceptr d_tempBuffer, d_outputBuffer;
    cudaMalloc((void **)&d_tempBuffer, gasBufferSizes.tempSizeInBytes);
    cudaMalloc((void **)&d_outputBuffer, gasBufferSizes.outputSizeInBytes);

    OptixTraversableHandle gasHandle;
    OPTIX_CHECK(optixAccelBuild(context, 0, &accelOptions, &buildInput, 1, d_tempBuffer,
                                gasBufferSizes.tempSizeInBytes, d_outputBuffer, gasBufferSizes.outputSizeInBytes,
                                &gasHandle, NULL, 0));

    cudaFree((void *)d_tempBuffer);
    std::cout << "Made the gasHandle." << std::endl;

    // Create pipeline
    OptixPipeline pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 1;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;

    OptixModule module;
    OptixModuleCompileOptions moduleCompileOptions = {};
    char log[2048];
    size_t logSize = sizeof(log);

    std::string device_programs = readFile("./device_programs.ptx");
    const char *dev_prog = device_programs.c_str();
    size_t dev_prog_sz = device_programs.size();

    OPTIX_CHECK(optixModuleCreateFromPTX(context, &moduleCompileOptions, &pipelineCompileOptions,
                                         dev_prog, dev_prog_sz, log, &logSize, &module));

    std::cout << "Module created successfully." << std::endl;

    OptixProgramGroup raygenProgramGroup;
    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module = module;
    raygenDesc.raygen.entryFunctionName = "__raygen__rg";

    OptixProgramGroupOptions programGroupOptions = {};
    OPTIX_CHECK(optixProgramGroupCreate(context, &raygenDesc, 1, &programGroupOptions, log, &logSize, &raygenProgramGroup));

    OptixProgramGroup missProgramGroup;
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module;                  // Use the same module as the raygen program
    missDesc.miss.entryFunctionName = "__miss__ms"; // Name of the miss program in PTX

    OPTIX_CHECK(optixProgramGroupCreate(context, &missDesc, 1, &programGroupOptions, log, &logSize, &missProgramGroup));
    std::cout << "Miss program group created successfully." << std::endl;

    OptixProgramGroup hitgroupProgramGroup;
    OptixProgramGroupDesc hitgroupDesc = {};
    hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupDesc.hitgroup.moduleCH = module;                        // Closest-hit program module
    hitgroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch"; // Closest-hit program in PTX
    hitgroupDesc.hitgroup.moduleAH = nullptr;                       // No any-hit program
    hitgroupDesc.hitgroup.entryFunctionNameAH = nullptr;            // No any-hit program
    hitgroupDesc.hitgroup.moduleIS = nullptr;                       // No intersection program
    hitgroupDesc.hitgroup.entryFunctionNameIS = nullptr;            // No intersection program

    OPTIX_CHECK(optixProgramGroupCreate(context, &hitgroupDesc, 1, &programGroupOptions, log, &logSize, &hitgroupProgramGroup));
    std::cout << "Hit group program group created successfully." << std::endl;

    std::cout << "Raygen program group created successfully." << std::endl;
    struct RaygenRecord
    {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    RaygenRecord rgRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenProgramGroup, &rgRecord));

    CUdeviceptr raygenRecord;
    cudaMalloc((void **)&raygenRecord, sizeof(RaygenRecord));

    // Copy the ray generation record from the host to the device
    cudaMemcpy((void *)raygenRecord, &rgRecord, sizeof(RaygenRecord), cudaMemcpyHostToDevice);

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = raygenRecord;

    struct MissRecord
    {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    MissRecord msRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(missProgramGroup, &msRecord));
    CUdeviceptr missRecord;
    cudaMalloc((void **)&missRecord, sizeof(MissRecord));
    cudaMemcpy((void *)missRecord, &msRecord, sizeof(MissRecord), cudaMemcpyHostToDevice);
    sbt.missRecordBase = missRecord;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = 1;

    struct HitgroupRecord
    {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    HitgroupRecord hgRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupProgramGroup, &hgRecord));
    CUdeviceptr hitgroupRecord;
    cudaMalloc((void **)&hitgroupRecord, sizeof(HitgroupRecord));
    cudaMemcpy((void *)hitgroupRecord, &hgRecord, sizeof(HitgroupRecord), cudaMemcpyHostToDevice);
    sbt.hitgroupRecordBase = hitgroupRecord;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = 1;

    std::cout << "yah..." << std::endl;
    std::cout << "Made shader binding table." << std::endl;

    OptixProgramGroup programGroups[] = {raygenProgramGroup, missProgramGroup, hitgroupProgramGroup};
    OPTIX_CHECK(optixPipelineCreate(context, &pipelineCompileOptions, &pipelineLinkOptions, programGroups, 3, log, &logSize, &pipeline));

    if (logSize > 0)
    {
        std::cerr << "OptiX Pipeline Creation Log: " << log << std::endl;
    }
    std::cout << "Pipeline created successfully." << std::endl;

    // Set up launch parameters
    Params params;
    params.handle = gasHandle;
    params.ray_origin = (float3){0.5f, 0.5f, -1.0f};
    params.ray_direction = (float3){0.0f, 0.0f, 1.0f};
    int output;
    cudaMalloc((void **)&params.output, sizeof(int));
    cudaMemcpy(params.output, &output, sizeof(int), cudaMemcpyHostToDevice);

    CUdeviceptr d_params;
    cudaMalloc((void **)&d_params, sizeof(Params));
    cudaMemcpy((void *)d_params, &params, sizeof(Params), cudaMemcpyHostToDevice);

    std::cout << "Launching Program...\n";

    // Launch the pipeline
    OPTIX_CHECK(optixLaunch(pipeline, 0, d_params, sizeof(Params), &sbt, 1, 1, 1));

    // Retrieve the result
    cudaMemcpy(&output, params.output, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Collision detected: %s\n", output ? "Yes" : "No");

    // Clean up
    cudaFree((void *)d_vertices);
    cudaFree((void *)d_outputBuffer);
    cudaFree((void *)raygenRecord);
    cudaFree((void *)params.output);
    cudaFree((void *)d_params);
    optixDeviceContextDestroy(context);

    return 0;
}