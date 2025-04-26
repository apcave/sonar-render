#include "Collision.hpp"
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include "OptiX/Exception.h"
#include <fstream>

// // Structure to hold a triangle (facet)
// #define OPTIX_CHECK(call)                                                          \
//     do                                                                             \
//     {                                                                              \
//         OptixResult res = call;                                                    \
//         if (res != OPTIX_SUCCESS)                                                  \
//         {                                                                          \
//             fprintf(stderr, "OptiX call (%s) failed with code %d (%s) at %s:%d\n", \
//                     #call, res, optixGetErrorString(res), __FILE__, __LINE__);     \
//             exit(1);                                                               \
//         }                                                                          \
//     } while (0)

std::string Collision::readFile(const std::string &filename)
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

Collision::~Collision()
{
    StopCollision();
    if (context != 0)
    {
        std::cout << "Destroying context." << std::endl;
        optixDeviceContextDestroy(context);
        context = 0;
    }
}

int Collision::StopCollision()
{
    std::cout << "Stopping Collision." << std::endl;
    if (d_vertices != 0)
    {
        std::cout << "Freeing d_vertices." << std::endl;
        cudaFree((void *)d_vertices);
        d_vertices = 0;
    }

    if (d_outputBuffer != 0)
    {
        std::cout << "Freeing d_outputBuffer." << std::endl;
        cudaFree((void *)d_outputBuffer);
        d_outputBuffer = 0;
    }
    FreePrams();

    return 0;
}

void Collision::FreePrams()
{

    std::cout << "Freeing d_optix_params." << std::endl;

    if (h_optix_params.vp1)
    {
        std::cout << "Freeing h_optix_params.vp1." << std::endl;
        CUDA_CHECK(cudaFree(h_optix_params.vp1));
        h_optix_params.vp1 = 0;
    }

    if (h_optix_params.vp2)
    {
        std::cout << "Freeing h_optix_params.vp2." << std::endl;

        CUDA_CHECK(cudaFree(h_optix_params.vp2));
        h_optix_params.vp2 = 0;
    }

    if (h_optix_params.output)
    {
        std::cout << "Freeing h_optix_params.output." << std::endl;
        CUDA_CHECK(cudaFree(h_optix_params.output));
        h_optix_params.output = 0;
    }

    if (d_optix_params)
    {
        std::cout << "Freeing d_optix_params." << std::endl;
        CUDA_CHECK(cudaFree((void *)d_optix_params));
        d_optix_params = 0;
    }
}

int *Collision::DoCollisions(std::vector<float3> &vp1, std::vector<float3> &vp2)
{
    FreePrams();
    int numSrc = vp1.size();
    int numDst = vp2.size();

    h_optix_params.handle = gasHandle;
    h_optix_params.numSrcPoints = numSrc;

    size_t numBytes = numSrc * sizeof(float3) + numDst * sizeof(float3) + numDst * numSrc * sizeof(int) + sizeof(Params);
    CUDA_CHECK(cudaMalloc((void **)&(h_optix_params.vp1), numSrc * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(h_optix_params.vp1, vp1.data(), numSrc * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&(h_optix_params.vp2), numDst * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(h_optix_params.vp2, vp2.data(), numDst * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&(h_optix_params.output), numDst * numSrc * sizeof(int)));
    CUDA_CHECK(cudaMemset(h_optix_params.output, 0, numDst * sizeof(int)));

    CUDA_CHECK(cudaMalloc((void **)&d_optix_params, sizeof(Params)));
    CUDA_CHECK(cudaMemcpy((void *)d_optix_params, &h_optix_params, sizeof(Params), cudaMemcpyHostToDevice));

    std::cout << "SBT Raygen Record: " << sbt.raygenRecord << std::endl;
    std::cout << "SBT Miss Record Base: " << sbt.missRecordBase << std::endl;
    std::cout << "SBT Hitgroup Record Base: " << sbt.hitgroupRecordBase << std::endl;

    std::cout << "Params.handle: " << h_optix_params.handle << std::endl;
    std::cout << "Params.numSrcPoints: " << h_optix_params.numSrcPoints << std::endl;
    std::cout << "Params.vp1: " << h_optix_params.vp1 << std::endl;
    std::cout << "Params.vp2: " << h_optix_params.vp2 << std::endl;
    std::cout << "Params.output: " << h_optix_params.output << std::endl;

    std::cout << "Number of source points: " << numSrc << std::endl;
    std::cout << "Number of destination points: " << numDst << std::endl;

    std::cout << "pipeline : " << pipeline << std::endl;

    std::cout << "Launching Collision Program...\n";
    OPTIX_CHECK(optixLaunch(
        pipeline,
        0,
        d_optix_params,
        sizeof(Params),
        &sbt,
        numSrc,
        numDst,
        1));
    std::cout << "Launched Collision Program.\n";
    cudaDeviceSynchronize();

    int *hasCollided = new int[numSrc * numDst];
    cudaMemcpy(hasCollided, h_optix_params.output, numSrc * numDst * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Collision results:" << std::endl;
    for (int i = 0; i < numSrc; i++)
    {
        for (int j = 0; j < numDst; j++)
        {
            // std::cout << hasCollided[i * numDst + j] << std::endl;
            if (hasCollided[i * numDst + j])
            {
                std::cout << " 1";
            }
            else
            {
                std::cout << " 0";
            }
        }
        std::cout << std::endl;
    }
    std::cout << "Done with collision results." << std::endl;
    // delete[] hasCollided;
    std::cout << "hasCollided." << std::endl;
    return hasCollided;
}

bool Collision::HasCollision(float3 p1, float3 p2)
{
    std::vector<float3> vp1(1, p1);
    std::vector<float3> vp2(1, p2);
    DoCollisions(vp1, vp2);
    return true;
}

void Collision::StartOptix()
{
    std::cout << "Collision::StartOptix()" << std::endl;
    OPTIX_CHECK(optixInit());
    OPTIX_CHECK(optixDeviceContextCreate(0, 0, &context));
    MakePipeline();
}

Collision::Collision()
{

    std::cout << "Made Collision object." << std::endl;

    // cuStreamCreate(&stream, CU_STREAM_DEFAULT);
}

// Function to load facets into OptiX
int Collision::CreateGeometry(const std::vector<Triangle> &facets)
{
    std::cout << "Collision Creating Geometry object." << std::endl;

    // Check maximum trace depth
    unsigned int maxTraceDepth = 0;
    OPTIX_CHECK(optixDeviceContextGetProperty(context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH, &maxTraceDepth, sizeof(maxTraceDepth)));
    std::cout << "Maximum trace depth: " << maxTraceDepth << std::endl;

    std::cout << "Number of facets :" << facets.size() << std::endl;

    // // Allocate device memory for vertices
    size_t verticesSize = facets.size() * 3 * sizeof(float3);
    CUDA_CHECK(cudaMalloc((void **)&d_vertices, verticesSize));
    CUDA_CHECK(cudaMemcpy((void *)d_vertices, facets.data(), verticesSize, cudaMemcpyHostToDevice));

    CUdeviceptr d_indices = 0;
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    buildInput.triangleArray.numVertices = 3 * facets.size();
    buildInput.triangleArray.vertexBuffers = &d_vertices;
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
    buildInput.triangleArray.numIndexTriplets = 0;
    buildInput.triangleArray.indexBuffer = 0; // No index buffer
    buildInput.triangleArray.numSbtRecords = 1;
    const unsigned int triangleFlags = OPTIX_GEOMETRY_FLAG_NONE;
    buildInput.triangleArray.flags = &triangleFlags;
    buildInput.triangleArray.sbtIndexOffsetBuffer = 0;
    buildInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    buildInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    // accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
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

    CUdeviceptr d_tempBuffer = 0;
    CUDA_CHECK(cudaMalloc((void **)&d_tempBuffer, gasBufferSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc((void **)&d_outputBuffer, gasBufferSizes.outputSizeInBytes));

    // Debugging output before optixAccelBuild
    std::cout << "Calling optixAccelBuild..." << std::endl;
    std::cout << "Context: " << context << std::endl;
    std::cout << "Temporary buffer address: " << d_tempBuffer << std::endl;
    std::cout << "Output buffer address: " << d_outputBuffer << std::endl;
    std::cout << "d_vertices buffer address: " << d_vertices << std::endl;

    OPTIX_CHECK(optixAccelBuild(context, 0, &accelOptions, &buildInput, 1, d_tempBuffer,
                                gasBufferSizes.tempSizeInBytes, d_outputBuffer, gasBufferSizes.outputSizeInBytes,
                                &gasHandle, NULL, 0));

    std::cout << "Made the gasHandle." << std::endl;
    cudaFree((void *)d_tempBuffer);

    std::cout << "Made the gasHandle." << std::endl;
    std::cout << "Vertices copied to d_vertices successfully." << std::endl;

    // MakePipeline();

    return 0;
}

int Collision::MakePipeline()
{
    // Create pipeline
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 1;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    OptixModule module;
    OptixModuleCompileOptions moduleCompileOptions = {};
    char log[2048];
    size_t logSize = sizeof(log);

    std::string device_programs = readFile("./build/Collision.ptx");
    const char *dev_prog = device_programs.c_str();
    size_t dev_prog_sz = device_programs.size();

    std::cout << "Program size: " << dev_prog_sz << std::endl;

    log[0] = '\0';         // Clear the log buffer
    logSize = sizeof(log); // Reset the log size
    OPTIX_CHECK(optixModuleCreateFromPTX(context, &moduleCompileOptions, &pipelineCompileOptions,
                                         dev_prog, dev_prog_sz, log, &logSize, &module));

    if (logSize > 0)
    {
        std::cout << "Log Size : " << logSize << std::endl;

        std::cerr << "OptiX Pipeline Creation Log: " << log << std::endl;
    }

    std::cout << "Module created successfully." << std::endl;

    OptixProgramGroup raygenProgramGroup;
    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module = module;
    raygenDesc.raygen.entryFunctionName = "__raygen__rg";

    OptixProgramGroupOptions programGroupOptions = {};

    log[0] = '\0';         // Clear the log buffer
    logSize = sizeof(log); // Reset the log size
    OPTIX_CHECK(optixProgramGroupCreate(context, &raygenDesc, 1, &programGroupOptions, log, &logSize, &raygenProgramGroup));

    if (logSize > 0)
    {

        std::cerr << "OptiX Pipeline Creation Log: " << log << std::endl;
    }

    OptixProgramGroup missProgramGroup;
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module;                  // Use the same module as the raygen program
    missDesc.miss.entryFunctionName = "__miss__ms"; // Name of the miss program in PTX

    log[0] = '\0';         // Clear the log buffer
    logSize = sizeof(log); // Reset the log size
    OPTIX_CHECK(optixProgramGroupCreate(context, &missDesc, 1, &programGroupOptions, log, &logSize, &missProgramGroup));
    std::cout << "Miss program group created successfully." << std::endl;

    if (logSize > 0)
    {
        std::cerr << "OptiX Pipeline Creation Log: " << log << std::endl;
    }

    OptixProgramGroup hitgroupProgramGroup;
    OptixProgramGroupDesc hitgroupDesc = {};
    hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupDesc.hitgroup.moduleCH = module;                        // Closest-hit program module
    hitgroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch"; // Closest-hit program in PTX
    hitgroupDesc.hitgroup.moduleAH = nullptr;                       // No any-hit program
    hitgroupDesc.hitgroup.entryFunctionNameAH = nullptr;            // No any-hit program
    hitgroupDesc.hitgroup.moduleIS = nullptr;                       // No intersection program
    hitgroupDesc.hitgroup.entryFunctionNameIS = nullptr;            // No intersection program

    log[0] = '\0';         // Clear the log buffer
    logSize = sizeof(log); // Reset the log size
    OPTIX_CHECK(optixProgramGroupCreate(context, &hitgroupDesc, 1, &programGroupOptions, log, &logSize, &hitgroupProgramGroup));
    std::cout << "Hit group program group created successfully." << std::endl;

    if (logSize > 0)
    {
        std::cerr << "OptiX Pipeline Creation Log: " << log << std::endl;
    }

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
    log[0] = '\0';         // Clear the log buffer
    logSize = sizeof(log); // Reset the log size
    OPTIX_CHECK(optixPipelineCreate(context, &pipelineCompileOptions, &pipelineLinkOptions, programGroups, 3, log, &logSize, &pipeline));

    if (logSize > 0)
    {
        std::cout << "Log Size : " << logSize << std::endl;
        std::cerr << "OptiX Pipeline Creation Log: " << log << std::endl;
    }
    std::cout << "Pipeline created successfully." << std::endl;

    return 0;
}

/**
 * @brief Pack all the facets into OptiX.
 */
int Collision::StartCollision(std::vector<Object *> &targetObjects)
{
    int numFacets = 0;
    for (auto object : targetObjects)
    {
        // Create geometry for each object
        numFacets += object->facets.size();
    }
    std::vector<Triangle> facets(numFacets);
    int i = 0;
    for (auto object : targetObjects)
    {
        for (auto facet : object->facets)
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
