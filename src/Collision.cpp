#include "Collision.hpp"
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include "OptiX/Exception.h"
#include <fstream>

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
    // if (raygenRecord)
    // {
    //     std::cout << "Destroying raygenRecord." << std::endl;
    //     cudaFree((void *)raygenRecord);
    //     raygenRecord = 0;
    // }

    // if (missRecord)
    // {
    //     std::cout << "Destroying missRecord." << std::endl;
    //     cudaFree((void *)missRecord);
    //     missRecord = 0;
    // }

    // if (hitgroupRecord)
    // {
    //     std::cout << "Destroying hitgroupRecord." << std::endl;
    //     cudaFree((void *)hitgroupRecord);
    //     hitgroupRecord = 0;
    // }

    if (context)
    {
        std::cout << "Destroying context." << std::endl;
        OPTIX_CHECK(optixDeviceContextDestroy(context));
        context = 0;
    }

    std::cout << "Deleted Collision." << std::endl;
}

int Collision::StopCollision()
{
    return 1;
    std::cout << "Stopped Collision." << std::endl;
    std::cout << "Stopping Collision." << std::endl;

    if (hasCollided)
    {
        std::cout << "Freeing hasCollided." << std::endl;
        delete[] hasCollided;
        hasCollided = 0;
    }

    FreePrams();
    return 0;
}

void Collision::FreeGeometry()
{
    if (d_vertices)
    {
        std::cout << "Freeing d_vertices." << std::endl;
        cudaFree((void *)d_vertices);
        d_vertices = 0;
    }

    if (d_outputBuffer)
    {
        std::cout << "Freeing d_outputBuffer." << std::endl;
        cudaFree((void *)d_outputBuffer);
        d_outputBuffer = 0;
    }

    if (d_tempBuffer != 0)
    {
        std::cout << "Freeing d_tempBuffer." << std::endl;
        cudaFree((void *)d_tempBuffer);
        d_tempBuffer = 0;
    }
}
void Collision::FreePrams()
{
    return;

    if (h_optix_params.vp1)
    {
        CUDA_CHECK(cudaFree(h_optix_params.vp1));
        h_optix_params.vp1 = 0;
    }

    if (h_optix_params.vp2)
    {
        CUDA_CHECK(cudaFree(h_optix_params.vp2));
        h_optix_params.vp2 = 0;
    }

    if (h_optix_params.output)
    {
        CUDA_CHECK(cudaFree(h_optix_params.output));
        h_optix_params.output = 0;
    }

    if (d_optix_params)
    {
        CUDA_CHECK(cudaFree((void *)d_optix_params));
        d_optix_params = 0;
    }
}

int *Collision::DoCollisions(std::vector<float3> &vp1, std::vector<float3> &vp2)
{
    if (hasCollided != 0)
    {
        std::cout << "Freeing hasCollided." << std::endl;
        delete[] hasCollided;
        hasCollided = 0;
    }

    // FreePrams();
    int numSrc = vp1.size();
    int numDst = vp2.size();

    h_optix_params.handle = gasHandle;
    h_optix_params.numDstPoints = numDst;

    std::cout << "Check collision." << std::endl;
    size_t numBytes = numSrc * sizeof(float3) + numDst * sizeof(float3) + numDst * numSrc * sizeof(int) + sizeof(Params);
    CUDA_CHECK(cudaMalloc((void **)&(h_optix_params.vp1), numSrc * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(h_optix_params.vp1, vp1.data(), numSrc * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&(h_optix_params.vp2), numDst * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(h_optix_params.vp2, vp2.data(), numDst * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&(h_optix_params.output), numDst * numSrc * sizeof(int)));
    CUDA_CHECK(cudaMemset(h_optix_params.output, 0, numDst * numSrc * sizeof(int)));

    CUDA_CHECK(cudaMalloc((void **)&d_optix_params, sizeof(Params)));
    CUDA_CHECK(cudaMemcpy((void *)d_optix_params, &h_optix_params, sizeof(Params), cudaMemcpyHostToDevice));

    // std::cout << "Launching Collision Program...\n";
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

    hasCollided = new int[numSrc * numDst];
    cudaMemcpy(hasCollided, h_optix_params.output, numSrc * numDst * sizeof(int), cudaMemcpyDeviceToHost);

    // std::cout << "Collision results:" << std::endl;
    // for (int i = 0; i < numSrc; i++)
    // {
    //     for (int j = 0; j < numDst; j++)
    //     {
    //         // std::cout << hasCollided[i * numDst + j] << std::endl;
    //         if (hasCollided[i * numDst + j])
    //         {
    //             std::cout << " 1";
    //         }
    //         else
    //         {
    //             std::cout << " 0";
    //         }
    //     }
    //     std::cout << std::endl;
    // }
    std::cout << "Done with collision results." << std::endl;
    // // delete[] hasCollided;
    // std::cout << "hasCollided." << std::endl;
    FreePrams();
    // delete[] hasCollided;

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
    if (context == 0)
    {
        OPTIX_CHECK(optixInit());
        OPTIX_CHECK(optixDeviceContextCreate(0, 0, &context));
        MakePipeline();
    }
}

Collision::Collision()
{
    // cuStreamCreate(&stream, CU_STREAM_DEFAULT);
}

// Function to load facets into OptiX
int Collision::CreateGeometry(const std::vector<Triangle> &facets)
{
    // if (d_vertices != 0)
    // {
    //     std::cout << "Freeing d_vertices." << std::endl;
    //     cudaFree((void *)d_vertices);
    //     d_vertices = 0;
    // }

    // if (d_outputBuffer != 0)
    // {
    //     std::cout << "Freeing d_outputBuffer." << std::endl;
    //     cudaFree((void *)d_outputBuffer);
    //     d_outputBuffer = 0;
    // }

    // if (d_tempBuffer != 0)
    // {
    //     std::cout << "Freeing d_tempBuffer." << std::endl;
    //     cudaFree((void *)d_tempBuffer);
    //     d_tempBuffer = 0;
    // }

    // Check maximum trace depth
    unsigned int maxTraceDepth = 0;
    OPTIX_CHECK(optixDeviceContextGetProperty(context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH, &maxTraceDepth, sizeof(maxTraceDepth)));

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

    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &gasBufferSizes));

    CUDA_CHECK(cudaMalloc((void **)&d_tempBuffer, gasBufferSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc((void **)&d_outputBuffer, gasBufferSizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(context, 0, &accelOptions, &buildInput, 1, d_tempBuffer,
                                gasBufferSizes.tempSizeInBytes, d_outputBuffer, gasBufferSizes.outputSizeInBytes,
                                &gasHandle, NULL, 0));

    // CUDA_CHECK(cudaFree((void *)d_tempBuffer));
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

    log[0] = '\0';         // Clear the log buffer
    logSize = sizeof(log); // Reset the log size
    OPTIX_CHECK(optixModuleCreateFromPTX(context, &moduleCompileOptions, &pipelineCompileOptions, dev_prog, dev_prog_sz, log, &logSize, &module));

    // if (logSize > 0)
    // {
    //     std::cerr << "OptiX Pipeline Creation Log: " << log << std::endl;
    // }

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

    // if (logSize > 0)
    // {

    //     std::cerr << "OptiX Pipeline Creation Log: " << log << std::endl;
    // }

    OptixProgramGroup missProgramGroup;
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module;                  // Use the same module as the raygen program
    missDesc.miss.entryFunctionName = "__miss__ms"; // Name of the miss program in PTX

    log[0] = '\0';         // Clear the log buffer
    logSize = sizeof(log); // Reset the log size
    OPTIX_CHECK(optixProgramGroupCreate(context, &missDesc, 1, &programGroupOptions, log, &logSize, &missProgramGroup));

    // if (logSize > 0)
    // {
    //     std::cerr << "OptiX Pipeline Creation Log: " << log << std::endl;
    // }

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

    // if (logSize > 0)
    // {
    //     std::cerr << "OptiX Pipeline Creation Log: " << log << std::endl;
    // }

    struct RaygenRecord
    {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    RaygenRecord rgRecord = {};

    // OPTIX_CHECK(optixSbtRecordPackHeader(raygenProgramGroup, &rgRecord));

    CUDA_CHECK(cudaMalloc((void **)&raygenRecord, sizeof(RaygenRecord)));

    // Copy the ray generation record from the host to the device
    cudaMemcpy((void *)raygenRecord, &rgRecord, sizeof(RaygenRecord), cudaMemcpyHostToDevice);

    sbt.raygenRecord = raygenRecord;

    struct MissRecord
    {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    MissRecord msRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(missProgramGroup, &msRecord));

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

    cudaMalloc((void **)&hitgroupRecord, sizeof(HitgroupRecord));
    cudaMemcpy((void *)hitgroupRecord, &hgRecord, sizeof(HitgroupRecord), cudaMemcpyHostToDevice);
    sbt.hitgroupRecordBase = hitgroupRecord;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = 1;

    OptixProgramGroup programGroups[] = {raygenProgramGroup, missProgramGroup, hitgroupProgramGroup};
    log[0] = '\0';         // Clear the log buffer
    logSize = sizeof(log); // Reset the log size
    OPTIX_CHECK(optixPipelineCreate(context, &pipelineCompileOptions, &pipelineLinkOptions, programGroups, 3, log, &logSize, &pipeline));

    // if (logSize > 0)
    // {
    //     std::cerr << "OptiX Pipeline Creation Log: " << log << std::endl;
    // }

    return 0;
}

/**
 * @brief Pack all the facets into OptiX.
 */
int Collision::StartCollision(std::vector<Object *> &targetObjects)
{
    if (!hasStarted)
    {
        std::cout << "Starting OptiX...***********************************************" << std::endl;
        StartOptix();
        hasStarted = true;
    }
    std::cout << "Starting Collision....." << OPTIX_SBT_RECORD_HEADER_SIZE << std::endl;

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
