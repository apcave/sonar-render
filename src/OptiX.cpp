#include "OptiX.hpp"
#include "OptiX/Exception.h"
#include <stdexcept>
#include <optix_function_table_definition.h>

std::string OptiX::readFile(const std::string &filename)
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

OptiX::~OptiX()
{
}

int OptiX::TearDown()
{
    if (hasCollided)
    {
        std::cout << "Freeing hasCollided." << std::endl;
        delete[] hasCollided;
        hasCollided = 0;
    }

    FreePrams();
    FreeGeometry();
    return 0;

    FreePipeline();
    if (context)
    {
        std::cout << "Destroying context." << std::endl;

        // This cannot be called from the destructor!
        OPTIX_CHECK(optixDeviceContextDestroy(context));
        context = 0;
    }
    hasStarted = false;

    std::cout << "Deleted OptiX." << std::endl;
    return 0;
}

void OptiX::FreePipeline()
{
    if (raygenRecord)
    {
        std::cout << "Destroying raygenRecord." << std::endl;
        cudaFree((void *)raygenRecord);
        raygenRecord = 0;
    }

    if (missRecord)
    {
        std::cout << "Destroying missRecord." << std::endl;
        cudaFree((void *)missRecord);
        missRecord = 0;
    }

    if (hitgroupRecord)
    {
        std::cout << "Destroying hitgroupRecord." << std::endl;
        cudaFree((void *)hitgroupRecord);
        hitgroupRecord = 0;
    }

    return;
}

void OptiX::FreeGeometry()
{
    if (d_vertices)
    {
        // std::cout << "Freeing d_vertices." << std::endl;
        cudaFree((void *)d_vertices);
        d_vertices = 0;
    }

    if (d_outputBuffer)
    {
        // std::cout << "Freeing d_outputBuffer." << std::endl;
        cudaFree((void *)d_outputBuffer);
        d_outputBuffer = 0;
    }

    if (d_tempBuffer)
    {
        // std::cout << "Freeing d_tempBuffer." << std::endl;
        cudaFree((void *)d_tempBuffer);
        d_tempBuffer = 0;
    }
}
void OptiX::FreePrams()
{
    // if (h_optix_params.vp1)
    // {
    //     CUDA_CHECK(cudaFree(h_optix_params.vp1));
    //     h_optix_params.vp1 = 0;
    // }

    // if (h_optix_params.vp2)
    // {
    //     CUDA_CHECK(cudaFree(h_optix_params.vp2));
    //     h_optix_params.vp2 = 0;
    // }

    // if (h_optix_params.output)
    // {
    //     CUDA_CHECK(cudaFree(h_optix_params.output));
    //     h_optix_params.output = 0;
    // }

    // if (d_optix_params)
    // {
    //     CUDA_CHECK(cudaFree((void *)d_optix_params));
    //     d_optix_params = 0;
    // }
}

void OptiX::DoProjection(globalParams params)
{
    // The params are passed to the device.
    // The results are in CUDA memory.

    if (d_optix_params)
    {
        throw new std::runtime_error("OptiX::DoProjection: d_optix_params already allocated.");
    }
    std::cout << "OptiX::DoProjection()" << std::endl;
    params.handle = gasHandle;
    CUDA_CHECK(cudaMalloc((void **)&d_optix_params, sizeof(globalParams)));
    CUDA_CHECK(cudaMemcpy((void *)d_optix_params, &params, sizeof(globalParams), cudaMemcpyHostToDevice));

    int numFacets = params.srcObject.numFacets;
    if (params.calcType == SOURCE_POINTS)
    {
        numFacets = params.dstObject.numFacets;
    }
    std::cout << "Num facets: " << numFacets << std::endl;
    int numberToProcess = numFacets;
    // int max_ind = 1024;
    int max_ind = 9216;
    int num_ind = max_ind;
    while (numberToProcess != 0)
    {
        if (numberToProcess > max_ind)
        {
            num_ind = max_ind;
            numberToProcess -= max_ind;
        }
        else
        {
            num_ind = numberToProcess;
            numberToProcess = 0;
        }

        std::cout << "Launching Collision Program...\n";
        OPTIX_CHECK(optixLaunch(
            pipeline,
            0,
            d_optix_params,
            sizeof(globalParams),
            &sbt,
            numFacets,
            1,
            1));
        // cudaDeviceSynchronize();
    }
    std::cout << "Launched Collision Program.\n";

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    CUDA_CHECK(cudaFree((void *)d_optix_params));
    d_optix_params = 0;
}

int *OptiX::DoCollisions(std::vector<float3> &vp1, std::vector<float3> &vp2)
{
    // if (hasCollided)
    // {
    //     std::cout << "Freeing hasCollided." << std::endl;
    //     delete[] hasCollided;
    //     hasCollided = 0;
    // }

    // // FreePrams();
    // int numSrc = vp1.size();
    // int numDst = vp2.size();
    // hasCollided = new int[numSrc * numDst];

    // h_optix_params.handle = gasHandle;
    // h_optix_params.numDstPoints = numDst;

    // size_t numBytes = numSrc * sizeof(float3) + numDst * sizeof(float3) + numDst * numSrc * sizeof(int) + sizeof(Params);
    // CUDA_CHECK(cudaMalloc((void **)&(h_optix_params.vp1), numSrc * sizeof(float3)));
    // CUDA_CHECK(cudaMemcpy(h_optix_params.vp1, vp1.data(), numSrc * sizeof(float3), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMalloc((void **)&(h_optix_params.vp2), numDst * sizeof(float3)));
    // CUDA_CHECK(cudaMemcpy(h_optix_params.vp2, vp2.data(), numDst * sizeof(float3), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMalloc((void **)&(h_optix_params.output), numDst * numSrc * sizeof(int)));
    // CUDA_CHECK(cudaMemset(h_optix_params.output, 0, numDst * numSrc * sizeof(int)));

    // CUDA_CHECK(cudaMalloc((void **)&d_optix_params, sizeof(Params)));
    // CUDA_CHECK(cudaMemcpy((void *)d_optix_params, &h_optix_params, sizeof(Params), cudaMemcpyHostToDevice));

    // // std::cout << "Launching Collision Program...\n";
    // OPTIX_CHECK(optixLaunch(
    //     pipeline,
    //     0,
    //     d_optix_params,
    //     sizeof(Params),
    //     &sbt,
    //     numSrc,
    //     numDst,
    //     1));
    // // std::cout << "Launched Collision Program.\n";

    // cudaDeviceSynchronize();
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    // }
    // cudaMemcpy(hasCollided, h_optix_params.output, numSrc * numDst * sizeof(int), cudaMemcpyDeviceToHost);

    // // std::cout << "Collision results:" << std::endl;
    // // for (int i = 0; i < numSrc; i++)
    // // {
    // //     for (int j = 0; j < numDst; j++)
    // //     {
    // //         // std::cout << hasCollided[i * numDst + j] << std::endl;
    // //         if (hasCollided[i * numDst + j])
    // //         {
    // //             std::cout << " 1";
    // //         }
    // //         else
    // //         {
    // //             std::cout << " 0";
    // //         }
    // //     }
    // //     std::cout << std::endl;
    // // }
    // // std::cout << "Done with collision results." << std::endl;

    // FreePrams();
    // return hasCollided;
    return nullptr;
}

void OptiX::StartOptix()
{
    std::cout << "OptiX::StartOptix()" << std::endl;
    TearDown();
    if (context == 0)
    {
        OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions options = {};
        // options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
        OPTIX_CHECK(optixDeviceContextCreate(0, &options, &context));
        MakePipeline();
    }
}

/**
 * @brief Pack all the facets into OptiX.
 */
int OptiX::StartCollision(std::vector<Object *> &targets)
{
    if (!hasStarted)
    {
        std::cout << "Starting OptiX...***********************************************" << std::endl;
        StartOptix();
        hasStarted = true;
    }

    int numFacets = 0;
    for (auto object : targets)
    {
        // Create geometry for each object
        numFacets += object->facets.size();
    }

    std::vector<Triangle> facets(numFacets);
    int i = 0;
    for (auto object : targets)
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
    std::cout << "Loaded Collision Geometry into OptiX." << std::endl;

    return 0;
}

OptiX::OptiX()
{
    // cuStreamCreate(&stream, CU_STREAM_DEFAULT);
}

// Function to load facets into OptiX
int OptiX::CreateGeometry(const std::vector<Triangle> &facets)
{
    // std::cout << "Creating geometry... NumFacets :" << facets.size() << std::endl;
    FreeGeometry();

    // Check maximum trace depth
    unsigned int maxTraceDepth = 0;
    OPTIX_CHECK(optixDeviceContextGetProperty(context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH, &maxTraceDepth, sizeof(maxTraceDepth)));
    // // Allocate device memory for vertices
    size_t verticesSize = facets.size() * 3 * sizeof(float3);
    CUDA_CHECK(cudaMalloc((void **)&d_vertices, verticesSize));
    CUDA_CHECK(cudaMemcpy((void *)d_vertices, facets.data(), verticesSize, cudaMemcpyHostToDevice));

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
    // accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
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

int OptiX::MakePipeline()
{
    // Check maximum trace depth
    unsigned int maxTraceDepth = 0;
    OPTIX_CHECK(optixDeviceContextGetProperty(context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH, &maxTraceDepth, sizeof(maxTraceDepth)));

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

    std::string device_programs = readFile("./build/OptiX.ptx");
    const char *dev_prog = device_programs.c_str();
    size_t dev_prog_sz = device_programs.size();
    std::cout << "Device program size: " << dev_prog_sz << std::endl;

    log[0] = '\0';         // Clear the log buffer
    logSize = sizeof(log); // Reset the log size
    OPTIX_CHECK(optixModuleCreateFromPTX(context, &moduleCompileOptions, &pipelineCompileOptions, dev_prog, dev_prog_sz, log, &logSize, &module));

    if (logSize > 0)
    {
        std::cerr << "Load PTX file Log: " << log << std::endl;
    }

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
        std::cerr << "__raygen__rg Creation Log: " << log << std::endl;
    }

    OptixProgramGroup missProgramGroup;
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module;                  // Use the same module as the raygen program
    missDesc.miss.entryFunctionName = "__miss__ms"; // Name of the miss program in PTX

    log[0] = '\0';         // Clear the log buffer
    logSize = sizeof(log); // Reset the log size
    OPTIX_CHECK(optixProgramGroupCreate(context, &missDesc, 1, &programGroupOptions, log, &logSize, &missProgramGroup));

    if (logSize > 0)
    {
        std::cerr << "__miss__ms Creation Log: " << log << std::endl;
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

    if (logSize > 0)
    {
        std::cerr << "__closesthit__ch Creation Log: " << log << std::endl;
    }

    struct RaygenRecord
    {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    RaygenRecord rgRecord = {};

    OPTIX_CHECK(optixSbtRecordPackHeader(raygenProgramGroup, &rgRecord));

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

    if (logSize > 0)
    {
        std::cerr << "OptiX Pipeline Creation Log: " << log << std::endl;
    }

    std::cout << "OptiX Pipeline created successfully." << std::endl;
    return 0;
}
