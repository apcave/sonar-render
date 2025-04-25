#include "CollisionShared.h"
#include "CudaUtils.cuh"

#include <optix.h>
#include <optix_device.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

extern "C" __constant__ Params params;

extern "C" __global__ void __raygen__rg()
{
    // Get the thread index
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int srcIndex = idx.x;
    const unsigned int dstIndex = idx.y;
    const unsigned int outIndex = srcIndex * params.numSrcPoints + dstIndex;
    const float3 p1 = params.vp1[srcIndex];
    const float3 p2 = params.vp2[dstIndex];

    const float3 vp12 = subtract(p2, p1);
    const float length = GetVectorLength(vp12);
    const float3 direction = DivideVector(vp12, length);
    const float epsilon = 1e-5f;

    unsigned int hit = 0;
    optixTrace(
        params.handle,
        p1,
        direction,
        0.0f,             // Min t
        length - epsilon, // Max t
        0.0f,             // Ray time
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0, // SBT offset
        1, // SBT stride
        0, // Miss SBT index
        hit);

    // Write the result to the output buffer
    params.output[outIndex] = hit;
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