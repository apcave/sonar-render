#include "CollisionShared.h"
// #include "CudaUtils.cuh"

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
    const unsigned int outIndex = srcIndex * params.numDstPoints + dstIndex;
    const float3 p1 = params.vp1[srcIndex];
    const float3 p2 = params.vp2[dstIndex];

    float3 vp12;
    vp12.x = p2.x - p1.x;
    vp12.y = p2.y - p1.y;
    vp12.z = p2.z - p1.z;

    const float length = sqrtf(vp12.x * vp12.x + vp12.y * vp12.y + vp12.z * vp12.z);
    float3 direction;
    direction.x = vp12.x / length;
    direction.y = vp12.y / length;
    direction.z = vp12.z / length;
    const float epsilon = 1e-3f;

    // printf("p1: %f, %f, %f, p2: %f, %f, %f\n", p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
    //   printf("Ray origin: (%f, %f, %f), direction: (%f, %f, %f), tmin: %f, tmax: %f\n",
    //         p1.x, p1.y, p1.z, direction.x, direction.y, direction.z, 0.0f, length - epsilon);

    unsigned int hit = 0;
    optixTrace(
        params.handle,
        p1,
        direction,
        0.0f,             // Min t
        length - epsilon, // Max t
        0.0f,             // Ray time
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        0, // SBT offset
        1, // SBT stride
        0, // Miss SBT index
        hit);

    // Write the result to the output buffer
    // printf("outIndex: %u, srcIndex: %u, dstIndex: %u\n", outIndex, srcIndex, dstIndex);
    // printf("Hit: %d, out index %d\n", hit, outIndex);

    // printf("Hit: %d, Ray origin: (%f, %f, %f), direction: (%f, %f, %f), tmin: %f, tmax: %f\n",
    //        hit, p1.x, p1.y, p1.z, direction.x, direction.y, direction.z, 0.0f, length - epsilon);

    params.output[outIndex] = hit;
}

extern "C" __global__ void __closesthit__ch()
{
    printf("Hitt!\n");
    // Mark the ray as hitting geometry
    optixSetPayload_0(1);
}

extern "C" __global__ void __miss__ms()
{
    // Mark the ray as not hitting any geometry
    optixSetPayload_0(0);
}