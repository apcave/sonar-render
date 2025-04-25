#include <optix.h>
#include <cuda_runtime.h>

typedef struct
{
    OptixTraversableHandle handle;
    float3 ray_origin;
    float3 ray_direction;
    int *output;
} Params;

extern "C" __constant__ Params params;

extern "C" __global__ void __raygen__rg()
{
    unsigned int payload = 0;

    optixTrace(
        params.handle,
        params.ray_origin,
        params.ray_direction,
        0.0f,  // Min t
        1e16f, // Max t
        0.0f,  // Ray time
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0, // SBT offset
        1, // SBT stride
        0, // Miss SBT index
        payload);

    *params.output = payload;
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(0); // No collision
}

extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(1); // Collision detected
}