#ifndef COLLISION_SHARED_H
#define COLLISION_SHARED_H
#include <optix.h>

typedef struct
{
    OptixTraversableHandle handle; // Acceleration structure handle
    int *output;                   // Output buffer for results
    float3 *vp1;                   // Source Position
    float3 *vp2;                   // Destination Position (1024 max)
    int numSrcPoints;              // Number of source points
} Params;

#endif