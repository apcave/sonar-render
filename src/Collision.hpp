#ifndef _COLISSION_HPP
#define _COLISSION_HPP
#include "Object.hpp"
#include "CudaUtils.cuh"
#include "CollisionShared.h"

#include <optix.h>

#include <cuda_runtime.h>
#include <vector>

class Collision
{
    struct Triangle
    {
        float3 v0, v1, v2;
    };

private:
    Params h_optix_params;
    Params *d_optix_params = 0;
    OptixDeviceContext optxContext = 0;
    CUdeviceptr d_optxVertices = 0;
    OptixTraversableHandle gasHandle = 0;
    OptixPipeline pipeline = 0;
    CUstream stream = 0;
    OptixShaderBindingTable sbt = {};

public:
    Collision();
    ~Collision();

    bool HasCollision(float3 p1, float3 p2);

    // bool CheckCollision(float3 p1, float3 p2);
    int StartCollision(std::vector<Object> &targetObjects);

private:
    int StopCollision();
    int MakePipeline();
    int CreateGeometry(const std::vector<Triangle> &facets);
    int MakeShaderBufferTable();

    void FreePrams();
    void DoCollisions(std::vector<float3> vp1, std::vector<float3> vp2);
};
#endif