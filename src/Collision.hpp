#ifndef _COLISSION_HPP
#define _COLISSION_HPP
#include "Object.hpp"
#include "CudaUtils.cuh"
#include "CollisionShared.h"

#include <optix.h>

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cstring>

class Collision
{
    struct Triangle
    {
        float3 v0, v1, v2;
    };

private:
    Params h_optix_params;
    CUdeviceptr d_optix_params = 0;

    OptixTraversableHandle gasHandle;
    OptixPipeline pipeline = 0;
    OptixShaderBindingTable sbt = {};

    OptixDeviceContext context = 0;
    CUdeviceptr d_outputBuffer = 0;
    CUdeviceptr d_vertices = 0;

public:
    Collision();
    ~Collision();
    void StartOptix();

    bool HasCollision(float3 p1, float3 p2);

    // bool CheckCollision(float3 p1, float3 p2);
    int StartCollision(std::vector<Object *> &targetObjects);
    int StopCollision();

private:
    int MakePipeline();
    int CreateGeometry(const std::vector<Triangle> &facets);
    int MakeShaderBufferTable();

    void FreePrams();
    void DoCollisions(std::vector<float3> vp1, std::vector<float3> vp2);
    std::string readFile(const std::string &filename);
};
#endif