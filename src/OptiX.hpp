#ifndef _OPTIX_HPP
#define _OPTIX_HPP
#include "Object.hpp"
#include "CudaUtils.cuh"
#include "CollisionShared.h"

#include <optix.h>

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cstring>

class OptiX
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

    int *hasCollided = 0;

    CUdeviceptr raygenRecord = 0;
    CUdeviceptr missRecord = 0;
    CUdeviceptr hitgroupRecord = 0;
    CUdeviceptr d_tempBuffer = 0;

public:
    OptiX();
    ~OptiX();

    int *DoCollisions(std::vector<float3> &vp1, std::vector<float3> &vp2);

    int StartCollision(std::vector<Object *> &targetObjects);
    int TearDown();

private:
    void StartOptix();
    bool hasStarted = false;
    int MakePipeline();
    int CreateGeometry(const std::vector<Triangle> &facets);

    void FreePrams();
    void FreeGeometry();
    void FreePipeline();

    std::string readFile(const std::string &filename);
};
#endif