#include "FacetCuda.hpp"

FacetCuda::FacetCuda()
{
}

void FacetCuda::allocateCuda(float3 &normal,
                             float3 &base_point,
                             float3 &xAxis,
                             float3 &yAxis,
                             float *frag_area)
{
    frag_points.x = numXpnts;
    frag_points.y = numYpnts;
    frag_points.z = numXpntsNegative;
    cudaMalloc((void **)&dev_frag_points, sizeof(int3));

    // Allocate device memory for the facet data
    cudaMalloc((void **)&dev_facet_normal, sizeof(float3));
    cudaMalloc((void **)&dev_base_point, sizeof(float3));
    cudaMalloc((void **)&dev_xAxis, sizeof(float3));
    cudaMalloc((void **)&dev_yAxis, sizeof(float3));

    // Allocate device memory for the pressure data
    cudaMalloc((void **)&dev_Pr, numXpnts * numYpnts * sizeof(double));
    cudaMalloc((void **)&dev_Pi, numXpnts * numYpnts * sizeof(double));

    // Allocate device memory for the fragment area
    cudaMalloc((void **)&dev_frag_area, numXpnts * numYpnts * sizeof(double));

    // Allocate device memory for the initial pressure data
    cudaMalloc((void **)&dev_Pr_initial, numXpnts * numYpnts * sizeof(double));
    cudaMalloc((void **)&dev_Pi_initial, numXpnts * numYpnts * sizeof(double));

    // Allocate device memory for the result pressure data
    cudaMalloc((void **)&dev_Pr_result, numXpnts * numYpnts * sizeof(double));
    cudaMalloc((void **)&dev_Pi_result, numXpnts * numYpnts * sizeof(double));

    cudaMemcpy(dev_frag_points, &frag_points, sizeof(int3), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_facet_normal, &normal, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_base_point, &base_point, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_xAxis, &xAxis, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_yAxis, &yAxis, sizeof(float3), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_frag_area, frag_area, numXpnts * numYpnts * sizeof(float), cudaMemcpyHostToDevice);
}