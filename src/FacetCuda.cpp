#include "FacetCuda.hpp"

FacetCuda::FacetCuda()
{
}

void FacetCuda::AllocateCuda(float3 &normal,
                             float3 &base_point,
                             float3 &xAxis,
                             float3 &yAxis,
                             float *frag_area)
{
    frag_points.x = numXpnts;
    frag_points.y = numYpnts;
    frag_points.z = numXpntsNegative;

    dev_facet host_facet;
    host_facet.normal = normal;
    host_facet.base_point = base_point;
    host_facet.xAxis = xAxis;
    host_facet.yAxis = yAxis;
    host_facet.frag_points = frag_points;

    cudaMalloc((void **)&dev_data, sizeof(dev_facet));

    // Allocate device memory for the pressure data
    cudaMalloc((void **)&dev_Pr, numXpnts * numYpnts * sizeof(double));
    cudaMalloc((void **)&dev_Pi, numXpnts * numYpnts * sizeof(double));

    // Allocate device memory for the fragment area
    cudaMalloc((void **)&dev_frag_area, numXpnts * numYpnts * sizeof(float));

    // Allocate device memory for the initial pressure data
    cudaMalloc((void **)&dev_Pr_initial, numXpnts * numYpnts * sizeof(double));
    cudaMalloc((void **)&dev_Pi_initial, numXpnts * numYpnts * sizeof(double));

    // Allocate device memory for the result pressure data
    cudaMalloc((void **)&dev_Pr_result, numXpnts * numYpnts * sizeof(double));
    cudaMalloc((void **)&dev_Pi_result, numXpnts * numYpnts * sizeof(double));

    cudaMemcpy(dev_data, &host_facet, sizeof(dev_facet), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_frag_area, frag_area, numXpnts * numYpnts * sizeof(float), cudaMemcpyHostToDevice);
}

FacetCuda::~FacetCuda()
{
    if (dev_frag_area)
    {
        cudaFree(dev_frag_area);
    }
    if (dev_data)
    {
        cudaFree(dev_data);
    }
    if (dev_Pr)
    {
        cudaFree(dev_Pr);
    }
    if (dev_Pi)
    {
        cudaFree(dev_Pi);
    }
    if (dev_Pr_initial)
    {
        cudaFree(dev_Pr_initial);
    }
    if (dev_Pi_initial)
    {
        cudaFree(dev_Pi_initial);
    }
    if (dev_Pr_result)
    {
        cudaFree(dev_Pr_result);
    }
    if (dev_Pi_result)
    {
        cudaFree(dev_Pi_result);
    }
}