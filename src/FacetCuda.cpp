#include "FacetCuda.hpp"
#include <iostream>

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

    // If the object is a source the surface pressure fixed to 1 across the surface.

    if (objectType == OBJECT_TYPE_FIELD)
    {
        std::cout << "FacetCuda: AllocateCuda: Field object." << std::endl;
        std::cout << "Number of fragment points: " << numXpnts * numYpnts << std::endl;
    }

    if (objectType == OBJECT_TYPE_TARGET || objectType == OBJECT_TYPE_FIELD)
    {
        // These buffers are used to store surface pressure values.

        // Allocate device memory for the initial pressure data
        cudaMalloc((void **)&dev_Pr, numXpnts * numYpnts * sizeof(double));
        cudaMemset(dev_Pr, 0, numXpnts * numYpnts * sizeof(double));
        cudaMalloc((void **)&dev_Pi, numXpnts * numYpnts * sizeof(double));
        cudaMemset(dev_Pi, 0, numXpnts * numYpnts * sizeof(double));
    }

    // Allocate device memory for the fragment area
    cudaMalloc((void **)&dev_frag_area, numXpnts * numYpnts * sizeof(float));

    if (objectType = OBJECT_TYPE_TARGET)
    {
        // These buffers are used for the facet to facet calculations.

        // Allocate device memory for the initial pressure data
        cudaMalloc((void **)&dev_Pr_initial, numXpnts * numYpnts * sizeof(double));
        cudaMemset(dev_Pr_initial, 0, numXpnts * numYpnts * sizeof(double));
        cudaMalloc((void **)&dev_Pi_initial, numXpnts * numYpnts * sizeof(double));
        cudaMemset(dev_Pi_initial, 0, numXpnts * numYpnts * sizeof(double));

        // Allocate device memory for the result pressure data
        cudaMalloc((void **)&dev_Pr_result, numXpnts * numYpnts * sizeof(double));
        cudaMemset(dev_Pr_result, 0, numXpnts * numYpnts * sizeof(double));
        cudaMalloc((void **)&dev_Pi_result, numXpnts * numYpnts * sizeof(double));
        cudaMemset(dev_Pi_result, 0, numXpnts * numYpnts * sizeof(double));
    }
    cudaMemcpy(dev_data, &host_facet, sizeof(dev_facet), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_frag_area, frag_area, numXpnts * numYpnts * sizeof(float), cudaMemcpyHostToDevice);
}

FacetCuda::~FacetCuda()
{
    if (dev_frag_area)
    {
        cudaFree(dev_frag_area);
        dev_frag_area = 0;
    }
    if (dev_data)
    {
        cudaFree(dev_data);
        dev_data = 0;
    }
    if (dev_Pr)
    {
        cudaFree(dev_Pr);
        dev_Pr = 0;
    }
    if (dev_Pi)
    {
        cudaFree(dev_Pi);
        dev_Pi = 0;
    }
    if (dev_Pr_initial)
    {
        cudaFree(dev_Pr_initial);
        dev_Pr_initial = 0;
    }
    if (dev_Pi_initial)
    {
        cudaFree(dev_Pi_initial);
        dev_Pi_initial = 0;
    }
    if (dev_Pr_result)
    {
        cudaFree(dev_Pr_result);
        dev_Pr_result = 0;
    }
    if (dev_Pi_result)
    {
        cudaFree(dev_Pi_result);
        dev_Pi_result = 0;
    }
}

void FacetCuda::PrintMatrix()
{
    if (dev_Pr == nullptr)
    {
        std::cout << "FacetCuda: PrintMatrix(): dev_Pr is null." << std::endl;
        return;
    }
    std::cout << "FacetCuda: PrintMatrix()" << std::endl;
    std::cout << "Number of fragment points: " << numXpnts * numYpnts << std::endl;

    double *host_Pr = new double[numXpnts * numYpnts];
    cudaMemcpy(host_Pr, dev_Pr, numXpnts * numYpnts * sizeof(double), cudaMemcpyDeviceToHost);

    for (int j = numYpnts - 1; j >= 0; j--)
    {
        for (int i = 0; i < numXpnts; i++)
        {
            printf("%.3e ", host_Pr[j * numXpnts + i]);
        }
        printf("\n");
    }
    delete[] host_Pr;
}