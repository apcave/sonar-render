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

    host_facet.normal = normal;
    host_facet.base_point = base_point;
    host_facet.xAxis = xAxis;
    host_facet.yAxis = yAxis;
    host_facet.frag_points = frag_points;

    cudaMalloc((void **)&dev_data, sizeof(dev_facet));

    // If the object is a source the surface pressure fixed to 1 across the surface.

    dev_P = 0;
    if (objectType == OBJECT_TYPE_TARGET || objectType == OBJECT_TYPE_FIELD)
    {
        // These buffers are used to store surface pressure values.

        // Allocate device memory for the initial pressure data
        cudaMalloc((void **)&dev_P, numXpnts * numYpnts * sizeof(dcomplex));
        cudaMemset(dev_P, 0, numXpnts * numYpnts * sizeof(dcomplex));

        host_facet.P = dev_P;
        host_facet.P_out = dev_P; // The working buffer is not need for field objects.
    }

    // Allocate device memory for the fragment area
    cudaMalloc((void **)&dev_frag_area, numXpnts * numYpnts * sizeof(float));

    dev_P_out = 0;
    if (objectType == OBJECT_TYPE_TARGET)
    {
        // These buffers are used for the facet to facet calculations.

        // Allocate device memory for the initial pressure data
        cudaMalloc((void **)&dev_P_out, numXpnts * numYpnts * sizeof(dcomplex));
        cudaMemset(dev_P_out, 0, numXpnts * numYpnts * sizeof(dcomplex));
        host_facet.P_out = dev_P_out;
    }

    cudaMemcpy(dev_data, &host_facet, sizeof(dev_facet), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_frag_area, frag_area, numXpnts * numYpnts * sizeof(float), cudaMemcpyHostToDevice);

    host_facet.frag_area = dev_frag_area;
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
    if (dev_P)
    {
        cudaFree(dev_P);
        dev_P = 0;
    }

    if (dev_P_out)
    {
        cudaFree(dev_P_out);
        dev_P_out = 0;
    }
}

void FacetCuda::PrintMatrix()
{
    if (dev_P)
    {
        std::cout << "FacetCuda: PrintMatrix(): dev_Pr is null." << std::endl;
        return;
    }
    std::cout << "FacetCuda: PrintMatrix()" << std::endl;
    std::cout << "Number of fragment points: " << numXpnts * numYpnts << std::endl;

    dcomplex *host_P = new dcomplex[numXpnts * numYpnts];
    cudaMemcpy(host_P, dev_P, numXpnts * numYpnts * sizeof(dcomplex), cudaMemcpyDeviceToHost);

    for (int j = numYpnts - 1; j >= 0; j--)
    {
        for (int i = 0; i < numXpnts; i++)
        {
            printf("%.3e ", host_P[j * numXpnts + i].r);
        }
        printf("\n");
    }
    delete[] host_P;
}

dev_facet FacetCuda::MakeOptixStruct()
{
    // Makes a copy.
    if (objectType == OBJECT_TYPE_TARGET)
    {
        cudaMemset(host_facet.P_out, 0, numXpnts * numYpnts * sizeof(dcomplex));
    }
    return host_facet;
}
