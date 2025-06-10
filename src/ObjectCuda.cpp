#include "ObjectCuda.hpp"

ObjectCuda::ObjectCuda()
{
}

void ObjectCuda::WriteSurfaceToGlTexture(float min_dB, float max_dB, bool render_phase)
{
    for (auto facet : facets)
    {
        facet->WriteSurface(min_dB, max_dB, render_phase);
    }
    cudaDeviceSynchronize();
}


void ObjectCuda::MakeCudaObjects()
{
    for (auto facet : facets)
    {
        facet->MakeCuda();
    }
}

void ObjectCuda::PrintSurfacePressure()
{
    for (auto facet : facets)
    {
        facet->PrintMatrix();
    }
}

dev_object ObjectCuda::MakeOptixStructArray()
{
    // Make an array of facet structs for the OptiX kernel.

    if (h_obj.numFacets == 0 && facets.size() > 0)
    {
        h_obj.objectType = objectType;
        h_obj.delta = resolution;
        h_obj.numFacets = facets.size();
        auto h_facets = new dev_facet[h_obj.numFacets]; // Gets deallocated after use.
        for (int i = 0; i < h_obj.numFacets; i++)
        {
            h_facets[i] = facets[i]->MakeOptixStruct();
        }
        cudaMalloc(&h_obj.facets, h_obj.numFacets * sizeof(dev_facet));
        cudaMemcpy(h_obj.facets, h_facets, h_obj.numFacets * sizeof(dev_facet), cudaMemcpyHostToDevice);
        delete[] h_facets;
    }
    return h_obj;
}

ObjectCuda::~ObjectCuda()
{
    if (h_obj.facets)
    {
        cudaFree(h_obj.facets);
        h_obj.facets = 0;
    }
}

void ObjectCuda::AccumulatePressure()
{
    for (auto facet : facets)
    {
        facet->AccumulatePressure();
    }
    cudaDeviceSynchronize();
}

    void ObjectCuda::PrimeReflections()
    {
        for (auto facet : facets)
        {
            facet->PrimeReflections();
        }
        cudaDeviceSynchronize();
    }
    
    void ObjectCuda::SwapOutputToInputPressure()
    {
        for (auto facet : facets)
        {
            facet->SwapOutputToInputPressure();
        }
        cudaDeviceSynchronize();
    }