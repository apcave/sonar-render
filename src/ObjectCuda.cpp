#include "ObjectCuda.hpp"

ObjectCuda::ObjectCuda()
{
}

ObjectCuda::~ObjectCuda()
{
}

void ObjectCuda::WriteSurfaceToGlTexture(float *dev_frag_stats)
{
    for (auto facet : facets)
    {
        facet->WriteSurface(dev_frag_stats);
    }
    cudaDeviceSynchronize();
}

void ObjectCuda::GetSurfaceScalers(float *dev_frag_stats)
{
    for (auto facet : facets)
    {
        facet->GetSurfaceScalers(dev_frag_stats);
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