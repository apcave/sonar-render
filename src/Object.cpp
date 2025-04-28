#include "Object.hpp"

Object::Object(ObjectType type)
{
    objectType = type;
}

Object::~Object()
{
}

void Object::AddFacet(float3 v1, float3 v2, float3 v3)
{
    // Add the facet to the target object
    // This is a placeholder for the actual implementation
    auto facet = new Facet(v1, v2, v3, objectType);
    facets.push_back(facet);
}

void Object::MakeFragmentData(float frag_length)
{
    // Iterate over each facet and call the pixelation function
    for (auto &facet : facets)
    {
        // Call the pixelation function on each facet
        facet->MakeFragmentData(frag_length);
        // facet->PrintMatrix();
    }
}

std::vector<float3> &Object::GetCentroids()
{
    if (centroids.size() == facets.size())
    {
        return centroids;
    }
    else
    {
        centroids.clear();
        centroids.reserve(facets.size());
        for (auto &facet : facets)
        {
            centroids.push_back(facet->Centroid);
        }
    }
    return centroids;
}