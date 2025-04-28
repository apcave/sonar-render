#ifndef _OBJECT
#define _OBJECT

#include "ObjectCuda.hpp"
#include "Object.hpp"

class Object : public ObjectCuda
{

public:
    Object(ObjectType type);

    ~Object();
    void AddFacet(float3 v1, float3 v2, float3 v3);
    void MakeFragmentData(float frag_length);
    std::vector<float3> &GetCentroids();

private:
    std::vector<float3> centroids;
};

#endif